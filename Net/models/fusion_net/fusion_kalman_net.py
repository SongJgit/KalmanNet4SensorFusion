import math
import typing
from math import floor
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
from torch import Tensor

from Net.params.make_params import wraptopi
from Net.utils import PARAMS

from ..kalman_net.kalman_net import KalmanNet
from ..kalman_net.skalman_net import SplitKalmanNet
# mypy: ignore-errors


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

class FusionKalmanNet(KalmanNet):

    def init_KGainNet(self,
                      prior_Q: torch.Tensor | None = None,
                      prior_Sigma: torch.Tensor | None = None,
                      prior_S: torch.Tensor | None = None) -> None:
        if prior_Q is not None:
            self.prior_Q = prior_Q
            self.prior_Sigma = prior_Sigma
            self.prior_S = prior_S
        else:
            self.prior_Q = torch.eye(self.state_dim)
            self.prior_Sigma = torch.zeros((self.state_dim, self.state_dim))
            self.prior_S = torch.eye(self.observation_dim)
        # GRU to track Q
        self.d_input_Q = self.state_dim * self.in_mult_KNet
        self.d_hidden_Q = self.state_dim**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.state_dim * self.in_mult_KNet
        self.d_hidden_Sigma = self.state_dim**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)

        # GRU to track S
        self.d_input_S = self.observation_dim**2 + 2 * self.observation_dim * self.in_mult_KNet
        self.d_hidden_S = self.observation_dim**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)

        # Fully connected 1 for GRU_Sigma
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.observation_dim**2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU())

        # Fully connected 2 for GRU_
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.observation_dim * self.state_dim
        self.d_hidden_FC2 = self.d_input_FC2 * self.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2), nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.state_dim**2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU())

        # Fully connected 5
        self.d_input_FC5 = self.state_dim
        self.d_output_FC5 = self.state_dim * self.in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.state_dim
        self.d_output_FC6 = self.state_dim * self.in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU())

        # Fully connected 7
        self.d_input_FC7 = 2 * self.observation_dim
        self.d_output_FC7 = 2 * self.observation_dim * self.in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU())

        self.initialized_matrices = True

    def init_beliefs(self, state: torch.Tensor) -> None:
        """_summary_

        Args:
            state (torch.Tensor): [N, num_states]

        """
        # init hidden, like covariance sigma, innovation covar s and noise q.
        self._init_hidden(state)

        self.batch_size = state.shape[0]

        # assert self.batch_size == batch_size
        assert state.shape == (self.batch_size, self.state_dim,
                               1), f'{state.shape=}, {self.state_dim=}'

        self.m1x_posterior = state.detach().clone()
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_prior_previous = self.m1x_posterior.detach().clone()

        self.y_previous = self.params.get_obs_jac(
            self.m1x_posterior)[0].detach().clone()

        self._initialized = True

    @typing.no_type_check
    def step_prior(self, m1x_posterior: torch.Tensor,
                   sensor: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore
        batch_size = m1x_posterior.shape[
            0]  # [bs ,[x, y, x', y', theta, omega], 1]
        m1x_prior, f_jac = self.params.get_pred_jac(m1x_posterior, sensor)
        # m1x_prior = F @ tmp_tensor
        # m1x_prior[:, 4, :] = wraptopi(m1x_prior[:, 4, :])
        m1y, h_jac = self.params.get_obs_jac(m1x_prior)
        assert not torch.any(torch.isnan(m1y))
        assert not torch.any(torch.isnan(m1x_prior))
        return m1x_prior, f_jac, m1y, h_jac

    @typing.no_type_check
    def _forward_(self, sensor: Tensor,
                  correction: Tensor | None) -> torch.Tensor:  # type: ignore
        """_summary_
        Args:
            sensor (torch.Tensor): [bs, num_state, 1], [bs, [ax, ay, theta, omega], 1].
            correction (torch.Tensor): Like observation, [bs, num_state, 1], [bs, [x, y], 1].

        Returns:
            _type_: _description_
        """
        m1x_prior, f_jac, m1y, h_jac = self.step_prior(self.m1x_posterior,
                                                       sensor)

        # NOTE: if not correction, the self.correction is equal to self.y_previous
        mask = torch.isnan(correction)
        y = torch.zeros_like(self.y_previous)
        y[mask] = m1y[mask]
        y[~mask] = correction[~mask]

        # Compute Kalman Net input
        obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff = self.step_KGain_est(
            y, m1y, self.y_previous, self.m1x_posterior,
            self.m1x_posterior_previous, self.m1x_prior_previous)
        # Kalman Gain Network Step
        kalman_gain = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff,
                                      fw_update_diff).reshape(
                                          self.batch_size, self.state_dim,
                                          self.observation_dim)

        innov = y - m1y
        m1x_posterior = torch.baddbmm(m1x_prior, kalman_gain, innov)

        m1x_posterior = self.params.convert(m1x_posterior)
        self.innovation = innov.detach().clone()
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        self.y_previous = y.detach().clone()
        # self.innovation = innov
        # self.m1x_posterior_previous = self.m1x_posterior
        # self.m1x_posterior = m1x_posterior
        # self.m1x_prior_previous = m1x_prior
        # self.y_previous =y
        assert not torch.any(torch.isnan(m1x_posterior))
        return m1x_posterior.clone()

    @typing.no_type_check
    def forward(self, sensor: torch.Tensor,
                correction: torch.Tensor) -> torch.Tensor:  # type: ignore
        """_summary_
        Args:
            sensor (torch.Tensor): [bs, num_state, 1], [bs, [ax, ay, theta, omega], 1].
            correction (torch.Tensor): Like observation, [bs, num_state, 1], [bs, [x, y], 1].

        Returns:
            _type_: _description_
        """
        return self._forward_(sensor, correction)


class FusionSplitKalmanNet(SplitKalmanNet):

    @typing.no_type_check
    def predict(self, m1x_posterior: torch.Tensor,
                sensor: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore
        """_summary_

        Args:
            m1x_posterior (torch.Tensor): # [bs ,[x, y, x', y', theta, omega], 1]
            sensor (torch.Tensor): _description_

        Returns:
            Tuple[Any, ...]: _description_
        """

        m1x_prior, f_jac = self.params.get_pred_jac(m1x_posterior, sensor)
        m1y, h_jac = self.params.get_obs_jac(m1x_prior)
        # assert not torch.any(torch.isnan(m1y))
        assert not torch.any(torch.isnan(m1x_prior))
        return m1x_prior, f_jac, m1y, h_jac

    @typing.no_type_check
    def forward(self, sensor: torch.Tensor,
                correction: torch.Tensor) -> torch.Tensor:  # type: ignore
        """_summary_
        Args:
            sensor (torch.Tensor): [bs, num_state, 1], [bs, [ax, ay, theta, omega], 1].
            correction (torch.Tensor): Like observation, [bs, num_state, 1], [bs, [x, y], 1].
        Returns:
            _type_: _description_
        """
        m1x_prior, f_jac, m1y, h_jac = self.predict(self.m1x_posterior, sensor)

        # NOTE: if the correction is nan, the self.y is equal to self.y_previous
        # update mask.
        mask = torch.isnan(correction)
        y = torch.zeros_like(self.y_previous)
        y[mask] = m1y[mask]
        y[~mask] = correction[~mask]

        # input1
        state_inno = self.m1x_posterior_previous - self.m1x_prior_previous
        # input2 innovation

        # TODO:
        innov = torch.zeros_like(y)
        innov = y - m1y
        # input3
        diff_state = self.m1x_posterior - self.m1x_posterior_previous
        # input4
        diff_obs = y - self.y_previous
        # input5
        linear_error = m1y - h_jac @ m1x_prior

        Pk, Sk = self._forward_(state_inno, innov, diff_state, diff_obs,
                                linear_error, h_jac, f_jac)
        kalman_gain = Pk @ torch.transpose(h_jac, -1, -2) @ Sk
        m1x_posterior = torch.baddbmm(m1x_prior, kalman_gain, innov)
        m1x_posterior = self.params.convert(m1x_posterior)
        # not correction, use predict, not update.
        # m1x_posterior[mask[:, 0, :].flatten()] = m1x_prior[mask[:, 0, :].flatten()]
        self.innovation = innov
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        self.y_previous = y.detach().clone()
        assert not torch.any(torch.isnan(m1x_posterior))
        return m1x_posterior.clone()

