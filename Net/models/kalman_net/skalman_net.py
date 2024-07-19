import typing
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func
import torch.nn.init as init
from mmengine.config import Config
from torch import Tensor

from Net.utils import MODELS, PARAMS


class SplitKalmanNet(nn.Module):
    # origin split
    def __init__(self,
                 params: Dict[str, Any] | Any,
                 gru_scale_s: int = 2,
                 nGRU: int = 2,
                 device: str = None):
        super().__init__()
        # self.batch_size = batch_size
        self.device = device
        # init params
        if isinstance(params, dict):
            self.params = PARAMS.build(params)
        else:
            self.params = params

        self.state_dim = self.params.state_dim
        self.observation_dim = self.params.observation_dim
        self.gru_scale_s = gru_scale_s
        self.nGRU = nGRU

        self._initialized = False

        self.seq_len_input = 1
        self._init_net()

    def _init_net(self):
        H1 = (self.state_dim + self.observation_dim) * (10) * 8
        H2 = (self.state_dim * self.observation_dim) * 1 * (4)
        self.input_dim_1 = (self.state_dim) * 2 + self.observation_dim + (
            self.state_dim * self.observation_dim)
        self.input_dim_2 = (self.observation_dim) * 3 + (self.state_dim *
                                                         self.observation_dim)
        self.output_dim_1 = (self.state_dim * self.state_dim)
        self.output_dim_2 = self.observation_dim**2

        self.gru_hidden_dim = round(self.gru_scale_s *
                                    ((self.state_dim**2) +
                                     (self.observation_dim**2)))
        self.l1 = nn.Sequential(nn.Linear(self.input_dim_1, H1), nn.ReLU())
        # GRU
        self.gru_input_dim = H1
        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.nGRU)
        self.l2 = nn.Sequential(nn.Linear(self.gru_hidden_dim, H2), nn.ReLU(),
                                nn.Linear(H2, self.output_dim_1))
        self.l3 = nn.Sequential(nn.Linear(self.input_dim_2, H1), nn.ReLU())
        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.nGRU)

        # GRU output -> H2 -> Sk
        self.l4 = nn.Sequential(nn.Linear(self.gru_hidden_dim, H2), nn.ReLU(),
                                nn.Linear(H2, self.output_dim_2))

    def _init_hidden(self, state: torch.Tensor):
        """
            init hidden state of the new sequence.

        Args:
            device (_type_): _description_
        """
        device = state.device
        batch_size = state.shape[0]
        self.h_Sigma = torch.randn(self.nGRU,
                                   batch_size,
                                   self.gru_hidden_dim,
                                   device=device)
        self.h_S = torch.randn(self.nGRU,
                               batch_size,
                               self.gru_hidden_dim,
                               device=device)

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
    def predict(
            self,
            m1x_posterior: torch.Tensor) -> Tuple[Any, ...]:  # type: ignore
        """_summary_

        Args:
            m1x_posterior (torch.Tensor): # [bs ,[x, y, x', y', theta, omega], 1]
            sensor (torch.Tensor): _description_

        Returns:
            Tuple[Any, ...]: _description_
        """

        m1x_prior, f_jac = self.params.get_pred_jac(m1x_posterior)
        m1y, h_jac = self.params.get_obs_jac(m1x_prior)
        # assert not torch.any(torch.isnan(m1y))
        assert not torch.any(torch.isnan(m1x_prior))
        return m1x_prior, f_jac, m1y, h_jac

    def _forward_(self, state_inno, observation_inno, diff_state, diff_obs,
                  linear_error, h_jac, f_jac):
        batch_size = state_inno.shape[0]
        # f_jac = f_jac.reshape(batch_size, -1, 1)
        h_jac = h_jac.reshape(batch_size, -1, 1)
        # #[bs, num_state,1] -> [1, bs, num_state]
        input1 = torch.cat((state_inno, diff_state, linear_error, h_jac),
                           dim=1).permute(2, 0, 1)
        input2 = torch.cat((observation_inno, diff_obs, linear_error, h_jac),
                           dim=1).permute(2, 0, 1)

        l1_out = self.l1(input1)

        GRU_in = torch.zeros(1, self.batch_size,
                             self.gru_input_dim).to(l1_out.device)
        GRU_in[0, :, :] = l1_out
        GRU_out, self.h_Sigma = self.GRU1(GRU_in, self.h_Sigma)
        l2_out = self.l2(GRU_out)
        Pk = l2_out.reshape((batch_size, self.state_dim, self.state_dim))

        l3_out = self.l3(input2)
        GRU_in = torch.zeros(1, self.batch_size,
                             self.gru_input_dim).to(l3_out.device)
        GRU_in[0, :, :] = l3_out
        GRU_out, self.h_S = self.GRU2(GRU_in, self.h_S)
        l4_out = self.l4(GRU_out)
        Sk = l4_out.reshape(
            (batch_size, self.observation_dim, self.observation_dim))
        return Pk, Sk

    def _detach(self):
        self.h_Sigma = self.h_Sigma.detach().clone()
        self.h_S = self.h_S.detach().clone()
        self.innovation = self.innovation.detach().clone()
        self.m1x_posterior_previous = self.m1x_posterior_previous.detach(
        ).clone()
        self.m1x_posterior = self.m1x_posterior.detach().clone()
        self.m1x_prior_previous = self.m1x_prior_previous.detach().clone()
        self.y_previous = self.y_previous.detach().clone()

    @typing.no_type_check
    def forward(self, y: torch.Tensor) -> torch.Tensor:  # type: ignore
        """_summary_
        Args:
            sensor (torch.Tensor): [bs, num_state, 1], [bs, [ax, ay, theta, omega], 1].
            correction (torch.Tensor): Like observation, [bs, num_state, 1], [bs, [x, y], 1].
        Returns:
            _type_: _description_
        """
        m1x_prior, f_jac, m1y, h_jac = self.predict(self.m1x_posterior)

        # input1
        state_inno = self.m1x_posterior_previous - self.m1x_prior_previous
        # input2 innovation

        # TODO:
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
        # not correction, use predict, not update.
        # m1x_posterior[mask[:, 0, :].flatten()] = m1x_prior[mask[:, 0, :].flatten()]
        self.innovation = innov
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        self.y_previous = y.detach().clone()
        assert not torch.any(torch.isnan(m1x_posterior))
        return m1x_posterior.clone()



