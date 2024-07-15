from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
from mmengine.config import Config
from torch import Tensor

from Net.utils import MODELS, PARAMS


@MODELS.register_module()
class KalmanNet(nn.Module):

    def __init__(self,
                 params: Dict[str, Any] | Any,
                 in_mult_KNet: int = 5,
                 out_mult_KNet: int = 40,
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
        self.in_mult_KNet = in_mult_KNet
        self.out_mult_KNet = out_mult_KNet

        self.belief_state: torch.Tensor
        self.belief_covariance: torch.Tensor
        self._initialized = False

        self.seq_len_input = 1
        self.init_KGainNet()
        self.init_weights()

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

    def init_weights(self):
        # Initialize network parameters #
        init.kaiming_uniform_(self.FC2[0].weight, nonlinearity='relu')
        self.FC2[0].bias.data.fill_(0)
        init.kaiming_uniform_(self.FC2[2].weight, nonlinearity='relu')
        self.FC2[2].bias.data.fill_(0)

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

    # Compute Priors #

    def step_prior(self,
                   m1x_posterior: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute Priors
        """
        # Predict the 1-st moment of x
        m1x_prior, f_jac = self.params.get_pred_jac(m1x_posterior)

        # Predict the 1-st moment of y.device

        m1y, h_jac = self.params.get_obs_jac(m1x_prior)
        assert not torch.any(torch.isnan(m1y))
        assert not torch.any(torch.isnan(m1x_prior))
        return m1x_prior, m1y

    # Kalman Gain Estimation
    def step_KGain_est(self, y: torch.Tensor | None, m1y: torch.Tensor,
                       y_previous: torch.Tensor, m1x_posterior: torch.Tensor,
                       m1x_posterior_previous: torch.Tensor,
                       m1x_prior_previous: torch.Tensor) -> torch.Tensor:
        """Kalman Gain Estimation
        Args:
            y (Tensor): _description_
        """
        # both in size [batch_size, n]
        # compute F1 and F2

        obs_diff = torch.squeeze(y, 2) - torch.squeeze(y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(m1y, 2)

        # both in size [batch_size, m]
        # compute forward evolution difference F3 and forward update difference F4
        fw_evol_diff = torch.squeeze(m1x_posterior, 2) - torch.squeeze(
            m1x_posterior_previous, 2)
        fw_update_diff = torch.squeeze(m1x_posterior, 2) - torch.squeeze(
            m1x_prior_previous, 2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12,
                                  out=None).to(y.device)
        obs_innov_diff = func.normalize(obs_innov_diff,
                                        p=2,
                                        dim=1,
                                        eps=1e-12,
                                        out=None).to(y.device)
        fw_evol_diff = func.normalize(fw_evol_diff,
                                      p=2,
                                      dim=1,
                                      eps=1e-12,
                                      out=None).to(y.device)
        fw_update_diff = func.normalize(fw_update_diff,
                                        p=2,
                                        dim=1,
                                        eps=1e-12,
                                        out=None).to(y.device)
        return obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff

    #
    # Kalman Net Step #
    #
    def _forward_(self, y: torch.Tensor | None) -> torch.Tensor:

        # Compute Priors
        # new state and new obser
        m1x_prior, m1y = self.step_prior(self.m1x_posterior)

        # Compute Kalman inputs
        obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff = self.step_KGain_est(
            y, m1y, self.y_previous, self.m1x_posterior,
            self.m1x_posterior_previous, self.m1x_prior_previous)
        # Kalman Gain Network Step
        kalman_gain = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff,
                                      fw_update_diff).reshape(
                                          self.batch_size, self.state_dim,
                                          self.observation_dim)

        # Innovation
        diff_y = y - m1y  # [batch_size, n, 1]

        # compute current m1x
        m1x_posterior = torch.baddbmm(m1x_prior, kalman_gain, diff_y)
        self.innovation = diff_y.detach().clone()
        # update last m1x
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        # update current m1x, detach for next slide window
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        # update y_prev
        self.y_previous = y.detach().clone()

        assert not torch.any(torch.isnan(m1x_posterior))
        return m1x_posterior.clone()

    def KGain_step(self, obs_diff: torch.Tensor, obs_innov_diff: torch.Tensor,
                   fw_evol_diff: torch.Tensor,
                   fw_update_diff: torch.Tensor) -> torch.Tensor:
        """Kalman Gain Step, compute the kalman gain

        Args:
            obs_diff (Tensor): normalized observations difference [bs, observation_dim]
            obs_innov_diff (Tensor):  normalized observations innov difference [bs, observation_dim]
            fw_evol_diff (Tensor): forward evolution difference [bs, state_dim]
            fw_update_diff (Tensor): forward update difference [bs, state_dim]


        Returns:
            torch.Tensor: _description_
        """

        def expand_dim(x):
            # [batch_size, m] -> [1, batch_size, m]
            expanded = torch.empty(1, self.batch_size,
                                   x.shape[-1]).to(x.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # assert not torch.any(torch.isnan(obs_diff))
        # assert not torch.any(torch.isnan(obs_innov_diff))
        # assert not torch.any(torch.isnan(fw_evol_diff))
        # assert not torch.any(torch.isnan(fw_update_diff))

        # Forward Flow #
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # assert not torch.any(torch.isnan(in_FC5))
        # Q-GRU
        in_Q = out_FC5
        out_Q, h_Q = self.GRU_Q(in_Q,
                                self.h_Q)  # type: Tuple[torch.Tensor,Tensor]

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, h_Sigma = self.GRU_Sigma(
            in_Sigma, self.h_Sigma)  # type: Tuple[torch.Tensor,Tensor]

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)

        out_S, h_S = self.GRU_S(in_S,
                                self.h_S)  # type: Tuple[torch.Tensor,Tensor]

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)

        # TODO: check this

        out_FC2 = self.FC2(in_FC2)

        #
        # Backward Flow #
        #

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        # detach for next step in slide window.
        self.h_Sigma: torch.Tensor = out_FC4
        self.h_Q: torch.Tensor = h_Q
        self.h_S: torch.Tensor = h_S

        return out_FC2

    def _init_hidden(self, state: torch.Tensor):
        """
            init hidden state of the new sequence.

        Args:
            device (_type_): _description_
        """
        # weight = next(self.parameters()).data
        # hidden = weight.new(self.seq_len_input, self.batch_size,
        #                     self.d_hidden_S).zero_()
        # self.h_S = hidden.data
        device = state.device
        batch_size = state.shape[0]
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, batch_size,
            1).to(device)  # batch size expansion
        # hidden = weight.new(self.seq_len_input, self.batch_size,
        #                     self.d_hidden_Sigma).zero_()
        # self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, batch_size,
            1).to(device)  # batch size expansion
        # hidden = weight.new(self.seq_len_input, self.batch_size,
        #                     self.d_hidden_Q).zero_()
        # self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(
            self.seq_len_input, batch_size,
            1).to(device)  # batch size expansion

    def forward(self, y: torch.Tensor):
        return self._forward_(y)

    def _detach(self):
        self.h_Sigma = self.h_Sigma.detach()
        self.h_Q = self.h_Q.detach()
        self.h_S = self.h_S.detach()
        self.innovation = self.innovation.detach()
        self.m1x_posterior_previous = self.m1x_posterior_previous.detach()
        self.m1x_posterior = self.m1x_posterior.detach()
        self.m1x_prior_previous = self.m1x_prior_previous.detach()
        self.y_previous = self.y_previous.detach()
