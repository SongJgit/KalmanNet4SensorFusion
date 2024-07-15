from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp
import torch
from mmengine import ConfigDict
from torch.utils.data import Dataset

from Net.utils import FILTERDATASET, PARAMS


class Params:

    def __init__(self,
                 state_dim: int,
                 observation_axis: List[int] | None = None,
                 T: int = 1,
                 noise_r2: int | None | float = None,
                 noise_q2: int | None | float = None,
                 constant_noise: bool = True,
                 dataset: Dataset | ConfigDict = None):
        """ Q = {Gamma} @ {Gamma}^{T} * var

        Args:
            state_dim (int): state dimension
            observation_axis (List[int] | None, optional): len(observation_axis) == observation_dim. Defaults to None.
            T (int, optional): _description_. Defaults to 1.
            noise_r2 (int | None, optional): _description_. Defaults to None.
            noise_q2 (int | None, optional): _description_. Defaults to None.
            constant_noise (bool, optional): _description_. Defaults to True.
            dataset (str, optional): if not constant noise, R must from dataset.
        """
        self.state_dim = state_dim
        self.constant_noise = constant_noise
        self.dataset = dataset
        self.T = T
        self.noise_r2 = noise_r2
        self.noise_q2 = noise_q2

        # Check args
        if isinstance(self.dataset, (dict, ConfigDict)):
            self.dataset = FILTERDATASET.build(dataset)
        if self.dataset is not None:
            assert self.dataset.sample_time_interval == self.T or self.T is None
            self.T = self.dataset.sample_time_interval

        if observation_axis is None:
            self.observation_axis = list(range(state_dim))
        else:
            self.observation_axis = observation_axis
        self.observation_dim = len(self.observation_axis)
        assert len(self.observation_axis) <= self.state_dim

        if self.constant_noise:
            if self.dataset is None:
                assert noise_q2 is not None and noise_r2 is not None
            else:
                assert self.dataset.noise_q2 == self.noise_q2 or noise_q2 is None
                assert self.dataset.noise_r2 == self.noise_r2 or noise_r2 is None
                self.noise_q2 = self.dataset.noise_q2
                self.noise_r2 = self.dataset.noise_r2
        else:
            assert self.dataset is not None and self.noise_r2 is None

    def get(self, name: str) -> Any:
        return getattr(self, name)




@PARAMS.register_module()
class NCLTFusionIMUGPSParams(Params):

    def __init__(self,
                 state_dim: int = 6,
                 observation_axis: List[int] | None = [1, 3],
                 T: int = 1,
                 noise_r2: int | None = None,
                 noise_q2: int | None = None,
                 constant_noise: bool = True,
                 dataset: Dict[str, Any] | Dataset | None = None):
        super().__init__(state_dim, observation_axis, T, noise_r2, noise_q2,
                         constant_noise, dataset)
        """NCLT only need velocity parameters

        """
        self.F = torch.eye(state_dim)
        self.H = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                              dtype=torch.float32)
        self.sensor_based = 'imu'
        pass

    def get_dynamic_F(self,
                      prev_state: torch.tensor,
                      current_sensor: torch.tensor,
                      delta_t: torch.Tensor | int = 1) -> torch.Tensor:
        # deprecate
        """_summary_, imu use preva_state.

        Args:
                prev_state (torch.tensor): [x,y, x',y', theta_k, omega_k]
                current_sensor (torch.tensor): [x,y, x',y', ax, ay, theta, omega]
                delta_t (torch.Tensor | int, optional): _description_. Defaults to 1.
        Returns:
            torch.Tensor: _description_
        """
        # for state= [x, y, x', y', ax , ay , theta, omega]
        F_matrix = []
        theta = prev_state[:, 4, :]
        if isinstance(delta_t, int):
            delta_t = torch.fill(torch.zeros(theta.shape[0]), delta_t)
        for tta, dt in zip(theta, delta_t):
            costh = torch.cos(-tta)
            sinth = torch.sin(-tta)
            F = torch.tensor([
                [1, 0, dt, 0, 0.5 * costh * dt**2, -0.5 * sinth * dt**2, 0, 0],
                [0, 1, 0, dt, 0.5 * sinth * dt**2, 0.5 * costh * dt**2, 0, 0],
                [0, 0, 1, 0, dt * costh, -dt * sinth, 0, 0],  # ignore:E126
                [0, 0, 0, 1, dt * sinth, dt * costh, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ])
            F_matrix.append(F[None, ...])
        return torch.vstack(F_matrix)

    def get_pred_jac(self, prev_states,
                     current_states) -> Tuple[torch.Tensor, ...]:
        """Get pred and jacobian.

        In fact a more elegant implementation would be to use torch.autograd.grad for the jac calculation.
        However, in this repo earlier implementations, pl.predcit threw some exceptions(https://github.com/Lightning-AI/pytorch-lightning/issues/19910), 
        so we chose to calculate it manually.
        Although subsequently I found ways to avoid these exceptions, as I mentioned in the issue. But to prevent some other bugs from occurring, I didn't continue to modify it in this version (in fact, in this project, jac doesn't affect kalmannet/split-kalmannet computation).
        If you want to get the code to compute the jacobian using torch.autograd.grad, send me an email.


        Args:
            prev_states (_type_): [bs, [x, y, x', y', theta_k, omega_k],1]
            current_states (_type_):  [bs, [ax , ay , theta, omega],1]
        Returns :
                predict state, and jacobian
        """
        with torch.set_grad_enabled(True):
            batch_size, state_dim, _ = prev_states.shape
            prev_states = prev_states.detach().clone().repeat(
                1, 1, state_dim)  # for autograd
            current_states = current_states.detach().clone().repeat(
                1, 1, state_dim)
            prev_states.requires_grad_(True)
            current_states.requires_grad_(True)
            x_k = prev_states[:, 0, :]
            y_k = prev_states[:, 1, :]
            x_dot_k = prev_states[:, 2, :]
            y_dot_k = prev_states[:, 3, :]
            theta_k = prev_states[:, 4, :]
            ax_imu = current_states[:, 0, :]
            ay_imu = current_states[:, 1, :]
            heading_imu = current_states[:, 2, :]
            omega_imu = current_states[:, 3, :]
            dt_sym = 1
            ax_global = ax_imu * torch.cos(-theta_k) - ay_imu * torch.sin(
                -theta_k)
            ay_global = ax_imu * torch.sin(-theta_k) + ay_imu * torch.cos(
                -theta_k)
            f1 = x_k + x_dot_k * dt_sym + 0.5 * ax_global * dt_sym * dt_sym
            f2 = y_k + y_dot_k * dt_sym + 0.5 * ay_global * dt_sym * dt_sym
            f3 = x_dot_k + ax_global * dt_sym
            f4 = y_dot_k + ay_global * dt_sym
            f5 = heading_imu  # for autograd.
            f6 = omega_imu
            f = torch.cat([f1, f2, f3, f4, f5, f6],
                          -1).reshape(batch_size, state_dim, state_dim)
            mask = torch.eye(state_dim).repeat(batch_size, 1,
                                               1).to(prev_states.device)
            jac = torch.autograd.grad(f,
                                      prev_states,
                                      mask,
                                      create_graph=True,
                                      materialize_grads=True)[0].transpose(
                                          -1, -2)
            f[:, 4, 0] = wraptopi(f[:, 4, 0])
        return f[:, :, [0]], jac

    # def get_pred_jac(self,prev_states: torch.Tensor, current_states:torch.Tensor)-> Tuple[torch.Tensor,...]:
    #     """_summary_

    #     Args:
    #         prev_states (_type_): [bs, [x, y, x', y', theta_k, omega_k],1]
    #         current_states (_type_):  [bs, [ax , ay , theta, omega],1]
    #     Returns :
    #             predict state, and jacobian
    #     """
    #     batch_size, state_dim, _ = prev_states.shape
    #     device  = prev_states.device
    #     ##### Symbolic Variables #####
    #     # modified from https://github.com/AbhinavA10/mte546-project
    #     x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k, dt_sym = sp.symbols(
    #         'x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k, dt_sym', real=True)
    #     ax_imu, ay_imu, heading_imu, omega_imu = sp.symbols(
    #         'ax_imu, ay_imu, theta_imu, omega_imu', real=True)
    #     ax_global = ax_imu*sp.cos(-theta_k) - ay_imu*sp.sin(-theta_k)
    #     ay_global = ax_imu*sp.sin(-theta_k) + ay_imu*sp.cos(-theta_k)
    #     f1 = x_k + x_dot_k*dt_sym + 0.5*ax_global*dt_sym*dt_sym
    #     f2 = y_k + y_dot_k*dt_sym + 0.5*ay_global*dt_sym*dt_sym
    #     f3 = x_dot_k + ax_global*dt_sym
    #     f4 = y_dot_k + ay_global*dt_sym
    #     f5 = heading_imu # Theta estimate from IMU
    #     f6 = omega_imu
    #     f_imu_input=sp.Matrix([f1, f2, f3, f4, f5, f6])
    #     F_JACOB_IMU = f_imu_input.jacobian([x_k, y_k, x_dot_k, y_dot_k, theta_k, omega_k])

    #     f = []
    #     jac = []
    #     with torch.no_grad():
    #         prev_states = prev_states.cpu().numpy()
    #         current_states = current_states.cpu().numpy()
    #         for i in range(batch_size):
    #             prev_state = prev_states[i].squeeze(1)
    #             _ax = current_states[i,0, 0]
    #             _ay= current_states[i,1, 0]
    #             _theta = current_states[i,2, 0]
    #             _omega = current_states[i,3, 0]
    #             _dt = 1
    #             fx = np.array(f_imu_input.subs([(x_k,         prev_state[0]),
    #                                 (y_k,         prev_state[1]),
    #                                 (x_dot_k,     prev_state[2]),
    #                                 (y_dot_k,     prev_state[3]),
    #                                 (theta_k,     prev_state[4]),
    #                                 #    (omega_k,     prev_state[5]),
    #                                 (ax_imu,      _ax),
    #                                 (ay_imu,      _ay),
    #                                 (heading_imu, _theta),
    #                                 (omega_imu,   _omega),
    #                                 (dt_sym,      _dt)
    #                                 ])).astype(np.float32).flatten()

    #             f_jac = np.array(F_JACOB_IMU.subs([(x_k,         prev_state[0]),
    #                                             (y_k,         prev_state[1]),
    #                                             (x_dot_k,     prev_state[2]),
    #                                             (y_dot_k,     prev_state[3]),
    #                                             (theta_k,     prev_state[4]),
    #                                             (omega_k,     prev_state[5]),
    #                                             (ax_imu,      _ax),
    #                                             (ay_imu,      _ay),
    #                                             (heading_imu, _theta),
    #                                             (omega_imu,   _omega),
    #                                             (dt_sym,      _dt)
    #                                             ])).astype(np.float32)
    #             f.append(torch.from_numpy(fx))
    #             jac.append(torch.from_numpy(f_jac))
    #         f = torch.stack(f).unsqueeze(-1).to(device)# ->[bs,num_sstates,1]
    #         jac = torch.stack(jac).to(device)
    #         f[:, 4, 0] = wraptopi(f[:, 4, 0])
    #     return f, jac

    def get_obs_jac(self, states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """GPS Measurement Model is linear, so is H.

        Args:
            states (torch.Tensor): [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, ...]: y_pred [bs, num_obs_state, 1], jac
        """
        batch = states.shape[0]
        H = self.H[None, ...].repeat(batch, 1, 1)
        jac = H.to(states.device)
        y_pred = H.to(states.device) @ states
        return y_pred, jac


@PARAMS.register_module()
class NCLTFusionWheelGPSParams(Params):

    def __init__(
        self,
        state_dim: int = 6,
        observation_axis: List[int] | None = [1, 3],
        T: int = 1,
        noise_r2: int | None | float = None,
        noise_q2: int | None | float = None,
        constant_noise: bool = True,
        dataset: Dict[str, Any] | Dataset | None = None,
    ):
        super().__init__(state_dim, observation_axis, T, noise_r2, noise_q2,
                         constant_noise, dataset)
        """NCLT only need velocity parameters

        """
        self.F = torch.eye(state_dim)
        self.H = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                              dtype=torch.float32)
        self.sensor_based = 'wheel'

        pass

    @torch.enable_grad()
    def get_pred_jac(self, prev_states,
                     current_states) -> Tuple[torch.Tensor, ...]:
        """_summary_

        Args:
            prev_states (_type_): [bs, [x, y, x', y', theta_k, omega_k],1]
            current_states (_type_):  [bs, [ax , ay , theta, omega],1]
        Returns :
                predict state, and jacobian
        """

        batch_size, state_dim, _ = prev_states.shape
        prev_states = prev_states.detach().clone().repeat(
            1, 1, state_dim)  # for autograd
        current_states = current_states.detach().clone().repeat(
            1, 1, state_dim)
        prev_states = prev_states.requires_grad_(True)
        current_states = current_states.requires_grad_(True)

        vl_wheel = current_states[:, 0, :]
        vr_wheel = current_states[:, 1, :]
        heading = current_states[:, 2, :]
        omega = current_states[:, 3, :]
        x_k = prev_states[:, 0, :]
        y_k = prev_states[:, 1, :]
        x_dot_k = prev_states[:, 2, :]
        y_dot_k = prev_states[:, 3, :]
        theta_k = prev_states[:, 4, :]
        omega_k = prev_states[:, 5, :]
        dt_sym = 1
        with torch.enable_grad():
            # if pl.predict, dont work?
            v_c = (0.5 * (vl_wheel + vr_wheel))
            f1 = (x_k + v_c * torch.cos(heading) * dt_sym)
            f2 = (y_k + v_c * torch.sin(heading) * dt_sym)
            f3 = (v_c * torch.cos(heading))
            f4 = (v_c * torch.sin(heading))
            # x_dot, theta, omgea for auto grad unused,
            # use zero factor ensures that there is no impact on the jac results.
            f5 = heading
            f6 = omega
            f = torch.cat([f1, f2, f3, f4, f5, f6],
                          -1).reshape(batch_size, state_dim, state_dim)
            mask = torch.eye(state_dim).repeat(batch_size, 1, 1).to(f.device)
            # jac = torch.autograd.grad(f, prev_states, mask, create_graph =True,
            # materialize_grads=True)[0].transpose(-1,-2)
            # jac = torch.zeros_like(jac)
            jac = torch.zeros(batch_size, state_dim, state_dim).to(f.device)
            # BUG: The predict stage does not work with autograd for unknown reasons.
            jac[:, 0, 0] = 1
            jac[:, 1, 1] = 1
            f[:, 4, 0] = wraptopi(f[:, 4, 0])
        return f[:, :, [0]], jac

    def get_obs_jac(self, states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """GPS Measurement Model is linear, so is H.

        Args:
            states (torch.Tensor): [bs, num_state, 1]

        Returns:
            Tuple[torch.Tensor, ...]: y_pred [bs, num_obs_state, 1], jac
        """
        batch = states.shape[0]
        jac = self.H[None, ...].repeat(batch, 1, 1).to(states.device)
        y_pred = self.H[None, ...].to(states.device) @ states
        return y_pred, jac

    def get_dynamic_F(self,
                      prev_state: torch.tensor,
                      current_sensor: torch.tensor,
                      delta_t: torch.Tensor | int = 1) -> torch.Tensor:
        # deprecate
        """some sensor use prev_state.theta, some sensor use current sensor.

        Args:
            prev_state (torch.tensor): [x,y, x',y', theta_k, omega_k]
            current_sensor (torch.tensor): [x,y, x',y', ax, ay, theta, omega]
            delta_t (torch.Tensor | int, optional): _description_. Defaults to 1.

        Returns:
            torch.Tensor: _description_
        """
        F_matrix = []
        heading = current_sensor[:, 6, :]
        if isinstance(delta_t, int):
            delta_t = torch.fill(torch.zeros(heading.shape[0]), delta_t)
        for tta, dt in zip(heading, delta_t):
            costh = torch.cos(tta)
            sinth = torch.sin(tta)
            dt2 = 0.5 * dt
            F = torch.tensor([[1, 0, 0, 0, dt2 * costh, dt2 * costh, 0, 0],
                              [0, 1, 0, 0, dt2 * sinth, dt2 * sinth, 0, 0],
                              [0, 0, 0, 0, 0.5 * costh, 0.5 * costh, 0, 0],
                              [0, 0, 0, 0, 0.5 * sinth, 0.5 * sinth, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
            F_matrix.append(F[None, ...])
        return torch.vstack(F_matrix)

    def convert(self, states):
        states[:, 4, :] = wraptopi(states[:, 4, :])
        return states


def wraptopi(x: torch.Tensor) -> torch.Tensor:
    """
    Modified from mte546-project numpy.array to torch.batch.
    Wrap theta measurements to [-pi, pi].
    Accepts an angle measurement in radians and returns an angle measurement in radians
    Args:
        x (torch.Tensor): [bs, 1], [1, bs] or another shape.

    Returns:
        torch.Tensor: equips to x.shape
    """
    pos_pi_mask = x > torch.pi
    neg_pi_mask = x < -torch.pi
    x[pos_pi_mask] = x[pos_pi_mask] - (torch.floor(x[pos_pi_mask] /
                                                   (2 * torch.pi)) +
                                       1) * 2 * torch.pi
    x[neg_pi_mask] = x[neg_pi_mask] + (torch.floor(x[neg_pi_mask] /
                                                   (-2 * torch.pi)) +
                                       1) * 2 * torch.pi
    return x
