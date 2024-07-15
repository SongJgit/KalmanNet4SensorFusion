from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.regression.mse import _mean_squared_error_compute
from torchmetrics.utilities.checks import _check_same_shape


def _dB_compute(sum_squared_error: torch.Tensor, n_obs: int) -> Tensor:
    return 10 * torch.log10(torch.div(sum_squared_error, n_obs))


def _mean_squared_error_update(
        preds: Tensor,
        target: Tensor,
        dim: int | Tuple | None = None) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Squared Error.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """
    _check_same_shape(preds, target)
    diff = preds - target
    sum_squared_error = torch.sum(torch.square(diff), dim=dim)
    if dim is None:
        n_obs = target.numel()
    elif isinstance(dim, tuple):
        n_obs = 1
        for val in dim:
            n_obs *= target.shape[val]
    else:
        n_obs = target.shape[dim]
    return sum_squared_error, n_obs


class MSE(Metric):

    def __init__(self, dim: int | Tuple | None = None, squared=True):
        """MSE
        Note: Scalar metric or curves like metric.
        Args:
            dim (int): If not None, return curves metric.
            squared (bool, optional): compute RMSE if squared is False. Defaults to True.
        """
        super().__init__()
        self.add_state('sum_squared_error',
                       default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.add_state('total',
                       default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.squared = squared
        self.dim = dim

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> Tensor:
        assert preds.shape == target.shape
        sum_squared_error, n_obs = _mean_squared_error_update(
            preds, target, self.dim)
        self.sum_squared_error = self.sum_squared_error + sum_squared_error  # type: ignore
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes  mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total,
                                           self.squared)


class MSEdB(MSE):
    """Use dB to compute mse.
    None: Scalar Metric.

    Args:
        MSE (_type_): _description_
    """

    def compute(self) -> Tensor:
        """Computes dB mean squared error over state."""
        return _dB_compute(self.sum_squared_error, self.total)


class AxisMSE(Metric):

    def __init__(self,
                 num_axis: int,
                 dim: int | Tuple | None = None,
                 squared: bool = True):
        """Calculate the MSE for each step along multiple axis. like x axis MSE.

        Note: Scalar metric or curves like metric. Only compute data like [batch, axis,...]
            Because of the update and the compute mechanism of torchmetrics,
            it is not possible to share the same metric object when trying to compute metric for multiple axis.
            reference: https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html
        Examples:
            scalar:
                scalar_axis_mse = AxisMSE(num_axis= 3)
                preds = torch.randn(100, 3, 100) # [batch, axis, seq_len]
                targets = torch.randn(100, 3, 100)
                res: List[Tensor] = scalar_axis_mse(preds, targets) # len(res) == num_axis. res[0] is scalar.

                scalar_axis_rmse = AxisMSE(num_axis= 3, dim = None, squared=True)
                preds = torch.randn(100, 3, 100) # [batch, axis, seq_len]
                targets = torch.randn(100, 3, 100)
                res: List[Tensor] = scalar_axis_rmse(preds, targets) # len(res) == num_axis. res[0] is scalar.

            curves like:
                cal metric at each step.
                seq_curves_axis_mse = AxisMSE(num_axis= 3, dim = 0)
                preds = torch.randn(100, 3, 100) # [batch, axis, seq_len]
                targets = torch.randn(100, 3, 100)
                # len(res) == num_axis. res[0].shape == seq_len.
                res: List[Tensor] = seq_curves_axis_mse(preds, targets)

                seq_curves_axis_rmse = AxisMSE(num_axis= 3, dim = 0, squared=True)
                preds = torch.randn(100, 3, 100) # [batch, axis, seq_len]
                targets = torch.randn(100, 3, 100)
                # len(res) == num_axis. res[0].shape == seq_len.
                res: List[Tensor] = seq_curves_axis_rmse(preds, targets)

        Args:
            num_axis (int): The number of axis you want to calculate.
            dim (int): If not None, return curves metric.
            squared (bool, optional): compute RMSE if squared is False. Defaults to True.
        """
        super().__init__()
        for i in range(num_axis):
            self.add_state(f'sum_squared_error_{i}',
                           default=torch.tensor(0.0),
                           dist_reduce_fx='sum')
        self.add_state('total',
                       default=torch.tensor(0.0),
                       dist_reduce_fx='sum')
        self.num_axis = num_axis
        self.dim = dim
        self.squared = squared

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.ndim >= 2, f'The first axis must be batch,but got {preds.ndim=}'
        assert preds.shape == target.shape
        assert preds.shape[
            1] == self.num_axis, 'Compute axis must be equal to shape[1]'
        for i in range(self.num_axis):
            sum_squared_error, n_obs = _mean_squared_error_update(preds[:, i,
                                                                        ...],
                                                                  target[:, i,
                                                                         ...],
                                                                  dim=self.dim)
            temp = getattr(self, f'sum_squared_error_{i}')
            setattr(self, f'sum_squared_error_{i}', sum_squared_error + temp)
        self.total += n_obs

    def compute(self) -> List[torch.Tensor]:
        res = []
        for i in range(self.num_axis):
            sum_squared_error = getattr(self, f'sum_squared_error_{i}')
            res.append(
                _mean_squared_error_compute(sum_squared_error, self.total,
                                            self.squared))
        return res


class AxisMSEdB(AxisMSE):
    """Along axis compute MSE dB.


    ike x axis MSE dB.
    """

    def compute(self) -> List[torch.Tensor]:
        """Computes dB mean squared error over state."""
        res = []
        for i in range(self.num_axis):
            sum_squared_error = getattr(self, f'sum_squared_error_{i}')
            res.append(_dB_compute(sum_squared_error, self.total))
        return res


# def compute_metric(predict: Tensor,
#                    target: Tensor,
#                    save_path: str | None = None,
#                    name: str | None = None) -> None:
#     """generte metric for prediction.

#     Args:
#         predict (Tensor): shape of prediction is [batch, state, seq]
#         target (Tensor): shape of prediction is [batch, state, seq]
#         save_path (str | None): _description_
#         name (str | None): _description_
#     """
#     assert name is not None, 'Need name to display metric.'
#     loss_fn = nn.MSELoss(reduction='mean')
#     data_len = predict.shape[0]
#     MSE_linear_arr = torch.zeros(data_len)
#     for j in range(data_len):
#         MSE_linear_arr[j] = loss_fn(predict[j, :], target[j, :])
#     print(MSE_linear_arr.shape)
#     MSE_linear_avg = torch.mean(MSE_linear_arr)
#     MSE_dB_avg = 10 * torch.log10(MSE_linear_avg)

#     # Standard deviation
#     MSE_linear_std = torch.std(MSE_linear_arr, unbiased=True)
#     print(MSE_linear_std)
#     # Confidence interval
#     KF_std_dB = 10 * torch.log10(MSE_linear_std + MSE_linear_avg) - MSE_dB_avg

#     print(f'{name} - MSE LOSS: {MSE_dB_avg}[dB]')
#     print(f'{name} - STD: {KF_std_dB}[dB]')
#     if save_path is not None:
#         color = ['-ro', 'darkorange', 'k-', 'b-', 'g-']
#         legend = ['MSE_AVG', 'MSE_dB_AVG', 'MSE STD', 'MSE_STD_dB']
#         fig = plt.figure(figsize=(40, 20))
#         xplt = range(0, data_len)
#         yplt_0 = MSE_linear_avg * torch.ones(data_len)
#         plt.plot(xplt, yplt_0, color[0], label=legend[0])

#         yplt_1 = MSE_dB_avg * torch.ones(data_len)
#         plt.plot(xplt, yplt_1, color[1], label=legend[1])

#         yplt_2 = MSE_linear_std * torch.ones(data_len)
#         plt.plot(xplt, yplt_2, color[2], label=legend[2])

#         yplt_3 = KF_std_dB * torch.ones(data_len)
#         plt.plot(xplt, yplt_3, color[3], label=legend[3])
#         plt.xlabel('Number of samples', fontsize=32)
#         plt.ylabel('MSE Loss Value', fontsize=32)
#         plt.legend(fontsize=32)
#         plt.grid(True)
#         plt.savefig(save_path)


def compute_metric(
        preds: torch.Tensor,
        targets: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """_summary_

    Args:
        preds (torch.Tensor): [batch_size, num_axis, seq_length]
        targets (torch.Tensor): [batch_size, num_axis, seq_length]

    Returns:
        Tuple[Dict[str, torch.Tensor| List[torch.Tensor]]]: scalar and curves.
    """
    assert preds.shape == targets.shape, f' {preds.shape=} must be equal to {targets.shape=}'

    num_axis = preds.shape[1]

    # scalar metric
    mse_fn = MSE()
    mse_dB_fn = MSEdB()
    rmse_fn = MSE(squared=False)
    axis_mse_fn = AxisMSE(num_axis=num_axis)
    axis_mse_dB_fn = AxisMSEdB(num_axis=num_axis)
    axis_rmse_fn = AxisMSE(num_axis=num_axis, squared=False)

    mse = mse_fn(preds, targets)
    mse_dB = mse_dB_fn(preds, targets)
    rmse = rmse_fn(preds, targets)
    axis_mse = axis_mse_fn(preds, targets)
    axis_mse_dB = axis_mse_dB_fn(preds, targets)
    axis_rmse = axis_rmse_fn(preds, targets)

    scalar = dict(
        mse=mse,
        mse_dB=mse_dB,
        rmse=rmse,
        axis_mse=axis_mse,
        axis_mse_dB=axis_mse_dB,
        axis_rmse=axis_rmse,
    )

    # curves like metric, along batch compute each step
    mse_curves_fn = MSE(dim=(0, 1))
    mse_dB_curves_fn = MSEdB(dim=(0, 1))
    rmse_curves_fn = MSE(dim=(0, 1), squared=False)
    axis_mse_curves_fn = AxisMSE(num_axis=num_axis, dim=0)
    axis_mse_dB_curves_fn = AxisMSEdB(num_axis=num_axis, dim=0)
    axis_rmse_curves_fn = AxisMSE(num_axis=num_axis, dim=0, squared=False)

    mse_curves = mse_curves_fn(preds, targets)
    mse_dB_curves = mse_dB_curves_fn(preds, targets)
    rmse_curves = rmse_curves_fn(preds, targets)
    axis_mse_curves = axis_mse_curves_fn(preds, targets)
    axis_mse_dB_curves = axis_mse_dB_curves_fn(preds, targets)
    axis_rmse_curves = axis_rmse_curves_fn(preds, targets)

    curves = dict(
        mse_curves=mse_curves,
        mse_dB_curves=mse_dB_curves,
        rmse_curves=rmse_curves,
        axis_mse_curves=axis_mse_curves,
        axis_mse_dB_curves=axis_mse_dB_curves,
        axis_rmse_curves=axis_rmse_curves,
    )

    return scalar, curves


def print_metrics(predict: Tensor, target: Tensor, name: str = None):
    scalar, curves = compute_metric(predict, target)
    print(f"{name} - MSE LOSS: {scalar['mse_dB']}[dB]")
    print(f"{name} - MSE LOSS: {scalar['mse']}")
    print(f"{name} - RMSE LOSS: {scalar['rmse']}")
    return scalar, curves
