from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from Net.trainer.base_trainer import BaseFilterNet
from Net.utils import MODELS, MSE, AxisMSE, AxisMSEdB, MSEdB


@MODELS.register_module()
class LitFusionKalmanNet(BaseFilterNet):

    def __init__(self, cfg, save_dir: dict) -> None:
        super().__init__(cfg, save_dir)
        self.sensor_based = self.model.params.sensor_based
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        self.model.init_beliefs(batch['initial_state'])
        opt = self.optimizers()
        # set slide window
        batch_size, _, seq_len = batch[self.sensor_based].shape
        indices_win = torch.arange(0, seq_len)
        slide_win_size = self.cfg.trainer.slide_win_size if self.cfg.trainer.slide_win_size is not None else seq_len
        detach_step = self.cfg.trainer.detach_step if self.cfg.trainer.detach_step is not None else slide_win_size

        # indices_win = [
        #     indices_win[:slide_win_size],
        #     *torch.split(indices_win[slide_win_size:], 1)
        # ]
        indices_win = [
            indices_win[:slide_win_size],
            *torch.split(indices_win[slide_win_size:], slide_win_size)
        ]
        indices_win = indices_win[:-1] if len(
            indices_win[-1]) == 0 else indices_win
        # indices_win = [torch.arange(i,i+slide_win_size) for i in range(seq_len-slide_win_size+1)]

        # set predict
        preds = torch.zeros(batch_size, self.model.state_dim,
                            seq_len).to(self.device)
        loss_item = 0
        # forward
        # pbar = trange(len(indices_win))
        # for idx, indices in enumerate(indices_win):
        for i, indices in enumerate(indices_win):
            # print(indices)
            targets_per_win = batch['ground_truth'][
                ...,
                indices][:, self.
                         target_metric_mask, :]  # [bs, num_state, win_size]
            sensor_per_win = batch[self.sensor_based][:, :, indices]
            correction = batch['filtered_gps'][:, :, indices]
            mask = batch['mask'][:, indices]  # [bs, win_size]
            pred_per_win = torch.zeros(batch_size, self.model.state_dim,
                                       len(indices)).to(targets_per_win.device)
            for step in range(len(indices)):
                input = sensor_per_win[:, :, [step]]
                correct = correction[:, :, [step]]
                pred_per_win[:, :, [step]] = self.model(input, correct)
                if (step + 1) % detach_step == 0:  # noqa
                    self.model._detach()
            self.model._detach()
            preds[:, :, indices] = pred_per_win.detach()

            pred_per_win = pred_per_win[:, self.
                                        pred_metric_mask, :]  # -> state dim -> gt dim
            masked_pred_per_win = torch.masked_select(
                pred_per_win,
                mask[:, None, :])  # must match, else error broadcast.
            masked_target_per_win = torch.masked_select(
                targets_per_win, mask[:, None, :])

            train_loss = self.loss_fn(masked_pred_per_win,
                                      masked_target_per_win)
            loss_item += train_loss.item()

            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.cfg.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.cfg.trainer.
                gradient_clip_algorithm)
            opt.step()

        self.log('train_loss_per_win',
                 loss_item / len(indices_win),
                 prog_bar=True)
        self.log('train_loss_per_timestep', loss_item / seq_len, prog_bar=True)

        # only compute axis in Axis
        all_pred = preds[:, self.pred_metric_mask, :]
        all_target = batch['ground_truth'][:, self.target_metric_mask, :]

        # TODO: reomove this
        # if self.cfg.metric.inverse_transforms:
        #     # INFO:
        #     pred = self.trainer.datamodule.train.inverse_data(pred.cpu())

        #     target = self.trainer.datamodule.train.inverse_data(target.cpu())
        mask_pred = torch.masked_select(
            all_pred,
            batch['mask'][:, None, :])  # must match, else error broadcast.
        mask_target = torch.masked_select(all_target, batch['mask'][:,
                                                                    None, :])
        self.train_mse_dB.update(mask_pred, mask_target)
        # self.train_axis_mse_dB.update(pred, target)
        self.train_rmse.update(mask_pred, mask_target)
        # self.train_axis_rmse.update(pred, target)

        # return train_loss

    def validation_step(self, batch, batch_idx):
        # self.model.init_hidden_KNet(self.device)
        self.model.init_beliefs(batch['initial_state'])
        # preds = torch.zeros_like(batch['targts']).to(self.device)
        seq_len = batch['ground_truth'].shape[-1]
        preds = torch.zeros(batch['initial_state'].shape[0],
                            self.model.state_dim, seq_len).to(self.device)
        # forward
        for t in range(0, seq_len):
            preds[:, :, [t]] = self.model(
                batch[self.sensor_based][:, :, [t]],
                batch['filtered_gps'][:, :, [t]])  # [bs, num_state, seq_len)

        # use mask to select axis and unpaddded.
        preds = preds[:, self.pred_metric_mask, :]
        targets = batch['ground_truth'][:, self.
                                        target_metric_mask, :]  # type: ignore
        masked_preds = torch.masked_select(
            preds, batch['mask'][:,
                                 None, :])  # must match, else error broadcast.
        masked_targets = torch.masked_select(targets, batch['mask'][:,
                                                                    None, :])
        val_loss = self.loss_fn(masked_preds, masked_targets)
        self.log('val_loss', val_loss)

        # For metric
        # Inverse data
        # TODO: remove it.
        # if self.cfg.metric.inverse_transforms:
        #     # INFO:
        #     pred = self.trainer.datamodule.val.inverse_data(pred.cpu())
        #     target = self.trainer.datamodule.val.inverse_data(target.cpu())

        # masked_preds = torch.masked_select(preds, batch['mask'][:, None, :])
        # masked_targets = torch.masked_select(targets, batch['mask'][:, None, :])
        self.val_mse_dB.update(masked_preds, masked_targets)
        # self.val_axis_mse_dB.update(pred, target)
        self.val_rmse.update(masked_preds, masked_targets)
        # self.val_axis_rmse.update(pred, target)

        return val_loss

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        self.model.init_beliefs(batch['initial_state'])
        batch_size, _, seq_len = batch[self.sensor_based].shape
        preds = torch.zeros(batch_size, self.model.state_dim,
                            seq_len).to(self.device)

        for t in range(0, seq_len):
            preds[:, :, [t]] = self.model(batch[self.sensor_based][:, :, [t]],
                                          batch['filtered_gps'][:, :, [t]])
        preds = preds[:, self.pred_metric_mask, :]
        targets: torch.Tensor = batch['ground_truth']

        # Inverse data
        # if self.cfg.metric.inverse_transforms:
        #     # INFO:
        #     pred = self.trainer.datamodule.val.inverse_data(pred.cpu())
        #     target = self.trainer.datamodule.val.inverse_data(target.cpu())
        res = dict(preds=preds, targets=targets)
        res.update(batch)
        return res
