from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm
from torch import Tensor, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from Net.utils import MODELS, MSE, AxisMSE, AxisMSEdB, MSEdB

# cSpell:ignore preds, Kalman, rmse


class BaseFilterNet(pl.LightningModule):

    def __init__(self, cfg, save_dir: dict) -> None:
        super().__init__()
        self.model = MODELS.build(cfg.model)
        self.cfg = cfg
        self.save_dir = save_dir
        # select state to compute loss. dim must be equal.
        self.pred_metric_mask = cfg.metric.pred_metric_mask if cfg.metric.get(
            'pred_metric_mask') else torch.ones(self.model.state_dim)
        self.target_metric_mask = cfg.metric.target_metric_mask if cfg.metric.get(
            'target_metric_mask') else torch.ones(self.model.state_dim)

        if cfg.trainer.loss_name == 'MSE':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif cfg.trainer.loss_name == 'SmoothL1':
            self.loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.train_mse_dB = MSEdB()
        # self.train_axis_mse_dB = AxisMSEdB(cfg.metric.num_metric_dims)
        self.train_rmse = MSE(squared=False)
        # self.train_axis_rmse = AxisMSE(cfg.metric.num_metric_dims,
        #                                squared=False)

        self.test_mse_dB = MSEdB()

        self.val_mse_dB = MSEdB()
        # self.val_axis_mse_dB = AxisMSEdB(cfg.metric.num_metric_dims)
        self.val_rmse = MSE(squared=False)
        # self.val_axis_rmse = AxisMSE(cfg.metric.num_metric_dims, squared=False)
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False

    def on_train_start(self) -> None:
        # path = self.save_dir['log_images_dir']
        # prefix, img_list = get_img(path)
        # self.logger.log_image(key='samples', images=img_list, caption=prefix)
        path = self.save_dir['log_metrics_dir']
        if os.path.exists(osp.join(path, 'Metric.csv')):
            df = pd.read_csv(osp.join(path, 'Metric.csv'))
            self.logger.log_table(key='Metrics',
                                  columns=list(df.columns),
                                  data=df.values)

    def training_step(self, batch, batch_idx):

        # self.model.init_hidden_KNet(self.device)
        self.model.init_beliefs(batch['initial_state'])

        # preds = torch.zeros(
        #     batch['initial_state'].shape[0], self.sys_model.m,
        #     self.cfg.data.train_seq_len).to(self.device)

        # # FIXME: optim this for seq windows
        # for t in range(0, self.cfg.data.train_seq_len):
        #    preds[:, :, [t]] = self.model(batch['inputs'][:, :, [t]])

        # train_loss = self.loss_fn(preds, batch['targets'])
        # self.log('train_loss', train_loss)

        opt = self.optimizers()
        indices_win = torch.arange(0, self.cfg.data.train_seq_len).reshape(
            -1, self.cfg.trainer.slide_win_size)
        preds = torch.zeros(batch['initial_state'].shape[0],
                            self.model.state_dim,
                            self.cfg.data.train_seq_len).to(self.device)
        loss_item = 0
        for indices in indices_win:
            targets = batch['targets'][:, :, indices]
            targets = targets[:, self.target_metric_mask, :]
            pred = torch.zeros_like(targets)
            for idx, t in enumerate(indices):
                input = batch['inputs'][:, :, [t]]
                pred[:, :, [idx]] = self.model(input)
            train_loss = self.loss_fn(pred, targets)
            loss_item += train_loss.item()

            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.cfg.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.cfg.trainer.
                gradient_clip_algorithm)
            opt.step()
            preds[:, :, indices] = pred.detach()
        self.log('train_loss', loss_item / indices_win.shape[0])

        # only compute axis in Axis

        pred = preds[:, self.pred_metric_mask, :]
        target = batch['targets'][:, self.target_metric_mask, :]

        if self.cfg.metric.inverse_transforms:
            # INFO:
            pred = self.trainer.datamodule.train.inverse_data(pred.cpu())
            target = self.trainer.datamodule.train.inverse_data(target.cpu())

        self.train_mse_dB.update(pred, target)
        self.train_axis_mse_dB.update(pred, target)
        self.train_rmse.update(pred, target)
        self.train_axis_rmse.update(pred, target)

        # return train_loss

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        # TODO: random length hard to clear comput metric.
        # mse = self.train_axis_mse_dB.compute()
        # rmse = self.train_axis_rmse.compute()
        # values = dict()
        # for idx, name in enumerate(self.cfg.metric.metric_dim_names):
        #     values[f'train/{name}_Axis_MSE[dB]'] = mse[idx]
        #     values[f'train/{name}_Axis_RMSE'] = rmse[idx]

        # self.log_dict(values)
        # self.train_axis_mse_dB.reset()
        # self.train_axis_rmse.reset()

        self.log('train/RMSE', self.train_rmse, prog_bar=True)
        self.log('train/MSE[dB]', self.train_mse_dB, prog_bar=True)

    def on_validation_start(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        # self.model.init_hidden_KNet(self.device)
        self.model.init_beliefs(batch['initial_state'])
        # preds = torch.zeros_like(batch['targts']).to(self.device)
        seq_len = batch['targets'].shape[-1]
        preds = torch.zeros(batch['initial_state'].shape[0],
                            self.model.state_dim, seq_len).to(self.device)
        # forward
        for t in range(0, batch['targets'].shape[-1]):
            preds[:, :, [t]] = self.model(
                batch['inputs'][:, :, [t]])  # [bs, num_state, seq_len)

        # use mask to select axis and unpaddded.
        preds = preds[:, self.pred_metric_mask, :]
        targets = batch['targets'][:,
                                   self.target_metric_mask, :]  # type: ignore
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

    def on_validation_epoch_end(self) -> None:
        # mse = self.val_axis_mse_dB.compute()
        # rmse = self.val_axis_rmse.compute()
        # values = dict()
        # for idx, name in enumerate(self.cfg.metric.metric_dim_names):
        #     values[f'val/{name}_Axis_MSE[dB]'] = mse[idx]
        #     values[f'val/{name}_Axis_RMSE'] = rmse[idx]

        # self.log_dict(values)
        # self.val_axis_mse_dB.reset()
        # self.val_axis_rmse.reset()

        self.log('val/MSE[dB]', self.val_mse_dB)
        self.log('val/RMSE', self.val_rmse)
        self.log('val_MSE_dB', self.val_mse_dB)  # for monitor

    def test_step(self, batch, batch_idx):

        # self.model.init_hidden_KNet(self.device)
        self.model.init_beliefs(batch['initial_state'])
        x_out_test_batch = torch.zeros(
            batch['initial_state'].shape[0], self.sys_model.m,
            self.cfg.data.test_seq_len).to(self.device)
        for t in range(0, self.cfg.data.test_seq_len):
            x_out_test_batch[:, :, [t]] = self.model(batch['inputs'][:, :,
                                                                     [t]])

        mse_testbatch_linear_LOSS = self.loss_fn(x_out_test_batch,
                                                 batch['targets'])
        self.mse_test_linear_epoch.append(mse_testbatch_linear_LOSS.item())
        self.mse_test_dB_epoch.append(
            10 * torch.log10(self.mse_test_linear_epoch[-1]))

        self.log('test/loss', mse_testbatch_linear_LOSS)
        self.log('test/[dB]', self.mse_test_dB_epoch[-1])
        return mse_testbatch_linear_LOSS

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.cfg.optimizer.type == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay)
        elif self.cfg.optimizer.type == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay)
        elif self.cfg.optimizer.type == 'SGD':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.cfg.optimizer.lr,
                                  momentum=0.937,
                                  weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.schedule:
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=self.cfg.trainer.epochs // 1)

            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return {'optimizer': optimizer}

    def on_before_optimizer_step(self, optimizer):
        # inspect (unscaled) gradients here
        self.log_dict(grad_norm(self, norm_type=2))

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        self.model.init_beliefs(batch['initial_state'])
        preds = torch.zeros(batch['initial_state'].shape[0],
                            self.model.state_dim,
                            batch['targets'].shape[-1]).to(self.device)

        for t in range(0, self.cfg.data.val_seq_len):
            preds[:, :, [t]] = self.model(batch['inputs'][:, :, [t]])

        preds = preds[:, self.target_metric_mask, :]
        targets: torch.Tensor = batch['targets']

        # Inverse data
        # if self.cfg.metric.inverse_transforms:
        #     # INFO:
        #     pred = self.trainer.datamodule.val.inverse_data(pred.cpu())
        #     target = self.trainer.datamodule.val.inverse_data(target.cpu())
        return dict(preds=preds, targets=targets, masks=batch['mask'])
