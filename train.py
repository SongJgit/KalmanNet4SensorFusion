import argparse
import os.path as osp
from typing import Dict

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config, DictAction
from torch import Tensor

from Net.dataset.track_dataset import TrackDataModule
from Net.utils import (MODELS, generate_save_dir,
                           training_info)


def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()
    torch.manual_seed(3407)
    save_dir: Dict = generate_save_dir(root='./runs',
                                       project=cfg.logger.project,
                                       name=cfg.logger.name)
    cfg.logger.name = save_dir['new_name']
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))

    data_module = TrackDataModule(
        cfg, use_transform=cfg.data.transforms.use_transform)
    data_module.setup()
    # model.

    model = MODELS.build(
        dict(type=cfg.trainer.type, cfg=cfg, save_dir=save_dir))
    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_monitor = ModelCheckpoint(
        dirpath=save_dir['weight_dir'],
        filename='{epoch}-{val_loss:.2f}-{val_MSE_dB:.2f}',
        mode='min',
        save_top_k=10,
        monitor='val_MSE_dB')
    callbacks = [lr_monitor, model_monitor]
    wandb_logger = WandbLogger(project=cfg.logger.project,
                               name=cfg.logger.name,
                               offline=cfg.logger.offline)
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=cfg.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        detect_anomaly=cfg.trainer.detect_anomaly,
        callbacks=callbacks,
        devices=cfg.trainer.device,
        num_sanity_val_steps=0,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch if cfg.trainer.check_val_every_n_epoch is not None else 1

    )
    trainer.fit(model, datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='KalmanNet',
        description='Dataset, training and network parameters')
    parser.add_argument('--config',
                        '--cfg',
                        type=str,
                        metavar='config',
                        help='model and seq ')

    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(cfg)
    main(args, cfg)
