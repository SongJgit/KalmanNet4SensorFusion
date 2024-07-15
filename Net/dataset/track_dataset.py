from __future__ import annotations

from typing import Any, Dict, List, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from Net.utils import FILTERDATASET

class TrackDataModule(pl.LightningDataModule):

    def __init__(self, cfg, use_transform: bool = False):
        super().__init__()
        # self.data_dir = cfg.data.data_root
        self.cfg = cfg
        self.use_transform = use_transform

    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:

            # self.train = eval(self.cfg.data.type)(self.data_dir, 'train',
            #                                       self.cfg.data.train_seq_len,
            #                                       self.use_transform)
            # self.val = eval(self.cfg.data.type)(self.data_dir, 'val',
            #                                     self.cfg.data.val_seq_len,
            #                                     self.use_transform)
            self.train = FILTERDATASET.build(self.cfg.data.train_dataset)
            self.val = FILTERDATASET.build(self.cfg.data.val_dataset)

            print('train set size:', len(self.train))
            print('val set size:', len(self.val))

        if stage == 'test':
            self.test = FILTERDATASET.build(self.cfg.data.test_dataset)
            print('test set size:', len(self.test))

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.cfg.data.batch_size,
                          num_workers=self.cfg.data.num_workers,
                          drop_last=False,
                          shuffle=self.cfg.data.shuffle,
                          collate_fn=self.cfg.data.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.cfg.data.batch_size,
                          num_workers=self.cfg.data.num_workers,
                          drop_last=False,
                          collate_fn=self.cfg.data.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.cfg.data.batch_size,
                          num_workers=self.cfg.data.num_workers,
                          drop_last=False,
                          collate_fn=self.cfg.data.collate_fn)
