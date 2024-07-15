from __future__ import annotations

import os
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor


def training_info() -> str:
    today = datetime.today()
    now = datetime.now()
    str_today = today.strftime('%m.%d.%y')
    str_now = now.strftime('%H:%M:%S')
    str_time = ' '.join([str_today, str_now])
    print(f'Current Time = {str_time}')
    return str_time


def generate_save_dir(root: Optional[str] = None,
                      project: str = 'project',
                      name: Optional[str] = None,
                      mode: str = 'train') -> Dict:
    if not root:
        root = os.getcwd()

    project_root = osp.join(root, project, name + '_v0')
    save_dirs: Dict[str, str] = {'experiments_dir': project_root}
    save_dirs['new_name'] = name + '_v0'

    while osp.exists(project_root):
        suffix = int(project_root.split('_v')[-1]) + 1
        prefix = project_root.split('_')[:-1]
        prefix.append(f'v{suffix}')
        project_root = '_'.join(prefix)
        save_dirs['new_name'] = name + f'_v{suffix}'

    save_dirs['weight_dir'] = osp.join(project_root,
                                       'checkpoints')  # type: ignore

    save_dirs['config_dir'] = osp.join(project_root, 'configs')  # type: ignore

    save_dirs['eval_dir'] = osp.join(project_root,
                                     'eval_results')  # type: ignore

    save_dirs['log_images_dir'] = osp.join(project_root,
                                           'log_images')  # type: ignore
    save_dirs['log_metrics_dir'] = osp.join(project_root, 'log_metrics')

    # eval_sub_dir = osp.join(eval_dir, 'imgs')

    if mode == 'train':
        for key, dir in save_dirs.items():
            if not osp.exists(dir) and key.endswith('dir'):
                os.makedirs(dir)

    # save_dir["eval_sub_dir"] = eval_sub_dir
    return save_dirs

