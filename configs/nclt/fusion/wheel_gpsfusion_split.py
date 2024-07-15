from mmengine.config import read_base

from Net.dataset.nclt_dataset import NCLTDataset, nclt_collate
from Net.models import FusionKalmanNet, FusionSplitKalmanNet
from Net.params import NCLTFusionIMUGPSParams, NCLTFusionWheelGPSParams
from Net.trainer import LitFusionKalmanNet

# mypy: ignore-errors
test_path = './data/NCLT/processed/test.pt'
val_path = './data/NCLT/processed/val.pt'
val_path = [val_path]
train_path = './data/NCLT/processed/train.pt'
device = [0]
train_seq_len = 100
batch_size = 256
slide_win_size =100
detach_step = 100
loss_name = 'SmoothL1'
params = dict(type=NCLTFusionWheelGPSParams,
              T=1,
              constant_noise=True,
              state_dim=6,
              observation_axis=[0, 3],
              noise_q2=1,
              noise_r2=1)
metric = dict(
    pred_metric_mask=[True, True, False, False, False, False],
    target_metric_mask=[True, True],  # must match state
    # number of state to compute metric ,metric_dims must be equal to number of mask'True
    num_metric_dims=2,
    metric_dim_names=['X', 'Y'],
    inverse_transforms=False)

logger = dict(
    project='new_exp',
    name=f'Wheel_GPS_l{train_seq_len}w{slide_win_size}step{detach_step}_split',
    # offline=False)
    offline=True)
# logger = dict(project='Test', name='Wheel_GPS_rtk_l100w8', offline=True)
optimizer = dict(type='Adam', lr=0.001, weight_decay=1e-4, schedule=False)

val_dataset = dict(
    type=NCLTDataset,
    data_path=val_path,
    seq_len=None,
)
train_dataset = dict(
    type=NCLTDataset,
    data_path=train_path,
    seq_len=train_seq_len,
)
test_dataset = dict(
    type=NCLTDataset,
    data_path=test_path,
    seq_len=None,
)
data = dict(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    # test_seq_len=100,
    # train_seq_len=100,
    # val_seq_len=100,
    transforms=dict(use_transform=False, ),
    collate_fn=nclt_collate)

trainer = dict(
    type=LitFusionKalmanNet,
    detach_step=detach_step,
    epochs=50,
    batch_size=batch_size,
    gradient_clip_val=1,  # int | float | None
    gradient_clip_algorithm='norm',  # str | None
    slide_win_size=slide_win_size,
    device=device,
    detect_anomaly=False,
    check_val_every_n_epoch=5,
    loss_name=loss_name,
)

model = dict(
    type=FusionSplitKalmanNet,
    params=params,
    gru_scale_s=2,
    nGRU=2,
)
