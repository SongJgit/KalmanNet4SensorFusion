batch_size = 256
data = dict(
    batch_size=256,
    collate_fn='Net.dataset.nclt_dataset.nclt_collate',
    num_workers=4,
    shuffle=True,
    test_dataset=dict(
        data_path='./data/NCLT/test.pt',
        seq_len=None,
        type='Net.dataset.nclt_dataset.NCLTDataset'),
    train_dataset=dict(
        data_path='./data/NCLT/train.pt',
        seq_len=50,
        type='Net.dataset.nclt_dataset.NCLTDataset'),
    transforms=dict(use_transform=False),
    val_dataset=dict(
        data_path=[
            './data/NCLT/test.pt',
        ],
        seq_len=None,
        type='Net.dataset.nclt_dataset.NCLTDataset'))
detach_step = 2
device = [
    6,
]
logger = dict(
    name='Wheel_GPS_l50w4step2_origin_Baseline_v0',
    offline=False,
    project='new_exp')
loss_name = 'MSE'
metric = dict(
    inverse_transforms=False,
    metric_dim_names=[
        'X',
        'Y',
    ],
    num_metric_dims=2,
    pred_metric_mask=[
        True,
        True,
        False,
        False,
        False,
        False,
    ],
    target_metric_mask=[
        True,
        True,
    ])
model = dict(
    in_mult_KNet=5,
    out_mult_KNet=40,
    params=dict(
        T=1,
        constant_noise=True,
        noise_q2=1,
        noise_r2=1,
        observation_axis=[
            0,
            3,
        ],
        state_dim=6,
        type='Net.params.NCLTFusionWheelGPSParams'),
    type='Net.models.FusionKalmanNet')
optimizer = dict(lr=0.001, schedule=False, type='Adam', weight_decay=0.0001)
params = dict(
    T=1,
    constant_noise=True,
    noise_q2=1,
    noise_r2=1,
    observation_axis=[
        0,
        3,
    ],
    state_dim=6,
    type='Net.params.NCLTFusionWheelGPSParams')
slide_win_size = 4
test_dataset = dict(
    data_path='./data/NCLT/test.pt',
    seq_len=None,
    type='Net.dataset.nclt_dataset.NCLTDataset')
test_path = './data/NCLT/test.pt'
train_dataset = dict(
    data_path='./data/NCLT/train.pt',
    seq_len=50,
    type='Net.dataset.nclt_dataset.NCLTDataset')
train_path = './data/NCLT/train.pt'
train_seq_len = 50
trainer = dict(
    batch_size=256,
    check_val_every_n_epoch=5,
    detach_step=2,
    detect_anomaly=False,
    device=[
        6,
    ],
    epochs=50,
    gradient_clip_algorithm='norm',
    gradient_clip_val=1,
    loss_name='MSE',
    slide_win_size=4,
    type='Net.trainer.LitFusionKalmanNet')
val_dataset = dict(
    data_path=[
        './data/NCLT/test.pt',
    ],
    seq_len=None,
    type='Net.dataset.nclt_dataset.NCLTDataset')
val_path = [
    './data/NCLT/test.pt',
]
