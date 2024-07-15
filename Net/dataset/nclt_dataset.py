from typing import Any, Dict, List, Tuple
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

def get_nclt_initial_est(ground_truth, theta_gt):
    """_summary_

    Args:
        ground_truth (_type_): [seq_len, 2] means the x,y at each time.
        theta_gt (_type_): [seq_len ,1] means the theta at each time.

    Returns:
        _type_: _description_
    """
    return torch.tensor(
        [ground_truth[0][0], ground_truth[0][1], 0, 0, theta_gt[0], 0])


def nclt_split(T: int, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    T = T + 1  # 1 for init
    splited = []
    for data in tqdm(dataset):
        filtered_gps = data['filtered_gps']
        imu_sensor = data['imu_sensor']
        filtered_wheel = data['filtered_wheel']
        theta_gt = data['theta_gt']
        ground_truth = data['ground_truth']
        gps = data['gps']
        wheel = data['wheel']
        data_date = data['data_date']
        description = data['description']

        sub_gt = torch.split(ground_truth, T)[:]
        sub_imu = torch.split(imu_sensor, T)[:]
        sub_filtered_gps = torch.split(filtered_gps, T)[:]
        sub_filtered_wheel = torch.split(filtered_wheel, T)[:]
        sub_theta_gt = torch.split(theta_gt, T)[:]
        sub_gps = torch.split(gps, T)[:]
        sub_wheel = torch.split(wheel, T)[:]

        for i in range(len(sub_gt)): 
            initial_state = get_nclt_initial_est(sub_gt[i], sub_theta_gt[i])
            processed_data = dict(filtered_gps=sub_filtered_gps[i][1:],
                                  imu_sensor=sub_imu[i][1:],
                                  filtered_wheel=sub_filtered_wheel[i][1:],
                                  theta_gt=sub_theta_gt[i][1:],
                                  ground_truth=sub_gt[i][1:],
                                  wheel=sub_wheel[i][1:],
                                  gps=sub_gps[i][1:],
                                  initial_state=initial_state,
                                  data_date=data_date,
                                  description=description,
                                  sub_id=data_date + f'_{i}')
            splited.append(processed_data)
    return splited

class NCLTDataset(Dataset):

    def __init__(
        self,
        data_path=str| List[str],
        seq_len: int | None = None,
    ):
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.data_path = data_path
        self.seq_len = seq_len
        self.data = []
        for p in self.data_path:
            self.data.extend(torch.load(p))
        if self.seq_len is not None:
            num = len(self.data)
            self.data = nclt_split(self.seq_len, self.data)
            print(
                f'A total of {num} sequences, cut into {len(self.data)} subsequences of length {self.seq_len}'
            )

        # -> [num_state, seq_len]
        self.filtered_gps = [
            data['filtered_gps'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.imu_sensor = [
            data['imu_sensor'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.filtered_wheel = [
            data['filtered_wheel'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.theta_gt = [
            data['theta_gt'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.ground_truth = [
            data['ground_truth'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.wheel = [
            data['wheel'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.gps = [
            data['gps'].to(torch.float32).transpose(-1, -2)
            for data in self.data
        ]
        self.initial_state = [
            data['initial_state'].to(torch.float32)[..., None]
            for data in self.data
        ]
        self.data_date = [data['data_date'] for data in self.data]
        if self.seq_len is not None:
            self.sub_id = [data['sub_id'] for data in self.data]
        else:
            self.sub_id = ['-1' for data in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        # -> [num_state, seq_len]
        filtered_gps = self.filtered_gps[index]
        imu_sensor = self.imu_sensor[index]
        filtered_wheel = self.filtered_wheel[index]
        ground_truth = self.ground_truth[index]
        theta_gt = self.theta_gt[index]
        gps = self.gps[index]
        wheel = self.wheel[index]
        initial_state = self.initial_state[index]
        data_date = self.data_date[index]
        sub_id = self.sub_id[index]

        return {
            'filtered_gps': filtered_gps,
            'imu_sensor': imu_sensor,
            'filtered_wheel': filtered_wheel,
            'ground_truth': ground_truth,
            'theta_gt': theta_gt,
            'initial_state': initial_state,
            'gps': gps,
            'wheel': wheel,
            'data_date': data_date,
            'sub_id': sub_id
        }

def nclt_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """_summary_

    Args:
        batch (List[Dict[str, Any]]): [num_state, seq_len] in Dict.

    Returns:
        Dict[str, Any]: _description_
    """
    filtered_gps = [data['filtered_gps'] for data in batch]
    imu_sensor = [data['imu_sensor'] for data in batch]
    filtered_wheel = [data['filtered_wheel'] for data in batch]
    ground_truth = [data['ground_truth'] for data in batch]
    theta_gt = [data['theta_gt'] for data in batch]
    wheel = [data['wheel'] for data in batch]
    gps = [data['gps'] for data in batch]
    initial_state = [data['initial_state'] for data in batch]
    max_length = max([val.shape[-1] for val in filtered_gps])
    data_date = [data['data_date'] for data in batch]
    sub_id = [data['sub_id'] for data in batch]
    masks = []
    padded_filtered_gps = []
    padded_imu = []
    padded_filtered_wheel = []
    padded_gt = []
    padded_theta_gt = []
    padded_gps = []
    padded_wheel = []
    for filt_gps, imu, filt_wheel, gt, theta, _gps, _wheel in zip(
            filtered_gps, imu_sensor, filtered_wheel, ground_truth, theta_gt,
            gps, wheel):
        pad_len = max_length - filt_gps.shape[-1]  # type:ignore
        padded_filtered_gps.append(F.pad(filt_gps, (0, pad_len)))
        padded_imu.append(F.pad(imu, (0, pad_len)))
        padded_filtered_wheel.append(F.pad(filt_wheel, (0, pad_len)))
        padded_gt.append(F.pad(gt, (0, pad_len)))
        padded_theta_gt.append(F.pad(theta, (0, pad_len)))
        padded_gps.append(F.pad(_gps, (0, pad_len)))
        padded_wheel.append(F.pad(_wheel, (0, pad_len)))
        mask = torch.ones(filt_gps.shape[-1])  # type:ignore
        mask = F.pad(mask, (0, pad_len))
        masks.append(mask.to(torch.bool))
    return {
        'filtered_gps': default_collate(padded_filtered_gps),
        'imu': default_collate(padded_imu),
        'filtered_wheel': default_collate(padded_filtered_wheel),
        'ground_truth': default_collate(padded_gt),
        'theta_gt': default_collate(padded_theta_gt),
        'gps': default_collate(padded_gps),
        'wheel': default_collate(padded_wheel),
        'initial_state': default_collate(initial_state),
        'mask': default_collate(masks),
        'data_date': data_date,
        'sub_id': sub_id
    }
