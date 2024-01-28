import os.path as osp

import numpy as np
import pandas as pd
import scipy
import torch

# Modified from https://github.com/AbhinavA10/mte546-project
def local_to_gps_coord(x: list, y:list):
    """Convert list of local frame coords to GPS latitude/longitude
    Parameters: local frame coords (x,y) = (North, East) [meters]
    Returns: GPS lat/lon in degrees
    """
    r_ns, r_ew, LAT_0, LON_0 = _compute_gps_conversion_params()
    # Convert linearized local frame to GPS
    x = x - 76.50582406697139 # Tuned offset to adjust with Google earth
    y = y - 108.31373031919006 # Tuned offset to adjust with Google earth
    lat = np.arcsin(x/r_ns) + LAT_0
    lon = np.arcsin(y/(r_ew*np.cos(LAT_0))) + LON_0
    lat = np.rad2deg(lat) # Latitude, in degrees
    lon = np.rad2deg(lon) # Longitude, in degrees
    return (lat,lon)


def _format_lat_lon(lat: list, lon:list):
    """Format coords for KML file"""
    l = ["            {},{},1".format(lo, la) for la, lo in zip(lat, lon)]
    return "\n".join(l)


def calculate_hz(sensor_name: str, timestamps: list):
    """Calculate Hz of Sensor Data"""
    length = timestamps[-1] - timestamps[0]
    average_timestep = length / len(timestamps)
    hz = 1 / average_timestep
    print(f'{sensor_name} data, Hz: {hz}')


def gps_to_local_coord(lat: list, lon: list):
    """Convert list of latitude/longitude to local frame
    Parameters: latitude, longitudes in radians
    Returns: local frame coords (x,y) = (North, East) [meters]
    """
    r_ns, r_ew, LAT_0, LON_0 = _compute_gps_conversion_params()
    # Convert GPS coordinates to linearized local frame
    x = np.sin(lat - LAT_0) * r_ns  # North
    y = np.sin(lon - LON_0) * r_ew * np.cos(LAT_0)  # East
    return (x, y)


def _compute_gps_conversion_params():
    # Compute radii of Earth at origin of linearization
    LAT_0 = 0.738167915410646  # in radians, lat[0]
    LON_0 = -1.46098650670922  # in radians, lon[0]
    re = 6378135  # Earth Equatorial Radius [m]
    rp = 6356750  # Earth Polar Radius [m]
    r_ns = pow(re * rp, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 3 / 2)
    r_ew = pow(re, 2) / pow(
        pow(re * np.cos(LAT_0), 2) + pow(rp * np.sin(LAT_0), 2), 1 / 2)
    return (r_ns, r_ew, LAT_0, LON_0)


def read_gps(data_dir, data_date, use_rtk=False):
    filename = 'gps_rtk.csv' if use_rtk else 'gps.csv'

    filepath = osp.join(data_dir, 'sensor', data_date, filename)
    print(f'{filepath}')
    gps = np.loadtxt(filepath, delimiter=',')
    gps = np.delete(
        gps, np.where((gps[:, 1] < 3))[0],
        axis=0)  # filter out rows where fix_mode < 3 (invalid data)
    # perform conversion from lat/lon to local frame
    t = gps[:, 0]
    lat = gps[:, 3]
    lng = gps[:, 4]
    # t = t-t[0] # make timestamps relative
    t = t / 1000000
    if use_rtk:
        calculate_hz('GPS RTK', t)  # 2.5 Hz
    else:
        calculate_hz('GPS', t)  # 4 Hz
    x, y = gps_to_local_coord(lat, lng)  # North,East
    x = x + 76.50582406697139  # Offset to adjust with ground truth initial position
    y = y + 108.31373031919006  # Offset to adjust with ground truth initial position
    gps_data = np.array([])
    gps_data = np.vstack((t, x, y)).T
    # Filter data to campus map area
    gps_data = np.delete(
        gps_data,
        np.where((gps_data[:, 1] < -350) | (gps_data[:, 1] > 120))[0],  #
        axis=0)  # x
    gps_data = np.delete(gps_data,
                         np.where((gps_data[:, 2] < -750)
                                  | (gps_data[:, 2] > 150))[0],
                         axis=0)  # y

    # # Manually filter poor GPS readings - tuned for 2013-04-05 data
    # if dataset_date == "2013-04-05":
    #     gps_data = np.delete(gps_data, slice(2350,2650), axis=0)
    #     gps_data = np.delete(gps_data, slice(13800,14550), axis=0)

    # timestamp | x | y
    return gps_data


def read_imu(data_dir, data_date):
    # filepath = f"dataset/{dataset_date}/ms25.csv"
    filepath = osp.join(data_dir, 'sensor', data_date, 'ms25.csv')

    ms25 = np.loadtxt(filepath, delimiter=',')
    accel_x_OG = ms25[:, 4]
    accel_y_OG = ms25[:, 5]
    rot_h_OG = ms25[:, 9]
    t = ms25[:, 0]

    # Attempt to Remove bias by estimating it using the first 2 seconds of stationary data
    # print("a_x Bias", np.average(accel_x_OG[:100]))
    # print("a_y Bias", np.average(accel_y_OG[:100]))
    # accel_x_OG -= np.average(accel_x_OG[:100])
    # accel_y_OG -= np.average(accel_y_OG[:100])

    # apply rolling average to accelerations, to smooth noise
    accel_x_df = pd.DataFrame(accel_x_OG)
    accel_x_rolling = accel_x_df.rolling(50, min_periods=1).mean()
    accel_x_rolling = accel_x_rolling.to_numpy().flatten()
    accel_y_df = pd.DataFrame(accel_y_OG)
    accel_y_rolling = accel_y_df.rolling(50, min_periods=1).mean()
    accel_y_rolling = accel_y_rolling.to_numpy().flatten()

    # Relative timestamps
    # t = t-t[0]
    t = t / 1000000
    calculate_hz('IMU Accel and Omega', t)  # 47 Hz

    # have the following format:
    # timestamp | ax_robot | ay_robot | omega
    imu_data = np.array([])
    imu_data = np.vstack((t, accel_x_rolling, accel_y_rolling, rot_h_OG)).T
    return imu_data


def read_ground_truth(dir, dataset_date, truncation=-1):
    """Read Ground Truth Data
    Parameters: dataset date
    Returns: np.ndarray([timestamp, x, y, yaw])
    """
    # filepath_cov = f"{dataset_date}/odometry_cov_100hz.csv"
    filepath_cov = osp.join(dir, 'sensor', dataset_date,
                            'odometry_cov_100hz.csv')

    # data_date = osp.split(osp.dirname(data_dir))[-1]
    # filepath_gt = f"dir/ground_truth/groundtruth_{dataset_date}.csv"
    filepath_gt = osp.join(dir, 'ground_truth',
                           f'groundtruth_{dataset_date}.csv')

    gt = np.loadtxt(filepath_gt, delimiter=',')
    cov = np.loadtxt(filepath_cov, delimiter=',')
    # the first and second elements is nan
    gt = gt[2:truncation, :]
    cov = cov[:truncation, :]

    t = cov[:, 0]
    # t = cov[:,0]
    interp = scipy.interpolate.interp1d(gt[:, 0],
                                        gt[:, 1:],
                                        kind='nearest',
                                        axis=0,
                                        fill_value='extrapolate')
    pose_gt = interp(t)
    # t = t-t[0] # Make timestamps relative
    t = t / 1000000
    x = pose_gt[:, 0]  # North
    y = pose_gt[:, 1]  # East
    yaw = pose_gt[:, 5]

    # print("Ground truth x0: ", x[0])
    # print("Ground truth y0: ", y[0])

    calculate_hz('Ground Truth', t)  # 107 Hz

    ground_truth = np.array([])
    ground_truth = np.vstack((t, x, y, yaw)).T

    # timestamp | x | y | yaw
    return ground_truth


def read_euler(data_dir, data_date):
    filepath = osp.join(data_dir, 'sensor', data_date, 'ms25_euler.csv')

    euler = np.loadtxt(filepath, delimiter=',')

    t = euler[:, 0]
    h_OG = euler[:, 3]  # heading (z)

    # Relative timestamps
    # t = t-t[0]
    t = t / 1000000
    calculate_hz('IMU Euler', t)  # 47 Hz

    # have the following format:
    # timestamp | ax_robot | ay_robot | omega
    euler_data = np.array([])
    euler_data = np.vstack((t, h_OG)).T
    return euler_data


def read_wheels(data_dir, data_date):
    # filepath = f"dataset/{dataset_date}/wheels.csv"
    filepath = osp.join(data_dir, 'sensor', data_date, 'wheels.csv')

    wheel_vel = np.loadtxt(filepath, delimiter=',')

    t = wheel_vel[:, 0]

    left_wheel_vel = wheel_vel[:, 1]
    right_wheel_vel = wheel_vel[:, 2]
    robot_vel = 0.5 * (wheel_vel[:, 1] + wheel_vel[:, 2])

    # Relative timestamps
    # t = t-t[0]
    t = t / 1000000
    calculate_hz('Wheel Odometry', t)  # 37 Hz

    # have the following format:
    # timestamp | robot velocity | left wheel velocity | right wheel velocity
    wheel_data = np.array([])
    wheel_data = np.vstack((t, robot_vel, left_wheel_vel, right_wheel_vel)).T
    return wheel_data


def find_nearest_index(array: np.ndarray,
                       time):  # array of timesteps, time to search for
    """Find closest time in array, that has already passed"""
    diff_arr = array - time
    idx = np.where(diff_arr <= 0, diff_arr, -np.inf).argmax()
    # [-0.02 +0.02 +2] becomes  [-0.02  -inf -inf]
    return idx


def preprocess(dir, data_date, use_rtk=False):

    ground_truth = read_ground_truth(dir, data_date)
    gps_data = read_gps(dir, data_date, use_rtk)
    imu_data = read_imu(dir, data_date)
    euler_data = read_euler(dir, data_date)
    wheel_data = read_wheels(dir, data_date)  # 37 Hz

    TRUNCATION_END = -1

    ground_truth = ground_truth[:TRUNCATION_END, :]
    assert not np.any(np.isnan(ground_truth))
    gps_data = gps_data[:TRUNCATION_END, :]
    imu_data = imu_data[:TRUNCATION_END, :]
    euler_data = euler_data[:TRUNCATION_END, :]
    wheel_data = wheel_data[:TRUNCATION_END, :]

    x_true = ground_truth[:, 1]  # North
    y_true = ground_truth[:, 2]  # East
    theta_true = ground_truth[:, 3]  # Heading
    true_times = ground_truth[:, 0]

    KALMAN_FILTER_RATE = 1
    dt = 1 / KALMAN_FILTER_RATE

    t = np.arange(ground_truth[0, 0], ground_truth[-1, 0], dt)
    N = len(t)
    x_true_arr = np.zeros([N])  # Keep track of corresponding truths
    y_true_arr = np.zeros([N])
    theta_true_arr = np.zeros([N])
    x_true_arr[0] = x_true[0]  # initial state
    y_true_arr[0] = y_true[0]
    theta_true_arr[0] = theta_true[0]

    a_x = imu_data[:, 1]
    a_y = imu_data[:, 2]
    omega = imu_data[:, 3]
    theta_imu = euler_data[:, 1]

    gps_x = gps_data[:, 1]
    gps_y = gps_data[:, 2]
    robot_vel = wheel_data[:, 1]
    v_left_wheel = wheel_data[:, 2]
    v_right_wheel = wheel_data[:, 3]

    gps_times = gps_data[:, 0]
    imu_times = imu_data[:, 0]
    euler_times = euler_data[:, 0]
    wheel_times = wheel_data[:, 0]

    wheel_counter = 0
    gps_counter = 0
    imu_counter = 0
    ground_truth_counter = 0
    euler_counter = 0
    prev_gps_counter = -1
    prev_wheel_counter = -1

    imu_sensor = []
    gps = []  # for no filtered gps
    filtered_gps = []  # for filtered gps
    gt = []
    theta_gt = []
    filtered_wheel = []
    wheel = []
    x_est = np.array([x_true[0], y_true[0], 0, 0, theta_true[0], 0])
    for k in range(1, len(t)):
        # IMu
        ax = a_x[imu_counter]
        ay = a_y[imu_counter]
        theta = theta_imu[euler_counter]
        omega_imu = omega[imu_counter]
        imu_sensor.append(torch.tensor([ax, ay, theta, omega_imu]))
        gps.append(torch.tensor([gps_x[gps_counter], gps_y[gps_counter]]))
        wheel.append(
            torch.tensor([
                v_left_wheel[wheel_counter], v_right_wheel[wheel_counter],
                theta, omega_imu
            ]))

        imu_counter = find_nearest_index(imu_times, t[k])
        euler_counter = find_nearest_index(euler_times, t[k])
        ground_truth_counter = find_nearest_index(true_times, t[k])
        gps_counter = find_nearest_index(gps_times, t[k])
        wheel_counter = find_nearest_index(wheel_times, t[k])

        if gps_counter != prev_gps_counter:
            filtered_gps.append(
                torch.tensor([gps_x[gps_counter], gps_y[gps_counter]]))
            prev_gps_counter = gps_counter
        else:
            filtered_gps.append(torch.tensor([torch.nan, torch.nan]))

        if wheel_counter != prev_wheel_counter:
            filtered_wheel.append(
                torch.tensor([
                    v_left_wheel[wheel_counter], v_right_wheel[wheel_counter]
                ]))
            prev_wheel_counter = wheel_counter
        else:
            filtered_wheel.append(torch.tensor([torch.nan, torch.nan]))
        gt.append(
            torch.tensor(
                [x_true[ground_truth_counter], y_true[ground_truth_counter]]))
        theta_gt.append(torch.tensor([theta_true[ground_truth_counter]]))

    processed_data = dict(
        filtered_gps=torch.vstack(filtered_gps),
        imu=torch.vstack(imu_sensor),
        filtered_wheel=torch.vstack(filtered_wheel),
        theta_gt=torch.vstack(theta_gt),
        ground_truth=torch.vstack(gt),
        initial_state=torch.from_numpy(x_est),
        gps=torch.vstack(gps),
        wheel=torch.vstack(wheel),
        description='filtered data for correction, else for input.',
        data_date=data_date,
    )
    return processed_data


if __name__ == '__main__':
    import argparse
    opt = argparse.ArgumentParser(description='preprocess nclt')
    opt.add_argument('--use_rtk', action='store_true', help='use rtk')
    opt.add_argument('--random_split', action='store_true', help='use rtk')
    opt.add_argument(
        '--random_seed',
        default=45,
        type=int,
        help=  # noqa: E251, E261
        'work with random_seed is true, seed ensures that rtk and non-rtk generate the same splits'
    )
    args = opt.parse_args()

    import os

    from tqdm import tqdm
    listdir = os.listdir('../data/sensor/')
    processed = []
    dir = '../data/'

    if args.use_rtk:
        filename = '_rtk'
    else:
        filename = ''
    for date in tqdm(listdir):
        print(date)
        processed.append(preprocess(dir, date, args.use_rtk))
    t = range(processed.__len__())
    if args.random_split:
        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(args.random_seed)
        train_dataset, test_dataset, val_dataset = random_split(
            processed, [0.8, 0.1, 0.1])
    else:
        # For easy comparison with mte-546-project, specify that the
        # test contains '2013-04-05' and the rest are random cuts.
        train = [
            '2012-08-20', '2012-02-19', '2012-08-04', '2012-03-25',
            '2012-06-15', '2012-01-22', '2012-01-15', '2012-01-08',
            '2012-02-04', '2012-02-02', '2012-05-11', '2012-04-29',
            '2012-02-18', '2012-03-31', '2012-12-01', '2012-09-28',
            '2012-02-05', '2012-11-17', '2012-02-12', '2012-05-26',
            '2012-03-17', '2012-10-28'
        ]
        val = ['2013-01-10', '2013-02-23']
        test = ['2012-11-16', '2013-04-05', '2012-11-04']
        train_dataset = []
        test_dataset = []
        val_dataset = []
        for data in processed:
            if data['data_date'] in test:
                test_dataset.append(data)
            elif data['data_date'] in val:
                val_dataset.append(data)
            else:
                train_dataset.append(data)
    torch.save(list(train_dataset), f'./data/NCLT/processed/train{filename}.pt')
    torch.save(list(test_dataset), f'./data/NCLT/processed/test{filename}.pt')
    torch.save(list(val_dataset), f'./data/NCLT/processed/val{filename}.pt')
