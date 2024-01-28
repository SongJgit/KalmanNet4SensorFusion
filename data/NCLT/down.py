#!/usr/bin/env python

import sys
import os
import subprocess
import argparse

base_dir = 'http://robots.engin.umich.edu/nclt'
base_dir = 'https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu'
dates = [];
dates.append('2012-01-08')
dates.append('2012-01-15')
dates.append('2012-01-22')
dates.append('2012-02-02')
dates.append('2012-02-04')
dates.append('2012-02-05')
dates.append('2012-02-12')
dates.append('2012-02-18')
dates.append('2012-02-19')
dates.append('2012-03-17')
dates.append('2012-03-25')
dates.append('2012-03-31')
dates.append('2012-04-29')
dates.append('2012-05-11')
dates.append('2012-05-26')
dates.append('2012-06-15')
dates.append('2012-08-04')
dates.append('2012-08-20')
dates.append('2012-09-28')
dates.append('2012-10-28')
dates.append('2012-11-04')
dates.append('2012-11-16')
dates.append('2012-11-17')
dates.append('2012-12-01')
dates.append('2013-01-10')
dates.append('2013-02-23')
dates.append('2013-04-05')

def ensure_output_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def main (args):

    getopt = argparse.ArgumentParser(description='Download NCLT dataset')
    getopt.add_argument('--all', action='store_true',
            help='Download all data types')
    getopt.add_argument('--lb3', action='store_true',
            help='Download Ladybug3 Images')
    getopt.add_argument('--sen', action='store_true',
            help='Download sensor data (includes odometry)')
    getopt.add_argument('--vel', action='store_true',
            help='Download velodyne data')
    getopt.add_argument('--hokuyo', action='store_true',
            help='Download hokuyo data')
    getopt.add_argument('--gt', action='store_true',
            help='Download ground truth')
    getopt.add_argument('--gt_cov', action='store_true',
            help='Download ground truth covariance')
    getopt.add_argument('--date', help='Download specific date')
    args = getopt.parse_args()

    if not args.all and not args.lb3 and not args.sen and not args.vel and not args.gt and not args.gt_cov:
        print ("No data type specified. Use --help to see options.")

    for date in dates:
        if args.date is not None and args.date != date:
            continue
        if args.lb3 or args.all:
            ensure_output_dir('images')
            cmd = ['wget', '--continue',
                   '%s/images/%s_lb3.tar.gz' % (base_dir, date),
                   '-P', 'images']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)
        if args.sen or args.all:
            ensure_output_dir('sensor_data')
            cmd = ['wget', '--continue',
                   '%s/sensor_data/%s_sen.tar.gz' % (base_dir, date),
                   '-P', 'sensor_data']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)
        if args.vel or args.all:
            ensure_output_dir('velodyne_data')
            cmd = ['wget', '--continue',
                   '%s/velodyne_data/%s_vel.tar.gz' % (base_dir, date),
                   '-P', 'velodyne_data']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)
        if args.hokuyo or args.all:
            ensure_output_dir('hokuyo_data')
            cmd = ['wget', '--continue',
                   '%s/hokuyo_data/%s_hokuyo.tar.gz' % (base_dir, date),
                   '-P', 'hokuyo_data']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)
        if args.gt or args.all:
            ensure_output_dir('ground_truth')
            cmd = ['wget', '--continue',
                   '%s/ground_truth/groundtruth_%s.csv' % (base_dir, date),
                   '-P', 'ground_truth']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)
        if args.gt_cov or args.all:
            ensure_output_dir('ground_truth_cov')
            cmd = ['wget', '--continue',
                   '%s/covariance/cov_%s.csv' % (base_dir, date),
                   '-P', 'ground_truth_cov']
            print ("Calling: ", ' '.join(cmd))
            subprocess.call(cmd)

    return 0

if __name__ == '__main__':
    sys.exit (main(sys.argv))