# Copyright (C) Huangying Zhan 2019. All rights reserved.

import os
import argparse

from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--seq', type=str, required=True)
parser.add_argument('--align', type=str, default=None, required=False)

args = parser.parse_args()

eval_tool = KittiEvalOdom()

gt_path = args.gt_path
result_path = args.result_path
seq = args.seq
align = args.align

save_dir = os.path.join('./results', seq)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

eval_tool.eval(gt_path, result_path, save_dir, seq, align)
