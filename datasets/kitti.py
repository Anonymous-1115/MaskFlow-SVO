import collections
import glob
import os.path
import random
import warnings
import argparse

import numpy as np

warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
import datetime as dt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision.transforms import v2
from utils import *

warnings.simplefilter("ignore", UserWarning)

FramePath = collections.namedtuple(
    'FramePath', ['frame_id', 'timestamp', 'sequence_path', 'left_img_path', 'right_img_path',
                  'left_K', 'right_K', 'baseline', 'left_distortion', 'right_distortion', 'T_wb']
)

Frame = collections.namedtuple(
    'Frame', ['frame_id', 'timestamp', 'left_img', 'right_img', 'baseline', 'height', 'width',
              'left_K', 'right_K', 'T_wb']
)

Transform = collections.namedtuple(
    'Transform', ['T', 'R', 'so3', 'euler_zyx', 't']
)

Data = collections.namedtuple(
    'Data', ['frame1', 'frame2', 'sequence_path', 'intrinsic_layer',
             'transform_12', 'flow_l1l2', 'disp1', 'mask1', 'flow_l2l1', 'disp2', 'mask2']
)


class KITTI(Dataset):
    def __init__(self, cfg, split, transform=False, data_aug=False):
        super().__init__()
        self.kitti_data = []
        self.transform = transform
        self.data_aug = data_aug
        self.height = cfg['model'].network.height
        self.width = cfg['model'].network.width
        self.split = split

        if split == 'train':
            self.kitti_path = cfg['dataset'].training.path
            self.kitti_sequences = cfg['dataset'].training.seq
        elif split == 'validate':
            self.kitti_path = cfg['dataset'].validation.path
            self.kitti_sequences = cfg['dataset'].validation.seq
        elif split == 'predict':
            self.kitti_path = cfg['dataset'].prediction.path
            self.kitti_sequences = cfg['dataset'].prediction.seq
        else:
            raise NotImplementedError()

        self.kitti_sequence_path = os.path.join(self.kitti_path, 'data_odometry_color/dataset/sequences')
        self.kitti_pose_path = os.path.join(self.kitti_path, 'data_odometry_poses/dataset/poses')
        self.kitti_timestamp_root_path = os.path.join(self.kitti_path, 'data_odometry_calib/dataset/sequences')

        # for each sequence
        for seq in tqdm(self.kitti_sequences, desc='Loading sequence: '):
            sequence_path = os.path.join(self.kitti_sequence_path, seq)
            pose_path = os.path.join(self.kitti_pose_path, '{}.txt'.format(seq))
            calib_path = os.path.join(sequence_path, 'calib.txt')
            timestamp_path = os.path.join(self.kitti_timestamp_root_path, seq, 'times.txt')
            cache_root_path = os.path.join(sequence_path, 'cache')  # TODO: ['cache', 'cache_gvo']

            # read timestamp
            timestamps = []
            with open(timestamp_path, 'r') as f:
                for line in f.readlines():
                    timestamp = float(line.strip().split()[0])
                    timestamps.append(timestamp)

            # read calib file (4 cameras, we only use cam2 and cam3)
            K = []
            T_bc = []
            with open(calib_path, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    P = np.reshape(np.array([float(x) for x in value.split()]), (3, 4))
                    K.append(P[0:3, 0:3])

                    # Note: the projection matrix of KITTI brings a point X from the
                    # left rectified camera coordinate system to a point x in the i'th image plane
                    T_cb = np.eye(4)
                    T_cb[0, 3] = P[0, 3] / P[0, 0]
                    T_bc.append(np.linalg.inv(T_cb))
            baseline = T_bc[3][0, 3] - T_bc[2][0, 3]

            # read pose, 'b' means cam0
            poses = []
            try:
                with open(pose_path, 'r') as f:
                    for line in f.readlines():
                        T_wb = np.fromstring(line, dtype=float, sep=' ')
                        T_wb = T_wb.reshape(3, 4)
                        T_wb = np.vstack((T_wb, [0, 0, 0, 1]))
                        poses.append(T_wb)
            except FileNotFoundError:
                for _ in range(len(timestamps)):
                    poses.append(np.eye(4))
                print('Ground truth poses are not available for sequence ' + seq + '.')

            assert len(timestamps) == len(poses)

            # for each frame, read image path and construct data
            frames = []
            frame_list = glob.glob(os.path.join(sequence_path, 'image_2', '*.png'))
            frame_list.sort()
            for path in frame_list:
                frames.append(path[-10:-4])

            # make frame pairs
            left_distortion = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            right_distortion = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            for i in range(len(frames) - 1):
                left_img_path1 = os.path.join(sequence_path, 'image_2', frames[i] + '.png')
                right_img_path1 = os.path.join(sequence_path, 'image_3', frames[i] + '.png')
                left_img_path2 = os.path.join(sequence_path, 'image_2', frames[i + 1] + '.png')
                right_img_path2 = os.path.join(sequence_path, 'image_3', frames[i + 1] + '.png')
                cache_path = os.path.join(cache_root_path, str(frames[i]) + '_' + str(frames[i + 1]) + '.npz')

                frame_pair = {
                    'frame1': FramePath(
                        frame_id=frames[i], timestamp=timestamps[i],
                        sequence_path=sequence_path,
                        left_img_path=left_img_path1, right_img_path=right_img_path1,
                        baseline=baseline, left_K=K[2], right_K=K[3],
                        left_distortion=left_distortion, right_distortion=right_distortion,
                        T_wb=poses[i]
                    ),
                    'frame2': FramePath(
                        frame_id=frames[i + 1], timestamp=timestamps[i + 1],
                        sequence_path=sequence_path,
                        left_img_path=left_img_path2, right_img_path=right_img_path2,
                        baseline=baseline, left_K=K[2], right_K=K[3],
                        left_distortion=left_distortion, right_distortion=right_distortion,
                        T_wb=poses[i + 1]
                    ),
                    'cache': cache_path
                }

                self.kitti_data.append(frame_pair)

        # self.start = 0
        # self.end = -1
        # self.kitti_data = self.kitti_data[self.start:self.end]

    def __len__(self):
        return len(self.kitti_data)

    def __getitem__(self, index):
        data = self.kitti_data[index]
        sequence_path1 = data['frame1'].sequence_path
        sequence_path2 = data['frame2'].sequence_path
        cache_path = data['cache']
        frames = []

        # each sample read a pair of images
        for key, frame_data in data.items():
            if key not in ['frame1', 'frame2']:
                continue

            # read image
            left_img = cv2.imread(frame_data.left_img_path)
            height = left_img.shape[0]
            width = left_img.shape[1]
            if self.height == -1:
                self.height = height
            if self.width == -1:
                self.width = width
            scale_x = self.width / width
            scale_y = self.height / height
            left_img = cv2.resize(left_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            left_img = torch.tensor(left_img, dtype=torch.float32).permute(2, 0, 1) / 255  # (C, H, W), [0, 1]， BGR

            # update height and width
            height = left_img.shape[-2]
            width = left_img.shape[-1]

            right_img = cv2.imread(frame_data.right_img_path)
            right_img = cv2.resize(right_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            right_img = torch.tensor(right_img, dtype=torch.float32).permute(2, 0, 1) / 255  # (C, H, W), [0, 1]， BGR

            # adjust intrinsic
            left_K = frame_data.left_K.copy()
            left_K[0] = left_K[0] * scale_x
            left_K[1] = left_K[1] * scale_y
            left_K = torch.tensor(left_K, dtype=torch.float32)

            right_K = frame_data.right_K.copy()
            right_K[0] = right_K[0] * scale_x
            right_K[1] = right_K[1] * scale_y
            right_K = torch.tensor(right_K, dtype=torch.float32)

            # remains to Tensor
            baseline = torch.tensor(frame_data.baseline, dtype=torch.float32)
            T_wb = torch.tensor(frame_data.T_wb, dtype=torch.float32)

            # construct final data
            frames.append(
                Frame(frame_id=frame_data.frame_id, timestamp=frame_data.timestamp,
                      left_img=left_img, right_img=right_img,
                      height=height, width=width,
                      baseline=baseline, left_K=left_K, right_K=right_K,
                      T_wb=T_wb)
            )

        T_12 = torch.inverse(frames[0].T_wb) @ frames[1].T_wb
        R_12 = T_12[0:3, 0:3]
        t_12 = T_12[0:3, 3]
        r_12 = R.from_matrix(R_12)
        so3_12 = torch.tensor(r_12.as_rotvec(), dtype=torch.float32)
        euler_12 = torch.tensor(r_12.as_euler('zyx', degrees=False), dtype=torch.float32)
        transform_12 = Transform(T=T_12, R=R_12, so3=so3_12, euler_zyx=euler_12, t=t_12)

        T_21 = torch.inverse(frames[1].T_wb) @ frames[0].T_wb
        R_21 = T_21[0:3, 0:3]
        t_21 = T_21[0:3, 3]
        r_21 = R.from_matrix(R_21)
        so3_21 = torch.tensor(r_21.as_rotvec(), dtype=torch.float32)
        euler_21 = torch.tensor(r_21.as_euler('zyx', degrees=False), dtype=torch.float32)
        transform_21 = Transform(T=T_21, R=R_21, so3=so3_21, euler_zyx=euler_21, t=t_21)

        # intrinsic_layer
        v, u = torch.meshgrid(
            torch.arange(0, frames[0].height, dtype=torch.float32),
            torch.arange(0, frames[0].width, dtype=torch.float32),
            indexing='ij'
        )
        v, u = v.contiguous(), u.contiguous()
        fx = frames[0].left_K[0, 0].clone()
        fy = frames[0].left_K[1, 1].clone()
        cx = frames[0].left_K[0, 2].clone()
        cy = frames[0].left_K[1, 2].clone()
        u = (u - cx + 0.5) / fx
        v = (v - cy + 0.5) / fy
        intrinsic_layer = torch.stack((u, v))  # (2, H, W)

        # read cache
        cache = np.load(cache_path)
        flow_l1l2 = torch.tensor(cache['flow_l1l2'], dtype=torch.float32)
        flow_l2l1 = torch.tensor(cache['flow_l2l1'], dtype=torch.float32)
        disp1 = torch.tensor(cache['disp1'], dtype=torch.float32)
        disp2 = torch.tensor(cache['disp2'], dtype=torch.float32)
        mask1 = torch.tensor(cache['mask1'], dtype=torch.bool)
        mask2 = torch.tensor(cache['mask2'], dtype=torch.bool)

        if self.data_aug:
            q1 = random.random()
            if q1 > 0.2:
                scale = 1 + random.random()  # [1.0, 2.0]

                transforms_CR = v2.Compose([
                    v2.CenterCrop(size=(math.floor(self.height / scale), math.floor(self.width / scale))),
                    v2.Resize(size=(self.height, self.width), antialias=True)
                ])
                data_to_transform = torch.cat((frames[0].left_img, frames[0].right_img,
                                               frames[1].left_img, frames[1].right_img,
                                               flow_l1l2, flow_l2l1, intrinsic_layer,
                                               disp1, disp2))  # (20, H, W)
                data_transformed = transforms_CR(data_to_transform)

                frames[0] = frames[0]._replace(left_img=data_transformed[0:3], right_img=data_transformed[3:6])
                frames[1] = frames[1]._replace(left_img=data_transformed[6:9], right_img=data_transformed[9:12])
                flow_l1l2 = data_transformed[12:14]
                flow_l2l1 = data_transformed[14:16]
                intrinsic_layer = data_transformed[16:18]
                disp1 = data_transformed[18:19]
                disp2 = data_transformed[19:20]

                # crop_img = frames[0].left_img * 255
                # cv_crop_img = np.array(torch.einsum('chw->hwc', crop_img), dtype=np.uint8)
                # cv2.imshow('crop_img', cv_crop_img)
                # cv2.waitKey(0)

                # scale intrinsic, do not need to change (cx, cy), because we do not need them
                K = frames[0].left_K.clone()
                K[0, 0] = K[0, 0] * scale
                K[1, 1] = K[1, 1] * scale
                frames[0] = frames[0]._replace(left_K=K)

                K = frames[1].left_K.clone()
                K[0, 0] = K[0, 0] * scale
                K[1, 1] = K[1, 1] * scale
                frames[1] = frames[1]._replace(left_K=K)

                # scale flow and disparity
                flow_l1l2 = flow_l1l2 * scale
                flow_l2l1 = flow_l2l1 * scale
                disp1 = disp1 * scale
                disp2 = disp2 * scale

                # same to binary mask
                transforms_CR_nearest = v2.Compose([
                    v2.CenterCrop(size=(math.floor(self.height / scale), math.floor(self.width / scale))),
                    v2.Resize(size=(self.height, self.width), interpolation=v2.InterpolationMode.NEAREST)
                ])
                data_to_transform_nearest = torch.cat((mask1, mask2))  # (40, H, W)
                data_transformed_mask = transforms_CR_nearest(data_to_transform_nearest)
                mask1 = data_transformed_mask[0:20]
                mask2 = data_transformed_mask[20:40]

                # crop_img = mask1[-1].unsqueeze(0) * 255
                # cv_crop_img = np.array(torch.einsum('chw->hwc', crop_img), dtype=np.uint8)
                # cv2.imshow('crop_img', cv_crop_img)
                # cv2.waitKey(0)

        final_data = Data(frame1=frames[0], frame2=frames[1],
                          transform_12=transform_12,
                          flow_l1l2=flow_l1l2, disp1=disp1, mask1=mask1,
                          flow_l2l1=flow_l2l1, disp2=disp2, mask2=mask2,
                          sequence_path=sequence_path1, intrinsic_layer=intrinsic_layer)

        if self.transform:
            p = random.random()  # (0, 1)
            if p > 0.5:
                final_data = Data(frame1=frames[1], frame2=frames[0],
                                  transform_12=transform_21,
                                  flow_l1l2=flow_l2l1, disp1=disp2, mask1=mask2,
                                  flow_l2l1=flow_l1l2, disp2=disp1, mask2=mask1,
                                  sequence_path=sequence_path2, intrinsic_layer=intrinsic_layer)

        # # TODO: 有 10% 的概率学习纯静止
        # if self.split == 'train':
        #     q2 = random.random()
        #     if q2 < 0.1:
        #         T_12 = torch.eye(4)
        #         R_12 = T_12[:3, :3]
        #         so3_12 = torch.zeros_like(transform_12.so3)
        #         euler_12 = torch.zeros_like(transform_12.euler_zyx)
        #         t_12 = torch.zeros_like(transform_12.t)
        #
        #         transform_12 = Transform(T=T_12, R=R_12, so3=so3_12, euler_zyx=euler_12, t=t_12)
        #         flow_l1l2 = torch.zeros_like(flow_l1l2)
        #         flow_l2l1 = torch.zeros_like(flow_l2l1)
        #
        #         final_data = Data(frame1=frames[0], frame2=frames[0],
        #                           transform_12=transform_12,
        #                           flow_l1l2=flow_l1l2, disp1=disp1, mask1=mask1,
        #                           flow_l2l1=flow_l2l1, disp2=disp1, mask2=mask1,
        #                           sequence_path=sequence_path1, intrinsic_layer=intrinsic_layer)

        return final_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../configs/kitti.yaml')
    parser.add_argument('--model', type=str, default='../configs/model_train.yaml')
    args = parser.parse_args()

    cfg = {
        'dataset': Config(path=args.dataset),
        'model': Config(path=args.model)
    }

    dataset = KITTI(cfg, split='predict', data_aug=False)
    sample_id = random.randint(0, len(dataset) - 1)
    for i, data in tqdm(enumerate(dataset), total=len(dataset) - 1):
        timestamp = data.frame1.timestamp

        # TODO: debug show image
        left1 = data.frame1.left_img.unsqueeze(0) * 255
        right1 = data.frame1.right_img.unsqueeze(0) * 255
        diff = torch.abs(left1 - right1)

        flow_l1l2 = flow_to_image(data.flow_l1l2.unsqueeze(0))
        flow_l1l2 = flow_l1l2[:, [2, 1, 0]]
        img_show = torch.cat([diff, flow_l1l2], dim=-2)

        img_show = img_show[0]
        disp1 = data.disp1

        # cv2_show_image_and_depth(img_show, disp1, shape='chw', bgr=False, text='[{}]: {:.6f}'.format(i, timestamp))

        # print('trans {} -> {}: [{:.9f}, {:.9f}, {:.9f}]'.format(i, i + 1, data.transform_12.t[0].item(),
        #                                                         data.transform_12.t[1].item(),
        #                                                         data.transform_12.t[2].item()))

        mask1 = torch.prod(1 - data.mask1.float(), dim=0).unsqueeze(0).cpu()  # (C, H, W), dyna: 0, non-dyna: 1
        mask2 = torch.prod(1 - data.mask2.float(), dim=0).unsqueeze(0).cpu()  # (C, H, W), dyna: 0, non-dyna: 1
        # cv2_show_depth(mask1, shape='chw')
        output_dir = os.path.join(data.sequence_path, 'mask_dynaslam')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('Build: {}'.format(output_dir))
        output_path1 = os.path.join(output_dir, str(i).zfill(6) + '.png')
        cv2_save_img(mask1 * 255, shape='chw', path=output_path1, quiet=True)
        output_path2 = os.path.join(output_dir, str(i + 1).zfill(6) + '.png')
        cv2_save_img(mask2 * 255, shape='chw', path=output_path2, quiet=True)

    print('Done.')
