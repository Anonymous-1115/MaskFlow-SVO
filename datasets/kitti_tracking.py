import collections
import glob
import os
import warnings
import argparse
from datetime import datetime

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


class KITTI_Tracking(Dataset):
    def __init__(self, cfg, split, transform=False, part='training'):
        super().__init__()
        self.kitti_data = []
        self.transform = transform
        self.height = cfg['model'].network.height
        self.width = cfg['model'].network.width

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

        self.kitti_left_img_path = os.path.join(self.kitti_path, 'data_tracking_image_2', part, 'image_02')
        self.kitti_right_img_path = os.path.join(self.kitti_path, 'data_tracking_image_3', part, 'image_03')
        self.kitti_calib_path = os.path.join(self.kitti_path, 'data_tracking_calib', part, 'calib')
        self.kitti_pose_path = os.path.join(self.kitti_path, 'data_tracking_poses', part, 'poses')
        self.kitti_cache_path = os.path.join(self.kitti_path, 'data_tracking_cache', part, 'cache')

        # for each sequence
        for seq in tqdm(self.kitti_sequences, desc='Loading sequence: '):
            seq_left_img_path = os.path.join(self.kitti_left_img_path, str(int(seq)).zfill(4))
            seq_right_img_path = os.path.join(self.kitti_right_img_path, str(int(seq)).zfill(4))
            seq_cache_path = os.path.join(self.kitti_cache_path, str(int(seq)).zfill(4))

            seq_timestamp_path = os.path.join(seq_left_img_path, 'timestamps.txt')
            seq_calib_path = os.path.join(self.kitti_calib_path, str(int(seq)).zfill(4) + '.txt')
            seq_pose_path = os.path.join(self.kitti_pose_path, str(int(seq)).zfill(4) + '.txt')

            # read calib file (4 cameras, we only use cam2 and cam3)
            K, T_bc = [], []
            R_rect, Tr_velo_to_cam, Tr_imu_to_velo = np.eye(4), np.eye(4), np.eye(4)
            with open(seq_calib_path, 'r') as f:
                for line in f.readlines():
                    try:
                        key, value = line.split(':', 1)
                    except:
                        key, value = line.split(' ', 1)
                    value = value.strip()

                    if key in ['P0', 'P1', 'P2', 'P3']:
                        P = np.reshape(np.array([float(x) for x in value.split()]), (3, 4))
                        K.append(P[0:3, 0:3])

                        # Note: the projection matrix of KITTI brings a point X from the
                        # left rectified camera coordinate system to a point x in the i'th image plane
                        T_cb = np.eye(4)
                        T_cb[0, 3] = P[0, 3] / P[0, 0]
                        T_bc.append(np.linalg.inv(T_cb))
                    elif key == 'R_rect':
                        R_rect[:3, :3] = np.reshape(np.array([float(x) for x in value.split()]), (3, 3))
                    elif key == 'Tr_velo_cam':
                        Tr_velo_to_cam[:3, :4] = np.reshape(np.array([float(x) for x in value.split()]), (3, 4))
                    elif key == 'Tr_imu_velo':
                        Tr_imu_to_velo[:3, :4] = np.reshape(np.array([float(x) for x in value.split()]), (3, 4))
                    else:
                        raise NotImplementedError('Unrecognized key: {}'.format(key))

            baseline = T_bc[3][0, 3] - T_bc[2][0, 3]
            T_imu_to_cam2 = np.linalg.inv(T_bc[2]) @ R_rect @ Tr_velo_to_cam @ Tr_imu_to_velo

            # read pose, 'b' means cam0
            poses = []
            try:
                with open(seq_pose_path, 'r') as f:
                    for line in f.readlines():
                        # read IMU pose
                        T_wb = np.fromstring(line, dtype=float, sep=' ')
                        T_wb = T_wb.reshape(3, 4)
                        T_wb = np.vstack((T_wb, [0, 0, 0, 1]))

                        # from IMU pose to Camera pose
                        T_wb = T_imu_to_cam2 @ T_wb @ np.linalg.inv(T_imu_to_cam2)
                        poses.append(T_wb)
            except FileNotFoundError:
                print('Ground truth poses are not available for sequence ' + seq + '.')

            # for each frame, read image path and construct data
            frames = []
            frame_list = glob.glob(os.path.join(seq_left_img_path, '*.png'))
            frame_list.sort()
            for path in frame_list:
                frames.append(path[-10:-4])

            # read timestamp if possible
            timestamps = []
            try:
                with open(seq_timestamp_path, 'r') as f:
                    for line in f.readlines():
                        t = datetime.fromisoformat(line.strip()[:-3])
                        if len(timestamps) == 0:
                            t0 = t
                            timestamps.append(0.0)
                        else:
                            delta_t = (t - t0).seconds + (t - t0).microseconds * 1e-6
                            timestamps.append(delta_t)
            except:
                for i in range(len(frames)):
                    timestamps.append(i)

            # make frame pairs
            left_distortion = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            right_distortion = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
            for i in range(len(frames) - 1):
                left_img_path1 = os.path.join(seq_left_img_path, frames[i] + '.png')
                right_img_path1 = os.path.join(seq_right_img_path, frames[i] + '.png')
                left_img_path2 = os.path.join(seq_left_img_path, frames[i + 1] + '.png')
                right_img_path2 = os.path.join(seq_right_img_path, frames[i + 1] + '.png')
                cache_path = os.path.join(seq_cache_path, str(frames[i]) + '_' + str(frames[i + 1]) + '.npz')

                frame_pair = {
                    'frame1': FramePath(
                        frame_id=frames[i], timestamp=timestamps[i],
                        sequence_path=seq_left_img_path,
                        left_img_path=left_img_path1, right_img_path=right_img_path1,
                        baseline=baseline, left_K=K[2], right_K=K[3],
                        left_distortion=left_distortion, right_distortion=right_distortion,
                        T_wb=poses[i]
                    ),
                    'frame2': FramePath(
                        frame_id=frames[i + 1], timestamp=timestamps[i + 1],
                        sequence_path=seq_left_img_path,
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
            left_img = torch.tensor(left_img, dtype=torch.float32).permute(2, 0,
                                                                           1) / 255  # (C, H, W), [0, 1]， BGR

            # update height and width
            height = left_img.shape[-2]
            width = left_img.shape[-1]

            right_img = cv2.imread(frame_data.right_img_path)
            right_img = cv2.resize(right_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            right_img = torch.tensor(right_img, dtype=torch.float32).permute(2, 0,
                                                                             1) / 255  # (C, H, W), [0, 1]， BGR

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

        return final_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../configs/kitti_tracking.yaml')
    parser.add_argument('--model', type=str, default='../configs/model_predict.yaml')
    args = parser.parse_args()

    cfg = {
        'dataset': Config(path=args.dataset),
        'model': Config(path=args.model)
    }

    dataset = KITTI_Tracking(cfg, split='predict')
    sample_id = random.randint(0, len(dataset) - 1)
    for data in tqdm(dataset, total=len(dataset)-1):
        timestamp = "Timestamp: {:.2f}".format(data.frame1.timestamp)

        # TODO: debug show image
        left1 = data.frame1.left_img.unsqueeze(0)
        left1 = left1[0].permute(1, 2, 0) * 255
        cv2.imshow('img', np.array(left1, dtype=np.uint8))
        cv2.setWindowTitle('img', timestamp)
        cv2.waitKey(0)

    print('Done.')
