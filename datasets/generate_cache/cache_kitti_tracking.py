import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import glob
import collections
import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
from tqdm import tqdm

from torch.utils.data import Dataset

from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from utils import *
from ultralytics import YOLO

import warnings
warnings.filterwarnings('ignore')

FramePath = collections.namedtuple(
    'FramePath', ['frame_id', 'sequence_path', 'left_img_path', 'right_img_path']
)

Frame = collections.namedtuple(
    'Frame', ['frame_id', 'left_img', 'right_img']
)

Data = collections.namedtuple(
    'Data', ['frame1', 'frame2', 'sequence_path']
)


class Dilate:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)

    def __call__(self, mask):
        """
            mask: (B, 1, H, W), 0 for mask
        """
        return -self.maxpool(-mask.float())


class KITTI_Tracking(Dataset):
    def __init__(self, cfg, split, part='testing'):
        super().__init__()
        self.data = []
        self.height = cfg['model'].network.height
        self.width = cfg['model'].network.width

        if split == 'predict':
            self.kitti_path = cfg['dataset'].prediction.path
            self.sequences = cfg['dataset'].prediction.seq
        else:
            raise NotImplementedError()

        self.kitti_left_img_path = os.path.join(self.kitti_path, 'data_tracking_image_2', part, 'image_02')
        self.kitti_right_img_path = os.path.join(self.kitti_path, 'data_tracking_image_3', part, 'image_03')

        # for each sequence
        for seq in tqdm(self.sequences, desc='Loading sequence: '):
            seq_left_img_path = os.path.join(self.kitti_left_img_path, str(int(seq)).zfill(4))
            seq_right_img_path = os.path.join(self.kitti_right_img_path, str(int(seq)).zfill(4))

            # for each frame, read image path and construct data
            frames = []
            frame_list = glob.glob(os.path.join(seq_left_img_path, '*.png'))
            frame_list.sort()
            for path in frame_list:
                frames.append(path[-10:-4])

            # make frame pairs
            for i in range(len(frames) - 1):
                left_img_path1 = os.path.join(seq_left_img_path, frames[i] + '.png')
                right_img_path1 = os.path.join(seq_right_img_path, frames[i] + '.png')
                left_img_path2 = os.path.join(seq_left_img_path, frames[i + 1] + '.png')
                right_img_path2 = os.path.join(seq_right_img_path, frames[i + 1] + '.png')

                frame_pair = {
                    'frame1': FramePath(
                    frame_id=i, sequence_path=seq_left_img_path,
                    left_img_path=left_img_path1, right_img_path=right_img_path1,
                ),
                    'frame2': FramePath(
                    frame_id=i + 1, sequence_path=seq_left_img_path,
                    left_img_path=left_img_path2, right_img_path=right_img_path2
                )}

                self.data.append(frame_pair)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        sequence_path = data['frame1'].sequence_path
        frames = []

        # each sample read a pair of images
        for key, frame_data in data.items():
            # read image
            left_img = cv2.imread(frame_data.left_img_path)    # (H, W, C), [0, 255], BGR
            right_img = cv2.imread(frame_data.right_img_path)  # (H, W, C), [0, 255], BGR

            left_img = cv2.resize(left_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            right_img = cv2.resize(right_img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            left_img = torch.tensor(left_img, dtype=torch.float32).permute(2, 0, 1) / 255  # (C, H, W), [0, 1], BGR
            right_img = torch.tensor(right_img, dtype=torch.float32).permute(2, 0, 1) / 255  # (C, H, W), [0, 1], BGR

            left_img = left_img[[2, 1, 0], ...]  # (C, H, W), [0, 1], RGB
            right_img = right_img[[2, 1, 0], ...]  # (C, H, W), [0, 1], RGB
            left_img, right_img = left_img.contiguous(), right_img.contiguous()

            # construct final data
            frames.append(
                Frame(frame_id=frame_data.frame_id, left_img=left_img, right_img=right_img)
            )

        final_data = Data(frame1=frames[0], frame2=frames[1], sequence_path=sequence_path)

        return final_data


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_method', type=str, default='raft')
    parser.add_argument('--stereo_method', type=str, default='raft')
    parser.add_argument('--seg_method', type=str, default='yolo')
    parser.add_argument('--dataset', type=str, default='configs/kitti_tracking.yaml')
    parser.add_argument('--seq', type=str, default=None)
    parser.add_argument('--model', type=str, default='configs/model_predict.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cfg = {'dataset': Config(path=args.dataset), 'model': Config(path=args.model)}

    if args.seq is not None:
        cfg['dataset'].prediction.seq = []
        cfg['dataset'].prediction.seq.append(args.seq)
        print('Generate cache for sequence {}'.format(cfg['dataset'].prediction.seq[0]))

    # ---------------------- depth ----------------------
    if args.stereo_method == 'raft':
        weights = Raft_Large_Weights.DEFAULT
        raft = raft_large(weights=weights, progress=False).to(args.device)
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError

    # ---------------------- flow ----------------------
    if args.flow_method == 'raft':
        weights = Raft_Large_Weights.DEFAULT
        raft = raft_large(weights=weights, progress=False).to(args.device)
        raft.eval()
        for param in raft.parameters():
            param.requires_grad = False

    else:
        raise NotImplementedError

    # ---------------------- segmentation ----------------------
    if args.seg_method == 'yolo':
        yolo = YOLO("models/weights/yolo11s-seg.pt")
        dilator = Dilate(kernel_size=5)

    else:
        raise NotImplementedError

    dataset = KITTI_Tracking(cfg, split='predict')

    for data in tqdm(dataset):
        timestamp = data.frame1.frame_id
        left1 = data.frame1.left_img.unsqueeze(0)
        right1 = data.frame1.right_img.unsqueeze(0)
        left2 = data.frame2.left_img.unsqueeze(0)
        right2 = data.frame2.right_img.unsqueeze(0)
        sequence_path = data.sequence_path
        frame1_id = str(data.frame1.frame_id)
        frame2_id = str(data.frame2.frame_id)

        # ---------------------- depth ----------------------
        # TODO: deep stereo matching
        if args.stereo_method == 'raft':
            left1_ = F2.normalize(left1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)
            right1_ = F2.normalize(right1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)
            flow_l1r1 = raft(left1_, right1_)[-1]
            disp1 = flow_l1r1[0, 0:1]  # (1, H, W)

            left2_ = F2.normalize(left2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)
            right2_ = F2.normalize(right2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)
            flow_l2r2 = raft(left2_, right2_)[-1]
            disp2 = flow_l2r2[0, 0:1]

            # cv2_show_image_and_depth(left1.squeeze(0), disp1.cpu().detach(), shape='chw')

        else:
            raise NotImplementedError

        disp1_np = disp1.cpu().numpy()
        disp2_np = disp2.cpu().numpy()

        # ---------------------- segmentation ----------------------
        # TODO: semantic segmentation
        if args.seg_method == 'yolo':
            # first one
            yolo_results1 = yolo.predict(
                source=left1,
                classes=[0, 1, 2, 3, 5, 7],
                agnostic_nms=True,
                verbose=False
            )
            yolo_result1 = yolo_results1[0]
            if yolo_result1.masks is not None:
                prior_masks1 = dilator(yolo_result1.masks.data).unsqueeze(0)  # (B, n, H, W)

                b, n, h, w = prior_masks1.shape
                if n < 20:
                    pad = torch.zeros((b, 20 - n, h, w), dtype=torch.float32, device=args.device)
                    prior_masks1 = torch.cat((prior_masks1, pad), dim=1)  # (B, N, H, W)
                else:
                    prior_masks1 = prior_masks1[:, :20]
            else:
                b, _, h, w = left1.shape
                prior_masks1 = torch.zeros((b, 20, h, w), dtype=torch.float32, device=args.device)

            # second one
            yolo_results2 = yolo.predict(
                source=left2,
                classes=[0, 1, 2, 3, 5, 7],
                agnostic_nms=True,
                verbose=False
            )
            yolo_result2 = yolo_results2[0]
            if yolo_result2.masks is not None:
                prior_masks2 = dilator(yolo_result2.masks.data).unsqueeze(0)  # (B, n, H, W)

                b, n, h, w = prior_masks2.shape
                if n < 20:
                    pad = torch.zeros((b, 20 - n, h, w), dtype=torch.float32, device=args.device)
                    prior_masks2 = torch.cat((prior_masks2, pad), dim=1)  # (B, N, H, W)
                else:
                    prior_masks2 = prior_masks2[:, :20]
            else:
                b, _, h, w = left2.shape
                prior_masks2 = torch.zeros((b, 20, h, w), dtype=torch.float32, device=args.device)

        else:
            raise NotImplementedError

        prior_masks1_np = prior_masks1[0].cpu().numpy().astype(bool)  # bool
        prior_masks2_np = prior_masks2[0].cpu().numpy().astype(bool)  # bool

        # final_mask = torch.sum(prior_masks1, dim=1).clamp(max=1)
        # cv2_show_image_and_depth(left1.squeeze(0), final_mask.cpu(), shape='chw')

        # ---------------------- flow ----------------------
        # TODO: deep optical flow
        if args.flow_method == 'raft':
            left1_ = F2.normalize(left1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)
            left2_ = F2.normalize(left2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous().to(args.device)

            flow_l1l2 = raft(left1_, left2_)[-1]
            flow_l2l1 = raft(left2_, left1_)[-1].cpu()

        else:
            raise NotImplementedError

        flow_l1l2_np = flow_l1l2[0].cpu().numpy()
        flow_l2l1_np = flow_l2l1[0].cpu().numpy()

        # flow_l1l2_img = flow_to_image(flow_l1l2)
        # flow_l2l1_img = flow_to_image(flow_l2l1)
        # image_and_flow = torch.cat((left1 * 255, flow_l1l2_img, flow_l2l1_img), dim=-2)
        # cv2_show_image(image_and_flow, 'bchw')

        # save cache
        path = os.path.join(sequence_path, 'cache')
        if not os.path.exists(path):
            os.mkdir(path)
            print('Make directory: {}'.format(path))

        outfile = os.path.join(path, '{}_{}.npz'.format(frame1_id, frame2_id))
        np.savez_compressed(outfile,
                            flow_l1l2=flow_l1l2_np,    # (2, H, W)
                            flow_l2l1=flow_l2l1_np,    # (2, H, W)
                            disp1=disp1_np,               # (1, H, W)
                            disp2=disp2_np,               # (1, H, W)
                            mask1=prior_masks1_np,     # (20, H, W)
                            mask2=prior_masks2_np      # (20, H, W)
                            )

    print('Done.')


if __name__ == '__main__':
    main()
