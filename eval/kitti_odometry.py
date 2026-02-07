# Copyright (C) Huangying Zhan 2019. All rights reserved.

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from rich.panel import Panel
from rich.console import Console
from rich.columns import Columns

IS_TERMINAL_MODE = False
TERMINAL_WIDTH = 150
try:
    TERMINAL_WIDTH = os.get_terminal_size().columns
    IS_TERMINAL_MODE = False
except OSError:
    pass
GlobalConsole = Console(width=TERMINAL_WIDTH)


class Trajectory:
    def __init__(self, file_name):
        self.file_name = file_name
        self.timestamps = []
        self.traj = []  # T_wc

    def __len__(self):
        return len(self.timestamps)

    def add_traj(self, timestamp, traj):
        self.timestamps.append(timestamp)
        self.traj.append(traj)

    def set_first_identity(self):
        traj_0 = self.traj[0].copy()
        self.traj = [np.linalg.inv(traj_0) @ traj for traj in self.traj]

    def reduce_to_ids(self, ids):
        self.timestamps = [self.timestamps[idx] for idx in ids]
        self.traj = [self.traj[idx] for idx in ids]


class KittiEvalOdom:
    """Evaluate odometry result
    Usage example:
        vo_eval = KittiEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)
        Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)

        Args:
            file_name (str): txt file path
        Returns:
            poses (dict): {idx: 4x4 array}
        """
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        traj = Trajectory(file_name=file_name)
        for line in s:
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i != ""]

            try:
                assert len(line_split) == 13
            except:
                raise NotImplementedError(
                    '{} should be correct format: [timestamp, T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23]'.format(
                        file_name))

            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + 1]

            timestamp = line_split[0]
            traj.add_traj(timestamp, P)

        return traj

    def trajectory_distances(self, traj_gt):
        """
            Compute distance for each pose w.r.t frame-0
        """
        dist = [0]
        for i in range(len(traj_gt) - 1):
            P1 = traj_gt.traj[i]
            P2 = traj_gt.traj[i + 1]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))

        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return trans_error

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, traj_gt, traj_result):
        """
            calculate sequence error
        """
        err = []
        dist = self.trajectory_distances(traj_gt)
        self.step_size = 10

        for first_frame in range(0, len(traj_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(
                    dist, first_frame, len_
                )

                # Continue if sequence not long enough
                if last_frame == -1:
                    continue

                # compute rotational and translational errors
                pose_delta_gt = np.dot(
                    np.linalg.inv(traj_gt.traj[first_frame]),
                    traj_gt.traj[last_frame]
                )
                pose_delta_result = np.dot(
                    np.linalg.inv(traj_result.traj[first_frame]),
                    traj_result.traj[last_frame]
                )
                pose_error = np.dot(
                    np.linalg.inv(pose_delta_result),
                    pose_delta_gt
                )

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # compute speed
                # FIXME: 10FPS is assumed
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])

        return err

    def compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors
        Args:
            seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
                - r_err (float): rotation error
                - t_err (float): translation error
        Returns:
            ave_t_err (float): average translation error
            ave_r_err (float): average rotation error
        """
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0

    def plot_error(self, avg_segment_errs, seq, plot_error_dir):
        """Plot per-length error
        Args:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
            seq (str): sequence index.
        """
        # Translation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig, ax = plt.subplots()
        ax.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel('Translation Error (%)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = plot_error_dir + "/trans_err_{:02}.pdf".format(int(seq))
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Rotation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig, ax = plt.subplots()
        ax.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize_)
        plt.xlabel('Path Length (m)', fontsize=fontsize_)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = plot_error_dir + "/rot_err_{:02}.pdf".format(int(seq))
        plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def compute_segment_error(self, seq_errs):
        """This function calculates average errors for different segment.
        Args:
            seq_errs (list list): list of errs; [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        Returns:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}    
        """

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def matching_time_indices(self, stamps_1: list, stamps_2: list, max_diff: float = 0.01):
        stamps_1 = np.array(stamps_1)
        stamps_2 = np.array(stamps_2)

        matching_indices_1 = []
        matching_indices_2 = []
        stamps_2 = copy.deepcopy(stamps_2)
        for index_1, stamp_1 in enumerate(stamps_1):
            diffs = np.abs(stamps_2 - stamp_1)
            index_2 = int(np.argmin(diffs))
            if diffs[index_2] <= max_diff:
                matching_indices_1.append(index_1)
                matching_indices_2.append(index_2)

        return matching_indices_1, matching_indices_2

    def associate_trajectories(self, traj_1, traj_2, max_diff):
        snd_longer = len(traj_2.timestamps) > len(traj_1.timestamps)
        traj_long = copy.deepcopy(traj_2) if snd_longer else copy.deepcopy(traj_1)
        traj_short = copy.deepcopy(traj_1) if snd_longer else copy.deepcopy(traj_2)

        matching_indices_short, matching_indices_long = self.matching_time_indices(
            traj_short.timestamps, traj_long.timestamps, max_diff
        )
        assert len(matching_indices_short) == len(matching_indices_long)
        num_matches = len(matching_indices_long)
        traj_short.reduce_to_ids(matching_indices_short)
        traj_long.reduce_to_ids(matching_indices_long)

        traj_1 = traj_short if snd_longer else traj_long
        traj_2 = traj_long if snd_longer else traj_short

        if num_matches == 0:
            raise Exception(
                "found no matching timestamps between {} and {} with max. time diff {} (s) ".format(traj_1.file_name,
                                                                                                    traj_2.file_name,
                                                                                                    max_diff))

        return traj_1, traj_2

    def scale_lse_solver(self, X, Y):
        """Least-sqaure-error solver
        Compute optimal scaling factor so that s(X)-Y is minimum
        Args:
            X (KxN array): current data
            Y (KxN array): reference data
        Returns:
            scale (float): scaling factor
        """
        scale = np.sum(X * Y) / np.sum(X ** 2)
        return scale

    def scale_optimization(self, gt, pred):
        """ Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in range(len(pred)):
            pose_pred = pred.traj[i]
            pose_ref = gt.traj[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = self.scale_lse_solver(xyz_pred, xyz_ref)
        for i in range(len(pred_updated)):
            pred_updated.traj[i][:3, 3] *= scale
        return pred_updated

    def umeyama_alignment(self, x, y, with_scale=False):
        """
        Computes the least squares solution parameters of an Sim(m) matrix
        that minimizes the distance between a set of registered points.
        Umeyama, Shinji: Least-squares estimation of transformation parameters
                         between two point patterns. IEEE PAMI, 1991
        :param x: mxn matrix of points, m = dimension, n = nr. of data points
        :param y: mxn matrix of points, m = dimension, n = nr. of data points
        :param with_scale: set to True to align also the scale (default: 1.0 scale)
        :return: r, t, c - rotation matrix, translation vector and scale factor
        """
        if x.shape != y.shape:
            assert False, "x.shape not equal to y.shape"

        # m = dimension, n = nr. of data points
        m, n = x.shape

        # means, eq. 34 and 35
        mean_x = x.mean(axis=1)
        mean_y = y.mean(axis=1)

        # variance, eq. 36
        # "transpose" for column subtraction
        sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

        # covariance matrix, eq. 38
        outer_sum = np.zeros((m, m))
        for i in range(n):
            outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
        cov_xy = np.multiply(1.0 / n, outer_sum)

        # SVD (text betw. eq. 38 and 39)
        u, d, v = np.linalg.svd(cov_xy)

        # S matrix, eq. 43
        s = np.eye(m)
        if np.linalg.det(u) * np.linalg.det(v) < 0.0:
            # Ensure a RHS coordinate system (Kabsch algorithm).
            s[m - 1, m - 1] = -1

        # rotation, eq. 40
        r = u.dot(s).dot(v)

        # scale & translation, eq. 42 and 41
        c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
        t = mean_y - np.multiply(c, r.dot(mean_x))

        return r, t, c

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []
        idx_0 = 0
        gt_0 = gt.traj[idx_0]
        pred_0 = pred.traj[idx_0]

        for i in range(len(pred)):
            # cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_gt = gt.traj[i]
            gt_xyz = cur_gt[:3, 3]

            # cur_pred = np.linalg.inv(pred_0) @ pred[i]
            cur_pred = pred.traj[i]
            pred_xyz = cur_pred[:3, 3]

            align_err = gt_xyz - pred_xyz

            # print('i: ', i)
            # print("gt: ", gt_xyz)
            # print("pred: ", pred_xyz)
            # input("debug")
            errors.append(np.sqrt(np.sum(align_err ** 2)))

        # FIXME
        # ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
        ate = np.mean(np.asarray(errors))

        return ate

    def compute_RPE(self, gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in range(len(pred) - 1):
            gt1 = gt.traj[i]
            gt2 = gt.traj[i + 1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred.traj[i]
            pred2 = pred.traj[i + 1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

    def eval(self, gt_path, result_path, result_dir=None, seq=None, alignment=None, max_diff=0.01, quiet=False):
        """Evaulate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            seq: sequence name
            max_diff (float): max. allowed absolute time difference
            quiet (bool): whether concise output
        """
        # Read pose txt
        traj_gt = self.load_poses_from_txt(gt_path)
        traj_result = self.load_poses_from_txt(result_path)

        # Pose alignment to first frame
        traj_gt.set_first_identity()
        traj_result.set_first_identity()

        # Associate timestamps
        traj_gt, traj_result = self.associate_trajectories(traj_gt, traj_result, max_diff)
        assert len(traj_gt) == len(traj_result)

        if alignment == "scale":
            traj_result = self.scale_optimization(traj_gt, traj_result)
        elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
            # get XYZ
            xyz_gt = []
            xyz_result = []
            for cnt in range(len(traj_result)):
                xyz_gt.append([traj_gt.traj[cnt][0, 3], traj_gt.traj[cnt][1, 3], traj_gt.traj[cnt][2, 3]])
                xyz_result.append([traj_result.traj[cnt][0, 3], traj_result.traj[cnt][1, 3], traj_result.traj[cnt][2, 3]])
            xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
            xyz_result = np.asarray(xyz_result).transpose(1, 0)

            r, t, scale = self.umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")

            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t

            for cnt in range(len(traj_result)):
                traj_result.traj[cnt][:3, 3] *= scale
                if alignment == "7dof" or alignment == "6dof":
                    traj_result.traj[cnt] = align_transformation @ traj_result.traj[cnt]

        # compute sequence errors
        seq_err = self.calc_sequence_errors(traj_gt, traj_result)

        # compute overall error
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)

        # Compute ATE
        ate = self.compute_ATE(traj_gt, traj_result)

        # Compute RPE
        rpe_trans, rpe_rot = self.compute_RPE(traj_gt, traj_result)

        if not quiet:
            print("Sequence: {}".format(seq))
            print("File: {}".format(traj_result.file_name))
            print("Translational error (%): ", ave_t_err * 100)
            print("Rotational error (deg/100m): ", ave_r_err / np.pi * 180 * 100)
            print("ATE (m): ", ate)
            print("RPE (m): ", rpe_trans)
            print("RPE (deg): ", rpe_rot * 180 / np.pi)

        # Plotting
        if result_dir is not None:
            assert seq is not None
            plot_error_dir = result_dir + "/plot_error"
            if not os.path.exists(plot_error_dir):
                os.makedirs(plot_error_dir)

            avg_segment_errs = self.compute_segment_error(seq_err)
            self.plot_error(avg_segment_errs, seq, plot_error_dir)

        t_rel = ave_t_err * 100
        r_rel = ave_r_err / np.pi * 180 * 100
        rpe_rot = rpe_rot * 180 / np.pi

        if not quiet:
            print("-------------------- For Copying ------------------------------")
            print("{0:.2f}".format(t_rel))
            print("{0:.2f}".format(r_rel))
        else:
            box1 = Panel.fit(
                "\n".join(
                    [
                        "t_rel (%):           {0:.2f}".format(t_rel),
                        "r_rel (deg/100m):    {0:.2f}".format(r_rel),
                    ]
                ),
                title="Drift Metrics",
                title_align="center",
            )
            box2 = Panel.fit(
                "\n".join(
                    [
                        "ATE (m):            {0:.2f}".format(ate),
                    ]
                ),
                title="ATE",
                title_align="center",
            )
            box3 = Panel.fit(
                "\n".join(
                    [
                        "RPE (m):            {0:.4f}".format(rpe_trans),
                        "RPE (deg):          {0:.4f}".format(rpe_rot),
                    ]
                ),
                title="RPE",
                title_align="center",
            )
            GlobalConsole.print(Columns([box1, box2, box3]))

        return t_rel, r_rel, ate, rpe_trans, rpe_rot
