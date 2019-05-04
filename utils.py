import logging
import os
import time
from math import cos, sin

import h5py


def get_formatted_time(timestamp=None):
    # assert isinstance(timestamp)
    if not timestamp:
        return time.strftime('%Y-%m-%d_%H-%M-%S',
                             time.localtime())
    else:
        return time.strftime('%Y-%m-%d_%H-%M-%S',
                             time.localtime(timestamp))


def get_formatted_log_file_name():
    return time.strftime('%Y-%m-%d_%H-%M-%S_{}.log',
                         time.localtime())


def setup_logger(log_level='INFO'):
    assert isinstance(log_level, str)
    if not os.path.exists("log"):
        os.mkdir("log")
    log_format = "[%(levelname)s]\t%(asctime)s: %(message)s"
    # if config:
    #     log_level = config.log_level.upper()
    # else:
    #     log_level = "INFO"
    logging.basicConfig(filename="log/" + get_formatted_log_file_name().format("VLPmonitor"),
                        level=log_level.upper(),
                        format=log_format)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(console)


def read_data(h5_path):
    ret = {}
    with h5py.File(h5_path, 'r') as f:
        for k in f.keys():
            ret[k] = f[k][:]
    return ret


def rotmatrix(rot_m, roll, pitch, yaw):
    sin_roll = sin(roll)
    cos_roll = cos(roll)
    sin_pitch = sin(pitch)
    cos_pitch = cos(pitch)
    sin_yaw = sin(yaw)
    cos_yaw = cos(yaw)
    rot_m[0, 0] = cos_pitch * cos_yaw
    rot_m[0, 1] = sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw
    rot_m[0, 2] = cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw
    rot_m[1, 0] = cos_pitch * sin_yaw
    rot_m[1, 1] = sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw
    rot_m[1, 2] = cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw
    rot_m[2, 0] = -sin_pitch
    rot_m[2, 1] = sin_roll * cos_pitch
    rot_m[2, 2] = cos_roll * cos_pitch
    return rot_m


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# from utils import AverageMeter
import numpy as np
import cv2

ESC = 27


class Visualizer(object):
    def __init__(self, name, side_length, smooth=True, zoom=1.0, max_size=800):
        self.name = name
        self.max = AverageMeter()
        self.min = AverageMeter()
        self.mean = AverageMeter()
        self.std = AverageMeter()
        self.max_size = max_size
        self.smooth = smooth
        self.central_ratio = 1 / zoom
        self.side_length = side_length  # Side length is the length of the detecting area, in meters.
        self.image = np.empty((max_size, max_size, 3), dtype=np.uint8)
        self.len = self.image.shape[0]
        assert self.central_ratio <= 1

    def _scale(self, image):
        assert isinstance(self.image, np.ndarray)
        assert image.ndim == 2
        assert image.shape[0] == image.shape[1]
        return cv2.resize(image, (self.max_size, self.max_size))

    def _normalize_plain(self, image):
        return 255 * (image - image.min()) / (image.max() - image.min() + 1e-6)

    def _normalize_range(self, image):
        self.max.update(image.max())
        self.min.update(image.min())
        minimum = min(image.min(), self.min.avg)
        maximum = min(image.max(), self.max.avg)
        return 255 * (image - minimum) / (maximum - minimum + 1e-6)

    def _normalize_std(self, image):
        # Not used! Using Min-Max Normalization seems to be better.
        self.mean.update(image.mean())
        self.std.update(image.std())
        return np.clip((image - self.mean.avg) / self.std.avg * 128 + 128, 0, 255)

    def _add_reference(self):
        self.image[self.len // 2, :] = 255
        self.image[:, self.len // 2] = 255
        text_width = int(0.05 * self.len)
        self._put_text((self.len - 2 * text_width, self.len // 2),
                       "+{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._put_text((text_width, self.len // 2), "-{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._put_text((self.len // 2, self.len - text_width),
                       "-{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._put_text((self.len // 2, text_width), "+{:.1f}".format(self.side_length * self.central_ratio / 2))

    def _put_text(self, pos, text, color=(255, 255, 255), thickness=1, font_size=0.4):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, text, pos, font, font_size, color, thickness)

    def _zoom(self, image):
        if self.central_ratio < 1:
            x = image.shape[0] // 2
            y = image.shape[1] // 2
            keepx = int(x * self.central_ratio)
            keepy = int(y * self.central_ratio)
            return image[(x - keepx):(x + keepx), (y - keepy):(y + keepy)]
        return image

    def draw(self, image):
        # Input range: [0, 259.0], dtype=float64
        if self.smooth:
            heatmap = self._normalize_range(image)
        else:
            heatmap = self._normalize_plain(image)
        heatmap = heatmap.astype(np.uint8)
        heatmap = self._zoom(heatmap)
        heatmap = self._scale(heatmap)
        self.image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self._add_reference()
        return self.display()

    def display(self):
        cv2.imshow(self.name, self.image)
        key = cv2.waitKey(1)
        if key == ESC:
            cv2.destroyAllWindows()
            return True
        return False


from collections import deque


class FPSTimer(object):
    def __init__(self, log_interval=10):
        self.cnt = 0
        self.log_interval = log_interval
        self.queue = deque(maxlen=log_interval)
        self.queue.append(time.time())

    def __enter__(self):
        self.cnt += 1
        et = time.time()
        self.queue.append(et)
        if self.cnt < self.log_interval:
            return

        logging.info(
            "Average FPS {:.4f} (Smoothed by averaging last {} frames).".format(
                self.log_interval / (et - self.queue[0]),
                self.log_interval))
        # self.st = et

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass




        # def find_cluster():ster == 0)] = 0
        #     # if obstacles have good confidence level in local map or exist in chart, they will be marked with a very large cluster number 10000
        #     fast_cluster = np.zeros((num_grid, num_grid))
        #     fast_cluster[np.where(self.localmap >= minlevel_map)] = 10000
        #     fast_cluster[np.where(self.localchart == 1)] = 10000
        #     # DBSCAN density set to mindensity_fast,which is larger than mindensity_normal, to find cluster 10000
        #     fast_ctest = (fast_cluster > 0) * 1
        #     fast_cstate = fast_ctest * 1
        #     self.num_cluster = 0
        #     for p in range(num_grid):
        #         for q in range(num_grid):
        #             if self.image_binary[p, q] == 1 and (not (fast_ctest[p, q] == 1 and fast_cstate[p, q] == 0)):
        #                 neighbor_xmin = max(0, p - neighborsize)
        #                 neighbor_xmax = min(num_grid, p + neighborsize + 1)
        #                 neighbor_ymin = max(0, q - neighborsize)
        #                 neighbor_ymax = min(num_grid, q + neighborsize + 1)
        #                 neighbor_binary = self.image_binary[neighbor_xmin:neighbor_xmax, neighbor_ymin:neighbor_ymax]
        #                 neighbor_num = neighbor_binary.sum()
        #                 fast_ctest[p, q] = 1
        #                 if neighbor_num >= mindensity_fast:
        #                     fast_cstate[p, q] = 1
        #                     max_clusternum = (
        #                         fast_cluster[neighbor_xmin:neighbor_xmax, neighbor_ymin:neighbor_ymax] * fast_cstate[
        #                                                                                                  neighbor_xmin:neighbor_xmax,
        #                                                                                                  neighbor_ymin:neighbor_ymax]).max()  # max cluster number of fast core points in neighbor
        #                     # no fast core in  belong to existing cluster
        #                     if max_clusternum == 0:
        #                         self.num_cluster += 1
        #                         fast_cluster[p, q] = self.num_cluster
        #                         for m in range(neighbor_xmin, neighbor_xmax):
        #                             for n in range(neighbor_ymin, neighbor_ymax):
        #                                 if self.image_binary[m, n] == 1:
        #                                     fast_cluster[m, n] = self.num_cluster
        #                                     if fast_ctest[m, n] == 0:
        #                                         mn_neighbor_xmin = max(0, m - neighborsize)
        #                                         mn_neighbor_xmax = min(num_grid, m + neighborsize + 1)
        #                                         mn_neighbor_ymin = max(0, n - neighborsize)
        #                                         mn_neighbor_ymax = min(num_grid, n + neighborsize + 1)
        #                                         mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,
        #                                                              mn_neighbor_ymin:mn_neighbor_ymax]
        #                                         mn_neighbor_num = mn_neighbor_binary.sum()
        #                                         fast_ctest[m, n] = 1
        #                                         if mn_neighbor_num >= mindensity_fast:
        #                                             fast_cstate[m, n] = 1
        #                     else:
        #                         fast_cluster[p, q] = max_clusternum
        #                         for m in range(neighbor_xmin, neighbor_xmax):
        #                             for n in range(neighbor_ymin, neighbor_ymax):
        #                                 if self.image_binary[m, n] == 1:
        #                                     if fast_cluster[m, n] == 0:
        #                                         fast_cluster[m, n] = max_clusternum
        #                                         if fast_ctest[m, n] == 0:
        #                                             mn_neighbor_xmin = max(0, m - neighborsize)
        #                                             mn_neighbor_xmax = min(num_grid, m + neighborsize + 1)
        #                                             mn_neighbor_ymin = max(0, n - neighborsize)
        #                                             mn_neighbor_ymax = min(num_grid, n + neighborsize + 1)
        #                                             mn_neighbor_binary = self.image_binary[
        #                                                                  mn_neighbor_xmin:mn_neighbor_xmax,
        #                                                                  mn_neighbor_ymin:mn_neighbor_ymax]
        #                                             mn_neighbor_num = mn_neighbor_binary.sum()
        #                                             fast_ctest[m, n] = 1
        #                                             if mn_neighbor_num >= mindensity_fast:
        #                                                 fast_cstate[m, n] = 1
        #                                     elif fast_cluster[m, n] != max_clusternum:
        #                                         if fast_cstate[
        #                                             m, n] == 1:  # only core points can combine their cluster with others
        #                                             fast_cluster[
        #                                                 np.where(fast_cluster == fast_cluster[m, n])] = max_clusternum
        #     fast_cluster[np.where(fast_cluster != 10000)] = 0
        #     # find normal cluster without disturbance of cluster 10000
        #     self.image_binary[np.where(fast_cluster == 10000)] = 0
        #     # DBSCAN density set to mindensity_normal to find normal cluster
        #     normal_cluster = np.zeros((num_grid, num_grid))
        #     normal_ctest = np.zeros((num_grid, num_grid))
        #     normal_cstate = np.zeros((num_grid, num_grid))
        #     self.num_cluster = 0
        #     for p in range(num_grid):
        #         for q in range(num_grid):
        #             if self.image_binary[p, q] == 1 and (not (normal_ctest[p, q] == 1 and normal_cstate[p, q] == 0)):
        #                 neighbor_xmin = max(0, p - neighborsize)
        #                 neighbor_xmax = min(num_grid, p + neighborsize + 1)
        #                 neighbor_ymin = max(0, q - neighborsize)
        #                 neighbor_ymax = min(num_grid, q + neighborsize + 1)
        #                 neighbor_binary = self.image_binary[neighbor_xmin:neighbor_xmax, neighbor_ymin:neighbor_ymax]
        #                 neighbor_num = neighbor_binary.sum()
        #                 normal_ctest[p, q] = 1
        #                 if neighbor_num >= mindensity_normal:
        #                     normal_cstate[p, q] = 1
        #                     max_clusternum = (
        #                         normal_cluster[neighbor_xmin:neighbor_xmax, neighbor_ymin:neighbor_ymax] * normal_cstate[
        #                                                                                                    neighbor_xmin:neighbor_xmax,
        #                                                                                                    neighbor_ymin:neighbor_ymax]).max()  # max cluster number in neighbor
        #                     # no neighbor belong to existing cluster
        #                     if max_clusternum == 0:
        #                         self.num_cluster += 1
        #                         normal_cluster[p, q] = self.num_cluster
        #                         for m in range(neighbor_xmin, neighbor_xmax):
        #                             for n in range(neighbor_ymin, neighbor_ymax):
        #                                 if self.image_binary[m, n] == 1:
        #                                     normal_cluster[m, n] = self.num_cluster
        #                                     if normal_ctest[m, n] == 0:
        #                                         mn_neighbor_xmin = max(0, m - neighborsize)
        #                                         mn_neighbor_xmax = min(num_grid, m + neighborsize + 1)
        #                                         mn_neighbor_ymin = max(0, n - neighborsize)
        #                                         mn_neighbor_ymax = min(num_grid, n + neighborsize + 1)
        #                                         mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,
        #                                                              mn_neighbor_ymin:mn_neighbor_ymax]
        #                                         mn_neighbor_num = mn_neighbor_binary.sum()
        #                                         normal_ctest[m, n] = 1
        #                                         if mn_neighbor_num >= mindensity_normal:
        #                                             normal_cstate[m, n] = 1
        #                     else:
        #                         normal_cluster[p, q] = max_clusternum
        #                         for m in range(neighbor_xmin, neighbor_xmax):
        #                             for n in range(neighbor_ymin, neighbor_ymax):
        #                                 if self.image_binary[m, n] == 1:
        #                                     if normal_cluster[m, n] == 0:
        #                                         normal_cluster[m, n] = max_clusternum
        #                                         if normal_ctest[m, n] == 0:
        #                                             mn_neighbor_xmin = max(0, m - neighborsize)
        #                                             mn_neighbor_xmax = min(num_grid, m + neighborsize + 1)
        #                                             mn_neighbor_ymin = max(0, n - neighborsize)
        #                                             mn_neighbor_ymax = min(num_grid, n + neighborsize + 1)
        #                                             mn_neighbor_binary = self.image_binary[
        #                                                                  mn_neighbor_xmin:mn_neighbor_xmax,
        #                                                                  mn_neighbor_ymin:mn_neighbor_ymax]
        #                                             mn_neighbor_num = mn_neighbor_binary.sum()
        #                                             normal_ctest[m, n] = 1
        #                                             if mn_neighbor_num >= mindensity_normal:
        #                                                 normal_cstate[m, n] = 1
        #                                     elif normal_cluster[
        #                                         m, n] != max_clusternum:  # only core points can combine their cluster with others
        #                                         if normal_cstate[m, n] == 1:
        #                                             normal_cluster[
        #                                                 np.where(normal_cluster == normal_cluster[m, n])] = max_clusternum
        #     self.image_cluster = fast_cluster + normal_cluster
        #     # remove noise
        #     self.image_binary[np.where(self.image_clu
