import logging
import os
import time
from math import cos, sin

import cv2
import h5py
import numpy as np

ESC = 27


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


def np_logical_and_list(*lis):
    if len(lis) < 2:
        return lis
    mask = lis[0]
    for exp in lis[1:]:
        mask = np.logical_and(mask, exp)
    return mask


def setup_logger(log_level='INFO'):
    assert isinstance(log_level, str)
    if not os.path.exists("log"):
        os.mkdir("log")
    log_format = "[%(levelname)s]\t%(asctime)s: %(message)s"
    # if lidar_config:
    #     log_level = lidar_config.log_level.upper()
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


def lazy_read_data(h5_path):
    ret = {}
    f = h5py.File(h5_path, 'r')
    for k in f.keys():
        ret[k] = f[k]
    return ret


def build_index_to_location(config):
    def index_to_location(d):
        return d * config["grid_size"] - config["map_size"] / 2

    return index_to_location


def build_location_to_index(config):
    def location_to_index(x):
        return (np.floor((x[:, 0] + config["map_size"] / 2) / config["grid_size"]).astype(np.int),
                np.floor((x[:, 1] + config["map_size"] / 2) / config["grid_size"]).astype(np.int))

    return location_to_index


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


from collections import deque


class MovingAverage(object):
    def __init__(self, max_len=10):
        self.val = deque(maxlen=max_len)
        self.avg = None
        self.maxlen = max_len

    def update(self, val):
        self.val.append(val)
        self.avg = sum(self.val) / self.maxlen


color_white = (255, 255, 255)
color_yellow = (65, 254, 254)
color_green = (0, 255, 0)


class Visualizer(object):
    def __init__(self, name, side_length, smooth=True, zoom=1.0, max_size=800):
        self.name = name
        self.max = AverageMeter()
        self.min = AverageMeter()
        self.mean = AverageMeter()
        self.std = AverageMeter()
        self.max_size = max_size
        self.scale = 1.0
        self.smooth = smooth
        self.central_ratio = 1 / zoom
        self.side_length = side_length  # Side length is the length of the detecting area, in meters.
        self.image = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        self.len = self.image.shape[0]
        assert self.central_ratio <= 1

    def _scale(self, image):
        # assert isinstance(self.image, np.ndarray)
        # assert image.ndim == 2
        # assert image.shape[0] == image.shape[1]
        assert isinstance(image, np.ndarray)

        return cv2.resize(image.T[::-1, :], (self.max_size, self.max_size))

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
        self._draw_text((self.len - 2 * text_width, self.len // 2),
                        "+{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._draw_text((text_width, self.len // 2),
                        "-{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._draw_text((self.len // 2, self.len - text_width),
                        "-{:.1f}".format(self.side_length * self.central_ratio / 2))
        self._draw_text((self.len // 2, text_width),
                        "+{:.1f}".format(self.side_length * self.central_ratio / 2))

    def _draw_text(self, pos, text, color=color_white, thickness=1, font_size=0.4):
        pos = tuple(pos)
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

    def _convert(self, pos):
        return int(pos[0] * self.scale), int(
            self.max_size - pos[1] * self.scale / self.central_ratio)

    def _draw_bounding_box(self, bounding_box, color, info=None):
        x_min, y_min, x_max, y_max = bounding_box

        x_min, y_min = self._convert((x_min, y_min))
        x_max, y_max = self._convert((x_max, y_max))
        points = ((x_min, y_min),
                  (x_min, y_max),
                  (x_max, y_max),
                  (x_max, y_min),
                  (x_min, y_min))

        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(self.image, p1, p2, color)

        if info:
            if y_max > 0.95 * self.max_size:
                increment = -15
                ref = y_max
            else:
                increment = +15
                ref = y_min
            pos = [x_min, min(max(ref + increment, 0), self.max_size)]
            text = "{}: {}".format(info["name"], info["status"])

            self._draw_text(pos, text, color)
            pos[1] = min(max(pos[1] + increment, 0), self.max_size)

            text = "({:.2f},{:.2f})".format(*info["centroid"])
            self._draw_text(pos, text, color)
            pos[1] = min(max(pos[1] + increment, 0), self.max_size)

            for k, v in info.items():
                if isinstance(v, float):
                    text = "{:8.8}:{:.2f}".format(k, v)
                elif isinstance(v, int):
                    text = "{:8.8}:{}".format(k, v)
                else:
                    continue
                self._draw_text(pos, text, color)
                pos[1] = min(max(pos[1] + increment, 0), self.max_size)

    def _draw_objects(self, objects, target=None, color=color_white):
        for label, info in objects.items():
            if target and label == target:
                c = color_green
            else:
                c = color
            self._draw_bounding_box(info["bounding_box"], c, info=info)
            if "search_range" in info:
                self._draw_bounding_box(info["search_range"], color_yellow)

    def draw(self, image, objects=None, target=None):
        # We ask the coordination of all input are in the standard form.
        self.scale = self.max_size / image.shape[0]
        if self.smooth:
            heatmap = self._normalize_range(image)
        else:
            heatmap = self._normalize_plain(image)
        heatmap = heatmap.astype(np.uint8)
        heatmap = self._zoom(heatmap)
        heatmap = self._scale(heatmap)
        self.image = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self._add_reference()
        if objects:
            self._draw_objects(objects, target)
        if isinstance(target, list):
            self._draw_text(self._convert((10, 10)),
                            "Press Num Key to Specify Target: {}".format(target), font_size=0.8)
        else:
            self._draw_text(self._convert((10, 10)),
                            "Current Target: {} (Press Q to quit)".format(target), font_size=0.8)
        return self.display()

    def display(self):
        cv2.imshow(self.name, self.image)
        key = cv2.waitKey(1)
        if key == ESC:
            cv2.destroyAllWindows()
        return key


class Reader(object):  # This class should merged with Recorder. But I don't give a shit for this.
    def __init__(self, path):
        assert os.path.isfile(path)
        assert path.endswith(".h5")
        self.file = h5py.File(path, 'r')
        self.p = 0  # pointer
        self.len = len(self.file["lidar_data"])

    def update(self):  # match the API of the live stream VLP object.
        if self.p < self.len:
            ret = self.file["lidar_data"][self.p], self.file["extra_data"][self.p]
            self.p += 1
            return ret
        raise EOFError

    def close(self):
        self.file.close()


def assert_euqal(a1, a2):
    max_diff = np.max(np.abs(a1 - a2))
    logging.debug("Here is a assertion! Please delete it! The differences: {}".format(max_diff))
    assert max_diff < 1e-6


class FPSTimer(object):
    def __init__(self, force_fps=None, log_interval=10):
        self.cnt = 0
        self.log_interval = log_interval
        self.queue = deque(maxlen=log_interval)
        self.t = time.time()
        self.queue.append(self.t)
        self.force_fps = force_fps

    def __enter__(self):
        self.cnt += 1
        self.t = time.time()
        self.queue.append(self.t)
        if self.cnt < self.log_interval:
            return

        logging.info(
            "Average FPS {:.4f} (Smoothed by averaging last {} frames, total passed {} frames).".format(
                self.log_interval / (self.t - self.queue[0]),
                self.log_interval, self.cnt))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.force_fps:
            now = time.time()
            if 1 / (now - self.t) > self.force_fps:
                sleep = 1 / self.force_fps + self.t - now
                time.sleep(sleep)
