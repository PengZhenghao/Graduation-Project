import os
import os.path as osp
from math import ceil

import numpy as np

# VLP16 Settings
vlp_frequency = 10  # frequency of vlp16, matching its setting on web
res_azi = vlp_frequency * 2  # resolution of azimuth angle
num_packet = int(ceil(36000 / (res_azi * 24)))

grid_size = 0.2

lidar_config = {
    "grid_size": grid_size,
    "map_size": 50,
    "min_dis": 1,
    "z_limit": 1.5,
    "num_packet": num_packet
}


def int_ceil(x):
    return int(ceil(x))


# Chart settings
x_chart = np.array([-155, 45])  # x range in chart
y_chart = np.array([-100, 100])  # x range in chart
res_chart = 1  # grid size or resolution in chart
chart = np.zeros((int_ceil((x_chart[1] - x_chart[0]) / res_chart),
                  int_ceil((y_chart[1] - y_chart[0]) / res_chart)))
chart[149:200] = 1  # initialize chart
chart[:, 0:25] = 1
chart[:, 175:200] = 1

# Global map settings
x_global = [-155, 45]  # x range in global map
y_global = [-100, 100]  # y range in global map
res_global = 1  # grid size or resolution in global map
# Parameter Calculation
# num_grid = int_ceil(map_size / grid_size)
num_out_x = int_ceil((x_global[1] - x_global[0]) / res_global)
num_out_y = int_ceil((y_global[1] - y_global[0]) / res_global)
num_trans = int_ceil(res_global / grid_size)
num_chartrans = int_ceil(res_chart / grid_size)
vlp_frequency = 10
res_vlp = vlp_frequency * 2 / 100 / 57.3

# Standard number image settings

# # TODO remove the num grid
# num_grid = 250
# image_n_std = np.zeros((num_grid, num_grid))
# for p in range(num_grid):
#     for q in range(num_grid):
#         if p != (num_grid // 2) or q != (num_grid // 2):
#             dis_pq = grid_size * sqrt((p - num_grid / 2) ** 2 + (q - num_grid / 2) ** 2)
#             n_std_pq = ceil(grid_size / (dis_pq * res_vlp))
#             image_n_std[p, q] = n_std_pq * 0.2


# TODO check if every value here are PHYSICAL values s.t. international standard, not index!
slam_config = {
    "grid_size": grid_size,
    "map_size": 50,
    "vrange": 5,

    "init_confidence": 0,
    "min_detected_confidence": 40,
    "min_removal_confidence": 0,
    "max_confidence": 100,

    "search_confidence_ratio": 0.1,  # (max_conf - current_conf) * ratio * length = search range

    "min_cluster_occupied_grids": 1,

    "max_object_length": 5,  # Not reasonable, should be larger. 5M (checked)
    "max_object_bottom_height": 1.5,

    "neighborhood_size": 2,
    "neighborhood_min_samples": 3,

    "overlap_threshold": 0.05  # should be larger, 0.1 is too small.

}



class BaseConfig(object):
    # 本机的ip地址，即在network界面看到的值。以192.168开头。
    local_ip = "127.0.0.1"
    log_level = "INFO"

    # route_ip = "192.168.1.1"

    vlp_port = "55010"
    image_port = "55011"


class VLPConfig(BaseConfig):
    def __init__(self):
        super(VLPConfig, self).__init__()

    # 原生VLP输入端口
    vlp_raw_port = 2368
    fake_run_time = True
    update_interval = 0  # in second.


class ImageConfig(BaseConfig):
    def __init__(self):
        super(ImageConfig, self).__init__()
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)

    image_refresh_interval = 1  # period to refresh in second
    if_record_rawdata = True  # record raw data from lidar
    log_dir = './test'


class RecorderConfig(BaseConfig):
    metadata = {
        "buffer_size": 5,
        "save_dir": "experiment",
        "compress": "gzip",
        "log_level": "INFO",
        "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
        "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8", "timestamp": "float64"},
        "dataset_shapes": {"lidar_data": (30600,), "extra_data": (8,), "frame": (960, 1280, 3), "timestamp": (1,)},
        "use_video_writer": True
    }
