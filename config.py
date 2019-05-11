import os
import os.path as osp
from math import ceil

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

# TODO check if every value here are PHYSICAL values s.t. international standard, not index!
detector_config = {
    "grid_size": grid_size,
    "map_size": 50,
    "vrange": 5,

    "init_confidence": 0,
    "min_detected_confidence": 40,
    "min_removal_confidence": 0,
    "max_confidence": 200,

    "search_range_coefficient": 1.01,  # (max_conf - current_conf) * ratio * length = search range

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
        "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8",
                           "timestamp": "float64"},
        "dataset_shapes": {"lidar_data": (30600,), "extra_data": (8,), "frame": (960, 1280, 3),
                           "timestamp": (1,)},
        "use_video_writer": True
    }
