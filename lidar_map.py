import logging
from math import ceil, pi, cos, sin

import numpy as np

from utils import np_logical_and_list

# VLP16 Settings
vlp_frequency = 10  # frequency of vlp16, matching its setting on web
res_azi = vlp_frequency * 2  # resolution of azimuth angle
num_packet = int(ceil(36000 / (res_azi * 24)))

lidar_config = {
    "grid_size": 0.2,
    "map_size": 50,
    "min_dis": 1,
    "z_limit": 1.5,
    "num_packet": num_packet
}

from utils import build_index_to_location

class LidarMap(object):
    def __init__(self, config, offset=None):
        # self.rot_m = np.zeros((3, 3))  # rotation matrix
        if offset is None:
            import pickle
            offset = pickle.load(open("offset.pkl", "rb"))
            assert isinstance(offset, dict)
            logging.warning("We have load offset from file! The content is: {}".format(offset))
            offset = offset["offset"]
        self.runtime = 0
        self.posx = 0
        self.posy = 0
        self.config = config
        print("Offset set to:", offset)
        self.offset = offset
        self.grid_size = config["grid_size"]
        self.map_size = config["map_size"]
        self.min_dis = config["min_dis"]
        self.z_limit = config["z_limit"]
        self.num_packet = config["num_packet"]
        self.num_grid = int(ceil(self.map_size / self.grid_size))
        num_grid = self.num_grid
        self.map_high = np.ones([num_grid, num_grid], dtype=np.float32) * (-self.z_limit)
        self.map_low = np.ones([num_grid, num_grid], dtype=np.float32) * self.z_limit
        self.map_height = np.zeros([num_grid, num_grid], dtype=np.float32)
        self.map_n = np.zeros([num_grid, num_grid], dtype=np.int16)

        self.matrix_indices = np.unravel_index(np.arange(num_grid * num_grid),
                                               dims=self.map_n.shape)
        self.index_to_location = build_index_to_location(self.config)

        angs = np.empty((8, 2))
        for p in range(8):
            for q in range(2):
                angs[p, q] = (q * 16 + p * 2 - 15) / 180 * pi
                # angs[p, q] = (q * 16 + p * 2 - 15) / 180 * pi
        angs = angs.flatten()
        self.cos_angs = np.cos(angs)
        self.sin_angs = np.sin(angs)

    def refresh(self):
        self.map_high.fill(-self.z_limit)
        self.map_low.fill(self.z_limit)
        self.map_height.fill(0)
        self.map_n.fill(0)

    def parse(self, data):  # data = np.array[30600]

        distance, azimuth = np.split(data.reshape(self.num_packet, 408), (384,), axis=1)

        distance = distance.reshape(-1, 12, 2, 16) * 0.002
        azimuth = azimuth.reshape(-1, 12, 2, 1) * pi / 18000

        new_x = -distance * self.cos_angs * np.cos(azimuth + self.offset).repeat(16, axis=3)
        new_y = distance * self.cos_angs * np.sin(azimuth + self.offset).repeat(16, axis=3)
        new_z = (distance * self.sin_angs).flatten()  # 75 12 2 16

        new_points = np.stack([new_x.flatten(), new_y.flatten()], axis=1)
        new_position = np.floor((new_points + self.map_size / 2) / self.grid_size).astype(np.int)

        mask = np_logical_and_list(0 <= new_position[:, 0], new_position[:, 0] < self.num_grid,
                                   0 <= new_position[:, 1], new_position[:, 1] < self.num_grid,
                                   distance.flatten() > self.min_dis,
                                   -self.z_limit < new_z, new_z < self.z_limit)

        new_z = new_z[mask]
        new_position = new_position[mask]
        if new_position.size == 0:
            return {k: None for k in ("weight", "high", "low", "points", "indices")}

        high_indices = new_z.argsort()
        high_position = new_position[high_indices]
        high_pos_x, high_pos_y = high_position[:, 0], high_position[:, 1]
        self.map_high[high_pos_x, high_pos_y] = \
            np.maximum(self.map_high[high_pos_x, high_pos_y], new_z[high_indices])

        low_indices = high_indices[::-1]
        low_position = new_position[low_indices]
        low_pos_x, low_pos_y = low_position[:, 0], low_position[:, 1]
        self.map_low[low_pos_x, low_pos_y] = \
            np.minimum(self.map_low[low_pos_x, low_pos_y], new_z[low_indices])

        n_counts = np.bincount(self.map_n.shape[0] * new_position[:, 0] + new_position[:, 1])
        n_indices = self.matrix_indices[0][:len(n_counts)], self.matrix_indices[1][:len(n_counts)]
        self.map_n[n_indices] = n_counts

        non_zero_mask = n_counts != 0
        n_counts = n_counts[non_zero_mask]

        indices = np.stack(n_indices, axis=1)[non_zero_mask]  # (num_samples, 2) in int
        points = self.index_to_location(indices)  # (num_samples, 2) in float (physical distance)

        return {"weight": n_counts,
                "points": points,
                "indices": indices,
                "high": self.map_high[n_indices][non_zero_mask],
                "low": self.map_low[n_indices][non_zero_mask]}

    def gps_tranform(self, points, offset):

        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 2

        ret = []
        ret.append(points[:, 0] * cos(offset) - points[:, 1] * sin(offset))
        ret.append(points[:, 0] * sin(offset) + points[:, 1] * cos(offset))
        return np.stack(ret, axis=1)



    def update(self, lidar_data, extra_data):
        self.refresh()
        posx, posy = extra_data[:2]
        ret = self.parse(lidar_data)

        return {
            "posx": posx,
            "posy": posy,
            "runtime": self.runtime,
            "map_high": self.map_high,
            "map_low": self.map_low,
            "map_n": self.map_n,
            **ret
        }


if __name__ == "__main__":
    from utils import read_data, setup_logger, FPSTimer, Visualizer
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", '-r', action="store_true")
    parser.add_argument("--path", '-p', required=True, help="The path of target h5 file.")
    parser.add_argument("--save", '-s', action="store_true")
    args = parser.parse_args()

    assert args.path.endswith(".h5")
    args.save = args.save and (not os.path.exists(args.path.replace(".h5", ".pkl")))
    setup_logger()

    data = read_data(args.path)
    lidar_data = data['lidar_data']
    extra_data = data['extra_data']

    map = LidarMap(lidar_config)
    v = Visualizer("map_n", lidar_config["map_size"], zoom=5)
    fpstimer = FPSTimer()

    save_data = []
    for l, e in zip(lidar_data, extra_data):
        with fpstimer:
            ret = map.update(l, e)
            if args.save:
                save_data.append(ret)
            if args.render:
                stop = v.draw(ret["map_n"])
                if stop:
                    break

    if args.save:
        import pickle

        try:
            pickle.dump(save_data, open(args.path.replace(".h5", ".pkl"), "wb"))
        except Exception as e:
            os.remove(args.path.replace(".h5", ".pkl"))
            raise e
