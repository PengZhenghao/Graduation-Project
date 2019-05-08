from math import ceil, pi

import numpy as np

from utils import np_logical_and_list

# VLP16 Settings
vlp_frequency = 10  # frequency of vlp16, matching its setting on web
res_azi = vlp_frequency * 2  # resolution of azimuth angle
num_packet = int(ceil(36000 / (res_azi * 24)))

config = {
    "grid_size": 0.2,
    "map_size": 50,
    "min_dis": 1,
    "z_limit": 1.5,
    "num_packet": num_packet
}


class LidarMap(object):
    def __init__(self, config):
        # self.rot_m = np.zeros((3, 3))  # rotation matrix
        self.runtime = 0
        self.posx = 0
        self.posy = 0
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

        angs = np.empty((8, 2))
        for p in range(8):
            for q in range(2):
                angs[p, q] = (q * 16 + p * 2 - 15) / 180 * pi
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
        azimuth = azimuth.reshape(-1, 12, 2, 1) * np.pi / 18000

        new_x = distance * self.cos_angs * np.cos(azimuth).repeat(16, axis=3)
        new_y = distance * self.cos_angs * np.sin(azimuth).repeat(16, axis=3)
        new_z = - distance * self.sin_angs  # 75 12 2 16

        new_x = new_x.flatten()
        new_y = new_y.flatten()
        new_z = new_z.flatten()

        raw_points = np.stack((new_x, new_y, new_z), axis=1)

        pos_x = np.floor((new_x + self.map_size / 2) / self.grid_size).astype(np.int)
        pos_y = np.floor((new_y + self.map_size / 2) / self.grid_size).astype(np.int)

        mask = np_logical_and_list(0 <= pos_x, pos_x < self.num_grid,
                                   0 <= pos_y, pos_y < self.num_grid,
                                   distance.flatten() > self.min_dis,
                                   -self.z_limit < new_z, new_z < self.z_limit)

        new_z = new_z[mask]
        pos_x = pos_x[mask]
        pos_y = pos_y[mask]

        high_indices = new_z.argsort()
        high_pos_x = pos_x[high_indices]
        high_pos_y = pos_y[high_indices]

        self.map_high[high_pos_x, high_pos_y] = \
            np.maximum(self.map_high[high_pos_x, high_pos_y], new_z[high_indices])

        low_indices = high_indices[::-1]
        low_pos_x = pos_x[low_indices]
        low_pos_y = pos_y[low_indices]

        self.map_low[low_pos_x, low_pos_y] = \
            np.minimum(self.map_low[low_pos_x, low_pos_y], new_z[low_indices])

        n_counts = np.bincount(self.map_n.shape[0] * pos_x + pos_y)
        n_indices = np.unravel_index(np.arange(n_counts.size), dims=self.map_n.shape)
        self.map_n[n_indices] = n_counts

        return raw_points

    def update(self, lidar_data, extra_data):
        self.refresh()
        posx, posy, roll, pitch, yaw, posx102, posy102 \
            = extra_data[:7]
        points = self.parse(lidar_data)
        print(points.mean(axis=0))

        return {
            "posx": posx,
            "posy": posy,
            "runtime": self.runtime,
            "map_high": self.map_high,
            "map_low": self.map_low,
            "map_n": self.map_n,
            "points": points
        }


if __name__ == "__main__":  # This part is not allow to changed! Used for profiling!
    from utils import Visualizer
    from utils import read_data, setup_logger, FPSTimer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", '-r', action="store_true")
    parser.add_argument("--path", '-p', required=True, help="The path of target h5 file.")
    args = parser.parse_args()

    setup_logger()

    data = read_data(args.path)
    lidar_data = data['lidar_data']
    extra_data = data['extra_data']

    map = (config)
    v = Visualizer("map_n", config["map_size"], zoom=5)

    fpstimer = FPSTimer()

    import matplotlib.pyplot as plt

    plt.ion()

    for l, e in zip(lidar_data, extra_data):
        with fpstimer:
            ret = map.update(l, e)
            if args.render:
                stop = v.draw(ret["map_n"])
                # points = ret["points"]

                # plt.clf()
                #
                # plt.xlim(-50, 50)
                # plt.ylim(-50, 50)
                # plt.scatter(points[:,0], points[:,1], s=1)
                # plt.show()
                if stop:
                    break
