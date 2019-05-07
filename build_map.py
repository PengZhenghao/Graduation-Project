from math import ceil, pi

import numpy as np

from utils import rotmatrix

# VLP16 Settings
vlp_frequency = 10  # frequency of vlp16, matching its setting on web

# Image Settings
image_size = 50  # side length of a square image
# image_size = 5  # side length of a square image
grid_size = 0.2  # grid size in image
# grid_size = 0.02  # grid size in image
min_dis = 1  # ignore reflection nearer than the minimum distance
vrange = 1.5  # vertical value range from -vrange to vrange

# Parameter Calculation
res_azi = vlp_frequency * 2  # resolution of azimuth angle
num_packet = int(ceil(36000 / (res_azi * 24)))  # number of packets in one image
num_grid = int(ceil(image_size / grid_size))

angs = np.empty((8, 2))
for p in range(8):
    for q in range(2):
        angs[p, q] = (q * 16 + p * 2 - 15) / 180 * pi
cos_angs = np.cos(angs)
sin_angs = np.sin(angs)

angs = angs.flatten()  # angs = [16,]
new_cos_angs = np.cos(angs)
new_sin_angs = np.sin(angs)  # (16,)


def np_logical_and_list(*lis):
    if len(lis) < 2:
        return lis
    mask = lis[0]
    for exp in lis[1:]:
        mask = np.logical_and(mask, exp)
    return mask


class Image:
    def __init__(self):
        self.rot_m = np.zeros((3, 3))  # rotation matrix
        self.runtime = 0
        self.posx = 0
        self.posy = 0
        self.image_high = np.ones([num_grid, num_grid], dtype=np.float32) * (-vrange)
        self.image_low = np.ones([num_grid, num_grid], dtype=np.float32) * (vrange)
        self.image_height = np.zeros([num_grid, num_grid], dtype=np.float32)
        self.image_n = np.zeros([num_grid, num_grid], dtype=np.int16)

        self.debug_count = 0

    def refresh(self):
        self.image_high.fill(-vrange)
        self.image_low.fill(vrange)
        self.image_height.fill(0)
        self.image_n.fill(0)

    def parse(self, data):  # data = np.array[30600]

        distance, azimuth = np.split(data.reshape(75, 408), (384,), axis=1)

        distance = distance.reshape(-1, 12, 2, 16) * 0.002
        azimuth = azimuth.reshape(-1, 12, 2, 1) * np.pi / 18000

        new_x = distance * new_cos_angs * np.cos(azimuth).repeat(16, axis=3)
        new_y = distance * new_cos_angs * np.sin(azimuth).repeat(16, axis=3)
        new_z = - distance * new_sin_angs  # 75 12 2 16
        new_z = new_z.flatten()

        pos_x = np.floor((new_x + image_size / 2) / grid_size).astype(np.int).flatten()
        pos_y = np.floor((new_y + image_size / 2) / grid_size).astype(np.int).flatten()

        mask = np_logical_and_list(0 <= pos_x, pos_x < num_grid,
                                   0 <= pos_y, pos_y < num_grid,
                                   distance.flatten() > min_dis,
                                   -vrange < new_z, new_z < vrange)

        new_z = new_z[mask]
        pos_x = pos_x[mask]
        pos_y = pos_y[mask]

        high_indices = new_z.argsort()
        high_pos_x = pos_x[high_indices]
        high_pos_y = pos_y[high_indices]

        self.image_high[high_pos_x, high_pos_y] = \
            np.maximum(self.image_high[high_pos_x, high_pos_y], new_z[high_indices])

        low_indices = high_indices[::-1]
        low_pos_x = pos_x[low_indices]
        low_pos_y = pos_y[low_indices]

        self.image_low[low_pos_x, low_pos_y] = \
            np.minimum(self.image_low[low_pos_x, low_pos_y], new_z[low_indices])

        n_counts = np.bincount(self.image_n.shape[0] * pos_x + pos_y)
        n_indices = np.unravel_index(np.arange(n_counts.size), dims=self.image_n.shape)
        self.image_n[n_indices] = n_counts

    def update(self, lidar_data, extra_data):
        self.refresh()
        self.posx, self.posy, self.roll, self.pitch, self.yaw, self.posx102, self.posy102 \
            = extra_data[:7]
        self.rot_m = rotmatrix(self.rot_m, self.roll, self.pitch, self.yaw)
        self.parse(lidar_data)

        return {
            "posx": self.posx,
            "posy": self.posy,
            "runtime": self.runtime,
            "image_high": self.image_high,
            "image_low": self.image_low,
            "image_n": self.image_n
        }


ESC = 27
if __name__ == "__main__":  # This part is not allow to changed! Used for profiling!
    # from utils import Visualizer
    # # from VLP import setup_vlp
    from utils import read_data, setup_logger, FPSTimer

    setup_logger()

    data = read_data("data/profiling/profiling.h5")
    lidar_data = data['lidar_data']
    extra_data = data['extra_data']
    # assert lidar_data.shape[1] == 30600

    image = Image()
    # vlp = setup_vlp()
    # v = Visualizer("image_n", image_size, zoom=1)

    # from utils import FPSTimer

    fpstimer = FPSTimer()

    # while True:
    for l, e in zip(lidar_data, extra_data):
        with fpstimer:
            ret = image.update(l, e)
            # lidar_data, extra_data = vlp.update()
            # ret = image.update(lidar_data, extra_data)
            # stop = v.draw(image.image_high)
            # if stop:
            #     break
            # while True:
            #     lidar_data, extra_data = vlp.update()
            #     image.update(lidar_data, extra_data)
            #     stop = v.draw(image.image_n)
            #     if stop:
            #         break
            #
            # close_vlp(vlp)
