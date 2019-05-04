from math import sin, cos, floor, ceil, pi

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


class Image:
    def __init__(self):
        # create data document
        self.rot_m = np.zeros((3, 3))  # rotation matrix
        self.runtime = 0
        self.posx = 0
        self.posy = 0
        self.image_high = np.ones([num_grid, num_grid]) * (-vrange)
        self.image_low = np.ones([num_grid, num_grid]) * (vrange)
        self.image_height = np.zeros([num_grid, num_grid])
        self.image_n = np.zeros([num_grid, num_grid])

    def refresh(self):
        self.image_high.fill(-vrange)
        self.image_low.fill(vrange)
        self.image_height.fill(0)
        self.image_n.fill(0)

    def parse(self, data):
        for k in range(num_packet):
            dis384 = data[k * 408:(k * 408 + 384)].reshape(12, 2, 8, 2) * 0.002
            azi24 = data[(k * 408 + 384):(k * 408 + 408)].reshape(12, 2) / 18000 * pi
            for i in range(12):
                for j in range(2):
                    dis = dis384[i, j]
                    azi = azi24[i, j]
                    xs = dis * cos_angs * cos(azi)
                    ys = dis * cos_angs * sin(azi)
                    zs = -dis * sin_angs
                    for p in range(8):
                        for q in range(2):
                            if dis[p, q] > min_dis:
                                self.drawpoint(xs[p, q], ys[p, q], zs[p, q])

    def drawpoint(self, x, y, h):
        if -vrange < h < vrange:
            pos_x = floor((x + image_size / 2) / grid_size)
            pos_y = floor((y + image_size / 2) / grid_size)

            if 0 <= pos_x < num_grid and 0 <= pos_y < num_grid:
                if h > self.image_high[pos_x, pos_y]:
                    self.image_high[pos_x, pos_y] = h
                if h < self.image_low[pos_x, pos_y]:
                    self.image_low[pos_x, pos_y] = h
                self.image_n[pos_x, pos_y] += 1

    def update(self, lidar_data, extra_data):

        self.refresh()
        self.posx, self.posy, self.roll, self.pitch, self.yaw, self.posx102, self.posy102 = extra_data[:7]
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
if __name__ == "__main__":
    from utils import Visualizer
    from VLP import setup_vlp

    # data = read_data("raw_data/data/2019-04-25_13-48-02/2019-04-25_13-48-02.h5")
    # lidar_data = data['lidar_data']
    # extra_data = data['extra_data']
    # assert lidar_data.shape[1] == 30600

    image = Image()
    vlp = setup_vlp()
    v = Visualizer("image_n", image_size, zoom=1)

    from utils import FPSTimer

    fpstimer = FPSTimer()

    while True:
        with fpstimer:
            lidar_data, extra_data = vlp.update()
            ret = image.update(lidar_data, extra_data)
            stop = v.draw(image.image_high)
            if stop:
                break
                # while True:
                #     lidar_data, extra_data = vlp.update()
                #     image.update(lidar_data, extra_data)
                #     stop = v.draw(image.image_n)
                #     if stop:
                #         break
                #
                # close_vlp(vlp)
