import os

from VLP import setup_vlp
from build_map import Image
from utils import FPSTimer
from utils import Visualizer, Reader


class Detector(object):
    def __init__(self, source):
        # setup all objects.
        if os.path.isfile(source) and source.endswith("h5"):
            # Open the h5 file.
            vlp = Reader(source)
        elif source == "live":
            vlp = setup_vlp()
        else:
            raise ValueError("Source should be string 'live' or the path to h5 file!")
        self.data_provider = vlp
        # self.slam = SLAM()
        self.image = Image()
        self.visualizer = Visualizer("image_n", side_length=50, zoom=5)  # TODO remove args.

    def update(self):
        # two way getting data: from live stream, or from file.
        # make this choice invisible to the user.
        # get the new "frame raw data"
        lidar_data, extra_data = self.data_provider.update()
        ret = self.image.update(lidar_data, extra_data)
        # if

    def render(self):
        # pop up a OpenCV window. I am not really sure what to show, cause many info need to show.
        stop = self.visualizer.draw(self.image.image_n)
        return stop

    def close(self):
        # clean up everything
        self.data_provider.close()


import argparse
from utils import setup_logger

if __name__ == '__main__':
    # Only a skeleton, WIP

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    # parser.add_argument("--timestep", "-t", default=-1, type=int)
    # parser.add_argument("--fake-lidar", '-f', action="store_true", default=False)
    parser.add_argument("--render", '-r', action="store_true", default=False)
    # parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, default="examples")
    parser.add_argument("--no-camera", "-nc", action="store_true", default=False)
    args = parser.parse_args()

    setup_logger(args.log_level)

    d = Detector("data/0503lib/0503lib.h5")
    timer = FPSTimer()

    while True:
        with timer:
            try:
                d.update()
                if args.render:
                    if d.render():
                        break
            except KeyboardInterrupt as e:
                print("Keyboard Interrupted!")
                break
            except EOFError:
                print("Data Source Dried Up!")
                break
    d.close()
