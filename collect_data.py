import argparse
import logging
import time

from VLP import setup_vlp, close_vlp
from camera import setup_camera, close_camera, shot
from recorder import AsyncRecorder
from utils import setup_logger

"""
Example usage: python collect_data.py --exp-name 20190418-test1 -m
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--timestep", "-t", default=-1, type=int)
    parser.add_argument("--fake-lidar", '-f', action="store_true", default=False)
    parser.add_argument("--monitoring", '-m', action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()

    setup_logger(args.log_level)

    recorder = AsyncRecorder({"exp_name": args.exp_name, "save_dir": "experiments"}, monitoring=args.monitoring)
    vlp = setup_vlp(args.fake_lidar)
    cam = setup_camera()
    now = time.time()

    st = now
    cnt = 0
    log_interval = 10

    try:
        logging.info("Start Record Data!")
        while True:
            logging.debug("The {} iteration!".format(cnt))

            lidar_data, extra_data = vlp.update()
            frame = shot(cam)
            data_dict = {"lidar_data": lidar_data, "extra_data": extra_data, "frame": frame}
            recorder.add(data_dict)
            cnt += 1
            if args.timestep > 0 and cnt == args.timestep:
                break
    except KeyboardInterrupt:
        logging.critical("KEYBOARD INTERRUPTED!")
    finally:
        et = time.time()
        logging.info("Recording Finish! It take {} seconds and collect {} data! Average FPS {}.".format(
            et - st, cnt, cnt / (et - st)))
        close_vlp(vlp)
        close_camera(cam)
        recorder.close()
