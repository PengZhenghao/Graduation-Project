import argparse
import logging
import time

from VLP import setup_vlp, close_vlp
from recorder import Recorder
from utils import setup_logger

"""
Example usage: python collect_data.py --exp-name 20190418-test1 -m
"""

if __name__ == '__main__':

    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--timestep", "-t", default=-1, type=int)
    parser.add_argument("--fake-lidar", '-f', action="store_true", default=False)
    parser.add_argument("--monitoring", '-m', action="store_true", default=False)
    # parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--save-dir", type=str, default="examples")
    parser.add_argument("--no-camera", "-nc", action="store_true", default=False)
    args = parser.parse_args()

    setup_logger(args.log_level)

    args.save_dir = os.path.abspath(args.save_dir)
    logging.info("Current Save Directory: ".format(args.save_dir))

    # recorder = AsyncRecorder({"exp_name": args.exp_name,
    recorder = Recorder({"exp_name": args.exp_name,
                         "save_dir": args.save_dir,
                         "dataset_names": ("lidar_data", "extra_data", "timestamp"),
                         "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32",
                                            "timestamp": "float64"},
                         "dataset_shapes": {"lidar_data": (30600,), "extra_data": (8,),
                                            "timestamp": (1,)},
                         "use_video_writer": False
                         },
                        monitoring=args.monitoring)
    vlp = setup_vlp(args.fake_lidar)

    if not args.no_camera:
        from camera import setup_camera, close_camera, shot

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
            if not args.no_camera:
                frame = shot(cam)
                data_dict = {"lidar_data": lidar_data, "extra_data": extra_data, "frame": frame}
            else:
                data_dict = {"lidar_data": lidar_data, "extra_data": extra_data}
            recorder.add(data_dict)
            cnt += 1
            if args.timestep > 0 and cnt == args.timestep:
                break
    except KeyboardInterrupt:
        logging.critical("KEYBOARD INTERRUPTED!")
    finally:
        et = time.time()
        logging.info(
            "Recording Finish! It take {} seconds and collect {} data! Average FPS {}.".format(
                et - st, cnt, cnt / (et - st)))
        close_vlp(vlp)
        if not args.no_camera:
            close_camera(cam)
        recorder.close()
