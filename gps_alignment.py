# from PID import PID
from math import atan2, pi

from msgdev import MsgDevice, PeriodTimer
from utils import MovingAverage


class Interface(object):
    def __init__(self, sub_addr, ahrs_port, gnss_port, motor_port):
        self.dev = MsgDevice()
        self.dev.open()
        self.dev.sub_connect(sub_addr + ':' + ahrs_port)
        self.dev.sub_add_url('ahrs.roll')
        self.dev.sub_add_url('ahrs.pitch')
        self.dev.sub_add_url('ahrs.yaw')
        self.dev.sub_add_url('ahrs.roll_speed')
        self.dev.sub_add_url('ahrs.pitch_speed')
        self.dev.sub_add_url('ahrs.yaw_speed')
        self.dev.sub_add_url('ahrs.acce_x')
        self.dev.sub_add_url('ahrs.acce_y')
        self.dev.sub_add_url('ahrs.acce_z')

        self.dev.sub_connect(sub_addr + ':' + gnss_port)
        self.dev.sub_add_url('gps.time')
        self.dev.sub_add_url('gps.posx')
        self.dev.sub_add_url('gps.posy')
        self.dev.sub_add_url('gps.posz')
        self.dev.sub_add_url('gps.stdx')
        self.dev.sub_add_url('gps.stdy')
        self.dev.sub_add_url('gps.stdz')
        self.dev.sub_add_url('gps.satn')
        self.dev.sub_add_url('gps.hspeed')
        self.dev.sub_add_url('gps.vspeed')
        self.dev.sub_add_url('gps.track')

    def receive(self, *args):
        data = []
        for i in args:
            data.append(self.dev.sub_get1(i))
        return data


class PortPzh:
    def __init__(self, sub_addr, object_port):
        self.dev = MsgDevice()
        self.dev.open()
        self.dev.sub_connect(sub_addr + ':' + object_port)
        self.dev.sub_add_url("det.data", [-100] * 7)  # -100表示信号丢失，-99表示程序断了。

    def receive(self):
        data = self.dev.sub_get("det.data")
        # assert isinstance(data, list), "data should be a list"
        assert len(data) == 7, "data should have 3*2 position and 1 target, totally 7 numbers"
        ret = {
            "Object 0": [data[0], data[1]] if data[0] not in [-100, -99] else None,
            "Object 1": [data[2], data[3]] if data[2] not in [-100, -99] else None,
            "Object 2": [data[4], data[5]] if data[4] not in [-100, -99] else None,
            "target": "Object {}".format(int(data[6])) if int(data[6]) in [0, 1, 2] else None,
            "terminated": all([d == -99 for d in data])
        }
        return ret


def ship_initialize(USE_TLG001, USE_TLG002, USE_PZH):
    if USE_TLG001:
        sub_addr1 = 'tcp://192.168.1.150'  # 'tcp://127.0.0.1'
        ahrs_port1 = '55005'
        gnss_port1 = '55004'
        motor_port1 = '55002'
        interface001 = Interface(sub_addr1, ahrs_port1, gnss_port1, motor_port1)
    else:
        interface001 = None

    if USE_TLG002:
        sub_addr2 = 'tcp://192.168.1.152'  # 'tcp://127.0.0.1'
        ahrs_port2 = '55205'
        gnss_port2 = '55204'
        motor_port2 = '55202'
        interface002 = Interface(sub_addr2, ahrs_port2, gnss_port2, motor_port2)
    else:
        interface002 = None

    if USE_PZH:
        sub_addr3 = 'tcp://192.168.1.222'
        object_port = '55019'
        pzhdata = PortPzh(sub_addr3, object_port)
    else:
        pzhdata = None
    return interface001, interface002, pzhdata


# 下标宏定义
POS_X = 0
POS_Y = 1
YAW = 2
YAW_SPEED = 3
SPD = 4
SPD_DIR = 5


def main():
    USE_PZH = True

    # initialize
    interface001, interface002, pzhdata = ship_initialize(True, True, True)

    t = PeriodTimer(0.2)

    diff_x_average_gps = MovingAverage(100)
    diff_y_average_gps = MovingAverage(100)

    diff_x_average_lidar = MovingAverage(100)
    diff_y_average_lidar = MovingAverage(100)

    # t.start()

    cnt = 0
    end = 200

    try:
        while True:
            with t:
                self_state = interface001.receive('gps.posx', 'gps.posy', 'ahrs.yaw',
                                                  'ahrs.yaw_speed', 'gps.hspeed',
                                                  'gps.stdx', 'gps.stdy', 'gps.track')

                target_state = interface002.receive('gps.posx', 'gps.posy', 'ahrs.yaw',
                                                    'ahrs.yaw_speed', 'gps.hspeed',
                                                    'gps.stdx', 'gps.stdy', 'gps.track')

                assert pzhdata is not None
                lidar_data = pzhdata.receive()

                if lidar_data["terminated"]:
                    print(
                        "Peng Zhenghao's program is terminated. For safety we close this program.")
                    break

                target = lidar_data["target"]
                if not target:
                    print("No Target Specified!")
                    continue
                else:
                    cnt += 1
                    # print("Current CNT")
                    goal = lidar_data[target]  # goal = [x, y]
                    diff_x = target_state[POS_X] - self_state[POS_X]
                    diff_y = target_state[POS_Y] - self_state[POS_Y]
                    diff_x_average_gps.update(diff_x)
                    diff_y_average_gps.update(diff_y)
                    diff_x_average_lidar.update(goal[0])
                    diff_y_average_lidar.update(goal[1])

                    phi2 = -atan2(diff_y_average_gps.avg, diff_x_average_gps.avg) - pi / 2
                    phi1 = atan2(diff_y_average_lidar.avg, diff_x_average_lidar.avg)
                    out = phi1 + phi2 - pi / 2

                    # offset = atan2(diff_y_average_lidar.avg, diff_x_average_lidar.avg) - \
                    #          atan2(diff_y_average_gps.avg, diff_x_average_gps.avg)

                    print("[CNT {}] Current GPS ({}, {}), LiDAR ({}, {}). \
                    ph1{}, ph2 {}, out {} ({} deg).".format(cnt, diff_x_average_gps.avg,
                                                            diff_y_average_gps.avg,
                                                            diff_x_average_lidar.avg,
                                                            diff_y_average_lidar.avg,
                                                            phi1, phi2, out,
                                                            out * 180 / pi))
                    if cnt >= end:
                        break

    finally:
        import pickle
        import time

        def get_formatted_time(timestamp=None):
            if not timestamp:
                return time.strftime('%Y-%m-%d_%H-%M-%S',
                                     time.localtime())
            else:
                return time.strftime('%Y-%m-%d_%H-%M-%S',
                                     time.localtime(timestamp))

        if diff_y_average_lidar.avg is not None:
            phi2 = -atan2(diff_y_average_gps.avg, diff_x_average_gps.avg) - pi / 2
            phi1 = atan2(diff_y_average_lidar.avg, diff_x_average_lidar.avg)
            out = phi1 + phi2 - pi / 2

            out = atan2(diff_y_average_lidar.avg, diff_x_average_lidar.avg) - \
                  atan2(diff_y_average_gps.avg, diff_x_average_gps.avg)
            pickle.dump({"offset": out, "timestamp": time.time(), "time": get_formatted_time()},
                        open("offset.pkl", "wb"))
            print("Data have saved to offset.pkl")
        else:
            print("Data is not received.")

        time.sleep(0.5)
        interface001.dev.close()
        interface002.dev.close()
        pzhdata.dev.close()
        print('dev closed')


if __name__ == "__main__":
    main()
