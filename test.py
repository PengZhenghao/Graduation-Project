import logging
import pickle

from msgdev import MsgDevice


def build_push_data(object_dict, target):
    ret = {k: {"centroid": v["centroid"].tolist()} for k, v in object_dict.items()}
    ret["target"] = target
    # return ret
    return pickle.dumps(ret)


def build_push_detection_dev(push_detection_dev):
    push_detection_dev.open()
    # push_detection_dev.pub_bind('tcp://0.0.0.0:55010')  # 上传端口
    push_detection_dev.sub_connect("tcp://192.168.0.8:55019")
    push_detection_dev.sub_add_url("det.data", [-100] * 7)


if __name__ == "__main__":
    push_detection_dev = MsgDevice()
    build_push_detection_dev(push_detection_dev)

    from msgdev import PeriodTimer

    t = PeriodTimer(0.1)
    try:
        while True:
            with t:
                data = push_detection_dev.sub_get("det.data")
                print(data)
                # print(data, pickle.loads(data))

    finally:
        push_detection_dev.pub_set("data", pickle.dumps(None))
        push_detection_dev.close()
        logging.info("Everything Closed!")
