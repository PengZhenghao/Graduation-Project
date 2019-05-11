#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from config import slam_config
from utils import np_logical_and_list

# Image Settings
# Cluster Settings
# minarea_cluster = 3
# neighborsize = 2
# mindensity_normal = 3
# mindensity_fast = 6
# Object Settings
# maxlen_obj = 4
# maxlow_obj = 1.5
# Cluster to object settings
# c2o_hlimit = 0.5  # maximum difference in h
# c2o_alimit = 5  # maximum difference in area
# c2o_rlimit = 0.2  # maximum difference in B/L
# c2o_vlimit = 3  ##maximum velocity
#
# # Object to map settings
# o2m_dlimit = 1  # maximum moving distance of a static obstacle
#
# # Confidence level settings
# maxlevel_obj = 24
# minlevelgood_obj = 4
# # confidence_max = 20
# # confidence_min = 10  # minimum confidence level of static obstacles in global map
#
# # Kalman Filter of vx,vy settings
# kf_p0 = 1
# kf_q = 0.25
# kf_r = 1
# def pca(coor):
#     data_adj = coor - np.mean(coor, axis=0)
#     cov = np.cov(data_adj, rowvar=False)
#     eigVals, eigVects = np.linalg.eig(np.mat(cov))
#     eigValInd = np.argsort(eigVals)
#     redEigVects = eigVects[:, eigValInd]
#     mainvec0 = redEigVects[0, 1]
#     mainvec1 = redEigVects[1, 1]
#     if mainvec0 == 0:
#         heading = pi / 2
#     else:
#         heading = atan(mainvec1 / mainvec0)
#     return data_adj.dot(redEigVects), heading
# def kf(vx_old, vy_old, vx_sensor, vy_sensor, var_old):
#     k = (var_old + kf_q) / (var_old + kf_q + kf_r)
#     vx = (1 - k) * vx_old + k * vx_sensor
#     vy = (1 - k) * vy_old + k * vy_sensor
#     var = (1 - k) * (var_old + kf_q)
#     return vx, vy, var
# def sub_image(subnum, data, title, picsize, cmaptype='viridis'):
#     plt.subplot(subnum)
#     img = data * 1
#     for i in range(picsize):
#         img[:, i] = img[:, i][::-1]
#     imgplot = plt.imshow(img, cmap=cmaptype)
#     plt.xlabel('y')
#     plt.ylabel('x')
#     plt.colorbar()
#     plt.title(title)

EMPTY = -1

from utils import build_index_to_location, build_location_to_index


class SLAM(object):
    def __init__(self, config=slam_config):

        self.config = config

        self.grid_size = self.config["grid_size"]
        self.min_cluster_occupied_grids = self.config["min_cluster_occupied_grids"]
        # self.min_confidence = self.config["min_confidence"]

        self.map_size = self.config["map_size"]
        self.num_grid = round(self.map_size / self.grid_size)

        self.shape = (self.num_grid, self.num_grid)
        self.location_to_index = build_location_to_index(self.config)
        self.index_to_location = build_index_to_location(self.config)

        # CHART 是静态海图的意思！
        # self.global_chart = np.zeros((int_ceil((x_chart[1] - x_chart[0]) / self.grid_size),
        #                               int_ceil((y_chart[1] - y_chart[
        #                                   0]) / self.grid_size)))  # initialize global chart
        # self.global_chart = [1000, 1000]
        # self.local_chart = None


        # for i in range(int_ceil((x_chart[1] - x_chart[0]) / self.grid_size)):
        #     for j in range(int_ceil((y_chart[1] - y_chart[0]) / self.grid_size)):
        #         self.global_chart[i, j] = chart[i // num_chartrans, j // num_chartrans]

        # self.global_confidence_map = self.global_chart * self.config["min_confidence"]
        # initialize global static obstacle map

        # self.global_confidence_map_output = np.zeros((num_out_x, num_out_y))
        self._confidence_map = np.zeros(self.shape, dtype=np.int)

        self._object_dict = {}

        self._object_label_waiting_list = []

        self._detected_object_number = 0

        self._cluster_map = np.ones(self.shape, dtype=np.int) * EMPTY
        # self._occupied_map = np.zeros(self.shape, dtype=np.bool)

        self.debug = {}

        # initialize output global static obstacle map

        # LOOK AT HERE! TODO here is the format of object!
        # [area,x_min,x_max,y_min,y_max,l,b,heading,h,vx,vy,
        # var_v,dis_x,dis_y,confidence level,cluster_num,object label]

        self.objects = np.zeros((0, 17))
        self.runtime = 0
        self.num_pub = 0
        # self.object_label = 0

        self.clustering_algo = DBSCAN(eps=self.config["neighborhood_size"],
                                      min_samples=self.config[
                                          "neighborhood_min_samples"])  # min sample=6 should be the FAST MODE
        self.pca = PCA()  # min sample=6 should be the FAST MODE
        # Use config to change the hyper
        # TODO check if the input is in physical metrics,.

        # self.clustering_algo = DBSCAN(eps=1.5, min_samples=6) # Use config to change the hyper
        logging.info('Global Map Created.')

    # def _compute_iou(self, previous_points, current_points):
    def _compute_overlap(self, object, cluster):
        # Now the intesection over union is not proper here.
        # Since we are using the bounding box to serve as the target.
        # So it can be naturally a very small IoU even if the cluster is in the bounding box.
        # Therefore I choose to use ``the intersection over the cluster``,
        # which is still not greater than 1.

        current_points = cluster["occupied_grids"]

        # assert current_points.dtype == np.int
        # assert previous_points.dtype == np.int
        # assert current_points.ndim == 2
        # assert previous_points.ndim == 2

        x_min, y_min, x_max, y_max = object["bounding_box"]

        # points1_flat = current_points[:, 0] * self.num_grid + current_points[:, 1]
        # points2_flat = np.arange(x_min, x_max + 1)


        # np+current_points[:, 0]<=x_max
        intersection = np_logical_and_list(x_min <= current_points[:, 0],
                                           current_points[:, 0] <= x_max,
                                           y_min <= current_points[:, 1],
                                           current_points[:, 1] <= y_max).sum()
        return intersection / (current_points.shape[0] + 1e-6)

    def _remove_object(self, obj_label):
        assert self._object_dict[obj_label]["confidence"] == 0
        if obj_label not in self._object_dict:
            logging.warning("obj_label {} not found in self._cluster_dict, \
            which contains {}.".format(obj_label, self._object_dict))
            return
        self._object_label_waiting_list.append(obj_label)
        obj_info = self._object_dict.pop(obj_label)
        occupied_grids = obj_info["occupied_grids"]
        self._cluster_map[occupied_grids[:, 0], occupied_grids[:, 1]] = EMPTY

    def _fit(self, points, weight):
        if len(points) == 0:
            return None
        labels = self.clustering_algo.fit_predict(points, sample_weight=weight)
        return labels

    def _find_cluster(self, points, weight, high, low, occupied_grids):

        if len(points) < 1:
            logging.warning("No point found in receptive field.")
            return {}

        labels = self._fit(points, weight)

        self.debug["labels"] = labels

        cluster_num = labels.max()

        if cluster_num < 0:
            logging.debug("No cluster found.")
            return {}

        raw_cluster_dict = {
            i:
                {  # cluster info
                    "points": points[labels == i],
                    "weight": weight[labels == i],
                    "high": high[labels == i],
                    "low": low[labels == i],
                    "mask": labels == i,
                    "occupied_grids": occupied_grids[labels == i]
                }
            for i in range(cluster_num)
        }

        for label, cluster_info in raw_cluster_dict.items():
            points = cluster_info["points"]
            indices = self.location_to_index(points)
            self._cluster_map[indices] = label

        return raw_cluster_dict

    def _process_cluster(self, raw_cluster_dict):
        assert isinstance(raw_cluster_dict, dict)
        cluster_properties = {}

        for label, cluster_info in raw_cluster_dict.items():
            # locations = cluster_info["points"]

            points = cluster_info["points"]
            # assert points.shape[1] == 2
            # assert points.ndim == 2

            occupied_grids = len(points)
            # assert occupied_grids == len(np.unique(points, axis=0))

            # calculate basic properties of each cluster
            high_cluster = cluster_info["high"].max()  # get the highest
            low_cluster = cluster_info["low"].min()  # get the lowest

            weight = cluster_info["weight"]
            centroid = np.dot(weight, points) / weight.sum()

            length = sqrt(np.sum(np.square(points - centroid), axis=1).max()) * 2

            cluster_properties[label] = {
                "area": occupied_grids * self.grid_size * self.grid_size,
                # "x_min": points.
                "length": length,
                "centroid": centroid,
                "density": cluster_info["weight"].sum() / occupied_grids,
                "occupied": cluster_info["occupied_grids"].shape[0],
                # "b": b_cluster,
                # "heading": heading_cluster,
                "high": high_cluster,
                "low": low_cluster,
                "label": label,
                "occupied_grids": cluster_info["occupied_grids"],
                # "confidence": 0.0,
            }
        return cluster_properties

    def _get_new_label(self):
        if len(self._object_label_waiting_list) == 0:
            new_label = "Object {}".format(self._detected_object_number)
            self._detected_object_number += 1
        else:
            new_label = self._object_label_waiting_list.pop(0)
        return new_label

    def _should_discard_cluster(self, cluster):
        if len(cluster["occupied_grids"]) < self.min_cluster_occupied_grids:
            # some clusters were absorbed by others, some are too small to be an object
            return True
        if cluster["length"] > self.config["max_object_length"]:
            return True
        if cluster["high"] > self.config["max_object_bottom_height"]:
            return True
        return False

    def _increase_confidence(self, label):
        assert label in self._object_dict
        self._object_dict[label]["confidence"] = min(self._object_dict[label]["confidence"] + 1,
                                                     self.config["max_confidence"])

    def _decrease_confidence(self, label):
        assert label in self._object_dict
        self._object_dict[label]["confidence"] = max(self._object_dict[label]["confidence"] - 1, 0)

    def _create_object(self, cluster):
        label = self._get_new_label()
        cluster["confidence"] = self.config["init_confidence"]
        cluster["name"] = label
        points = cluster["occupied_grids"]
        cluster["bounding_box"] = points.min(0).tolist() + points.max(0).tolist()
        cluster["status"] = "JUST FOUND"
        self._object_dict[label] = cluster
        return label

    def _update_object(self, obj_label, cluster):
        points = cluster["occupied_grids"]
        cluster["bounding_box"] = points.min(0).tolist() + points.max(0).tolist()
        self._object_dict[obj_label].update(cluster)
        self._object_dict[obj_label]["status"] = "CONFIRMING" \
            if self._object_dict[obj_label]["confidence"] < self.config["min_detected_confidence"] \
            else "TRACKING"

    def _lost_object(self, obj_label):
        self._decrease_confidence(obj_label)
        self._object_dict[obj_label]["status"] = "LOST"
        if self._object_dict[obj_label]["confidence"] <= \
                self.config["min_removal_confidence"]:
            self._remove_object(obj_label)

    def _register_cluster(self, cluster_properties):
        objects_to_cluster = {k: [] for k in self._object_dict.keys()}

        for cluster_label, cluster in cluster_properties.items():  # for each cluster
            possible_objects = {}

            for label, obj in self._object_dict.items():  # for each object
                overlap = self._compute_overlap(obj, cluster)

                if overlap > self.config["overlap_threshold"]:
                    possible_objects[label] = overlap
                    objects_to_cluster[label].append(cluster_label)

            should_discard = self._should_discard_cluster(cluster)

            if not possible_objects:  # No match existing object
                if should_discard:
                    continue
                self._create_object(cluster)
            else:
                label = max(possible_objects)  # max would return the "key" of the maximum "value".
                if should_discard:
                    self._decrease_confidence(label)
                else:
                    self._increase_confidence(label)
                self._update_object(label, cluster)

        for obj_label, obj_cluster in objects_to_cluster.items():
            if len(obj_cluster) == 0:
                self._lost_object(obj_label)

        return self._object_dict

    @property
    def object_dict(self):
        return {obj_label: obj_info
                for obj_label, obj_info in self._object_dict.items()
                if obj_info["confidence"] > self.config["min_detected_confidence"]}

    def update(self, input_dict):

        if input_dict["points"] is None:
            return None

        # if input_dict

        # points, weight, high, low,
        self._cluster_map.fill(EMPTY)

        # newtime = runtime
        # self.interval = newtime - self.runtime

        ### PENGZHENGHAO WORKAROUND
        self.interval = 1

        # if self.interval != 0:
        # self.runtime = newtime
        # self.image_high = map_high
        # self.image_low = map_low
        # self.image_n = map_n
        # self._occupied_map = (self.image_n >= image_n_std) * 1
        # self.get_localmap(posx, posy)
        # self.confirm_localmap()
        # cluster_dict, labels_range = self.find_cluster(kwargs["points"], kwargs["weight"])
        points, weight, high, low = input_dict["points"], input_dict["weight"], input_dict["high"], \
                                    input_dict["low"]

        occupied_grids = input_dict["indices"]

        raw_cluster_dict = self._find_cluster(points, weight, high, low,
                                              occupied_grids)
        # if raw_cluster_dict is None:
        #     return None

        prop = self._process_cluster(raw_cluster_dict)

        self._register_cluster(prop)

        return {"cluster_map": self._cluster_map,
                # "object_dict": self.object_dict}
                "object_dict": self._object_dict}


        # self.analyze_cluster(0, 0, raw_cluster_dict)
        # self.map_cluster_to_object()
        # self.map_object_to_static_obstacle()
        # self.refresh_localmap()
        # self.refresh_globalmap(posx, posy)
        # self.publish(dev, posx, posy)

        # inf = [posx, posy, self.runtime]
        # globalmap_size = [num_out_x, x_global[0], x_global[1], num_out_y, y_global[0], y_global[1]]
        # objnum = len(self.objects)

        # objects_out = np.zeros((100, 17))
        # objects_out[0:objnum, :] = self.objects

        # self.num_pub += 1
        # return {
        #     'pos&time': inf,
        #     # publish image information
        #     'image_high': self.image_high,
        #     'image_low': self.image_low,
        #     'image_n': self.image_n,
        #     'cluster_map': self._cluster_map,
        #     'image_gain': self._occupied_map,
        #     # publish global_confidence_map information
        #     'global_confidence_map': self.global_confidence_map_output,
        #     'globalmap_size': globalmap_size,
        #     # publish object information
        #     'objects': objects_out,
        #     # publish time information
        #     'timestamp': self.num_pub
        # }


if __name__ == "__main__":
    from utils import setup_logger, FPSTimer, Visualizer
    import os
    import argparse
    from lidar_map import LidarMap, lidar_config
    import logging
    from utils import lazy_read_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", '-r', action="store_true")
    parser.add_argument("--path", '-p', required=True, help="The path of target h5 file.")
    parser.add_argument("--save", '-s', action="store_true")
    args = parser.parse_args()

    assert args.path.endswith(".h5")
    args.save = args.save and (not os.path.exists(args.path.replace(".h5", ".pkl")))
    setup_logger("DEBUG")

    data = lazy_read_data(args.path)
    lidar_data = data['lidar_data'][17100:]
    extra_data = data['extra_data'][17100:]

    # lidar_data = data['lidar_data']
    # extra_data = data['extra_data']

    map = LidarMap(lidar_config)
    slam = SLAM()

    OBSERVE = "cluster_map"
    v = Visualizer(OBSERVE, slam_config["map_size"], smooth=False, max_size=800)

    fps_timer = FPSTimer(force_fps=None)

    l2i = build_location_to_index(lidar_config)

    save_data = []
    stop = False
    for l, e in zip(lidar_data, extra_data):
        with fps_timer:
            ret = map.update(l, e)

            mr = slam.update(ret)

            if args.save:
                save_data.append(ret)
            if args.render:
                if mr is not None:
                    stop = v.draw(ret["map_n"], objects=mr["object_dict"])
                # TODO add fps display.

                if stop:
                    break

    if args.save:
        import pickle

        try:
            pickle.dump(save_data, open(args.path.replace(".h5", ".pkl"), "wb"))
        except Exception as e:
            os.remove(args.path.replace(".h5", ".pkl"))
            raise e
