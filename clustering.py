#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import sqrt

import numpy as np
from sklearn.cluster import DBSCAN

from config import detector_config
from utils import np_logical_and_list, build_index_to_location, build_location_to_index

LOST = "LOST"
JUST_FOUND = "JUST_FOUND"
TRACKING = "TRACKING"
CONFIRMING = "CONFIRMING"


class DetectedObject(dict):
    def __init__(self, name, cluster_property, config):
        super(DetectedObject, self).__init__(cluster_property)
        self.config = config or detector_config
        self.config["num_grid"] = round(self.config["map_size"] / self.config["grid_size"])
        self["confidence"] = self.config["init_confidence"]
        self["name"] = name
        self["status"] = JUST_FOUND
        points = cluster_property["occupied_grids"]
        self["bounding_box"] = points.min(0).tolist() + points.max(0).tolist()
        self["life"] = 0

    def increase_confidence(self):
        self["confidence"] = min(self["confidence"] + 1, self.config["max_confidence"])

    def decrease_confidence(self):
        self["confidence"] = max(self["confidence"] - 1, self.config["min_removal_confidence"])

    def update_property(self, cluster_property):
        self.update(cluster_property)
        points = cluster_property["occupied_grids"]
        self["bounding_box"] = points.min(0).tolist() + points.max(0).tolist()
        self["life"] += 1
        self["status"] = CONFIRMING if self["confidence"] < self.config["min_detected_confidence"] \
            else TRACKING
        if self["status"] is not LOST and "search_range" in self:
            self.pop("search_range")
            self.pop("search_radius")

    def lost(self):
        self["life"] += 1
        if self["status"] == LOST:
            if "search_radius" not in self:
                search_radius = 5 * (self["length"] / self.config["grid_size"])
            else:
                search_radius = self["search_radius"] + self["length"] * 2
            x_min, y_min, x_max, y_max = self["bounding_box"]
            x_min = max(0, x_min - search_radius // 2)
            y_min = max(0, y_min - search_radius // 2)
            x_max = min(self.config["num_grid"], x_max + search_radius // 2)
            y_max = min(self.config["num_grid"], y_max + search_radius // 2)
            self["search_range"] = x_min, y_min, x_max, y_max
            self["search_radius"] = search_radius
        self["status"] = LOST
        if self["confidence"] <= self.config["min_removal_confidence"]:
            return True
        return False

    def __repr__(self):
        return "<{}: {} (Life: {}, Confidence: {})>".format(self['name'], self['status'],
                                                            self["life"], self['confidence'])


class Detector(object):
    def __init__(self, config=detector_config):
        self.config = config
        self.grid_size = self.config["grid_size"]
        self.min_cluster_occupied_grids = self.config["min_cluster_occupied_grids"]
        self.map_size = self.config["map_size"]
        self.num_grid = round(self.map_size / self.grid_size)
        self.shape = (self.num_grid, self.num_grid)
        self.location_to_index = build_location_to_index(self.config)
        self.index_to_location = build_index_to_location(self.config)

        self._confidence_map = np.zeros(self.shape, dtype=np.int)
        self._object_dict = {}
        self._object_label_waiting_list = []
        self._detected_object_number = 0
        # self.runtime = 0
        # self.num_pub = 0
        self.clustering_algo = DBSCAN(eps=self.config["neighborhood_size"],
                                      min_samples=self.config[
                                          "neighborhood_min_samples"])  # min sample=6 should be the FAST MODE
        logging.info('Global Map Created.')

    def _compute_object_overlap(self, obj1, obj2):

        x_min1, y_min1, x_max1, y_max1 = obj1["bounding_box"]
        x_min2, y_min2, x_max2, y_max2 = obj2["bounding_box"]

        intersection = max(0, (min(x_max1, x_max2) - max(x_min1, x_min2))) * \
                       max(0, (min(y_max1, y_max2) - max(y_min1, y_min2)))
        if intersection < 1e-6:
            return 0
        union = (y_max2 - y_min2) * (x_max2 - x_min2) + (y_max1 - y_min1) * (
            x_max1 - x_min1) - intersection
        return intersection / union

    def _compute_overlap(self, obj, cluster):
        # Now the intesection over union is not proper here.
        # Since we are using the bounding box to serve as the target.
        # So it can be naturally a very small IoU even if the cluster is in the bounding box.
        # Therefore I choose to use ``the intersection over the cluster``,
        # which is still not greater than 1.

        current_points = cluster["occupied_grids"]
        current_weight = cluster["occupied_weight"]
        if "search_range" in obj:
            x_min, y_min, x_max, y_max = obj["search_range"]
        else:
            x_min, y_min, x_max, y_max = obj["bounding_box"]
        intersection_mask = np_logical_and_list(x_min <= current_points[:, 0],
                                                current_points[:, 0] <= x_max,
                                                y_min <= current_points[:, 1],
                                                current_points[:, 1] <= y_max)
        intersection = current_weight[intersection_mask].sum()
        union = current_weight.sum()
        return intersection / union

    def _remove_object(self, obj_label):
        if obj_label not in self._object_dict:
            return
        self._object_label_waiting_list.append(obj_label)
        self._object_dict.pop(obj_label)

    def _fit(self, points, weight):
        if len(points) == 0:
            return None
        labels = self.clustering_algo.fit_predict(points, sample_weight=weight)
        return labels

    def _find_cluster(self, points, weight, high, low, occupied_grids):
        if len(points) < 1:
            logging.info("No point found in receptive field.")
            return {}
        labels = self._fit(points, weight)
        cluster_num = labels.max()
        if cluster_num < 0:
            logging.info("No cluster found in receptive field.")
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
            for i in range(0, cluster_num + 1)
        }
        return raw_cluster_dict

    def _process_cluster(self, raw_cluster_dict):
        cluster_properties = {}
        for label, cluster_info in raw_cluster_dict.items():
            # calculate basic properties of each cluster
            points = cluster_info["points"]
            weight = cluster_info["weight"]
            centroid = np.dot(weight, points) / weight.sum()
            length = sqrt(np.sum(np.square(points - centroid), axis=1).max()) * 2

            num_occupied_grids = len(points)
            cluster_properties[label] = {
                "area": num_occupied_grids * self.grid_size * self.grid_size,
                "length": length,
                "centroid": centroid,
                "density": weight.sum(),
                "occupied": cluster_info["occupied_grids"].shape[0],
                "high": cluster_info["high"].max(),
                "low": cluster_info["low"].min(),
                # "label": label,
                "occupied_grids": cluster_info["occupied_grids"],
                "occupied_weight": weight,
            }
        return cluster_properties

    def _get_new_label(self):
        if len(self._object_label_waiting_list) == 0:
            new_label = "Object {}".format(self._detected_object_number)
            self._detected_object_number += 1
        else:
            new_label = self._object_label_waiting_list.pop(0)
        return new_label

    def _create_object(self, cluster):
        label = self._get_new_label()
        self._object_dict[label] = DetectedObject(label, cluster, self.config)
        return label

    def _register_cluster(self, cluster_properties):
        modified_objects = set()
        checked_cluster = set()

        for label, obj in self._object_dict.items():  # for each object
            possible_clusters = {}
            if obj["status"] is not LOST:
                continue

            for cluster_label, cluster in cluster_properties.items():
                overlap = self._compute_overlap(obj, cluster)
                if overlap > self.config["overlap_threshold"]:
                    possible_clusters[cluster_label] = overlap

            if possible_clusters:
                # max would return the "key" of the maximum "value".
                target_cluster_label = max(possible_clusters)
                obj.update_property(cluster_properties[target_cluster_label])
                modified_objects.add(label)
                checked_cluster.add(target_cluster_label)

        for cluster_label, cluster in cluster_properties.items():  # for each cluster
            if cluster_label in checked_cluster: continue
            possible_objects = {}
            for label, obj in self._object_dict.items():  # for each object
                if label in modified_objects: continue
                overlap = self._compute_overlap(obj, cluster)
                if overlap > self.config["overlap_threshold"]:
                    possible_objects[label] = overlap
            if not possible_objects:  # No match existing object
                label = self._create_object(cluster)
            else:
                label = max(possible_objects)  # max would return the "key" of the maximum "value".
                self._object_dict[label].update_property(cluster)  # _update_object(label, cluster)
            modified_objects.add(label)

        all_keys = set(self._object_dict.keys())
        not_modified_objects = all_keys.difference(modified_objects)

        for obj_label in not_modified_objects:
            should_remove = self._object_dict[obj_label].lost()
            if should_remove:
                self._remove_object(obj_label)

        all_objects = list(self._object_dict.items())
        for obj_label1, obj1 in all_objects[:-1]:
            if obj_label1 not in self._object_dict: continue
            for obj_label2, obj2 in all_objects[1:]:
                if obj_label2 not in self._object_dict: continue
                overlap = self._compute_object_overlap(obj1, obj2)
                if overlap > self.config["overlap_threshold"]:
                    if obj1["confidence"] < obj2["confidence"]:
                        self._remove_object(obj_label1)
                    else:
                        self._remove_object(obj_label2)
        return self._object_dict

    def _should_decrease_confidence(self, obj_cluster):
        should = False
        if len(obj_cluster["occupied_grids"]) < self.min_cluster_occupied_grids:
            should = True
        if obj_cluster["length"] > self.config["max_object_length"]:
            should = True
        if obj_cluster["high"] > self.config["max_object_bottom_height"]:
            should = True
        if obj_cluster["status"] == LOST:
            should = True
        return should

    def _verify_objects(self):
        for obj_label, obj in self._object_dict.items():
            if self._should_decrease_confidence(obj):
                obj.decrease_confidence()
            else:
                obj.increase_confidence()

    @property
    def object_dict(self):
        return {obj_label: obj_info
                for obj_label, obj_info in self._object_dict.items()
                if obj_info["confidence"] > self.config["min_detected_confidence"]}

    @property
    def availiable_obejct_keys(self):
        ret = {}
        for obj_label, obj_info in self._object_dict.items():
            if obj_info["confidence"] > self.config["min_detected_confidence"]:
                index = obj_label.split(' ')[1]
                key = ord(str(index))
                ret[key] = obj_label
        return ret

    def update(self, input_dict):
        points, weight, high, low = input_dict["points"], input_dict["weight"], input_dict["high"], \
                                    input_dict["low"]

        occupied_grids = input_dict["indices"]
        if points is not None:
            raw_cluster_dict = self._find_cluster(points, weight, high, low,
                                                  occupied_grids)
            prop = self._process_cluster(raw_cluster_dict)
        else:
            prop = {}

        self._register_cluster(prop)

        self._verify_objects()

        return self._object_dict


if __name__ == "__main__":
    from utils import setup_logger, FPSTimer, Visualizer, ESC
    import os
    import argparse
    from lidar_map import LidarMap, lidar_config
    import logging
    from utils import lazy_read_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", '-r', action="store_true")
    parser.add_argument("--path", '-p', required=True, help="The path of target h5 file.")
    parser.add_argument("--save", '-s', action="store_true")
    parser.add_argument("--fps", type=int, default=-1)
    args = parser.parse_args()

    assert args.path.endswith(".h5")
    args.save = args.save and (not os.path.exists(args.path.replace(".h5", ".pkl")))
    setup_logger("DEBUG")

    data = lazy_read_data(args.path)

    start = 17100
    # start = 17300

    lidar_data = data['lidar_data'][start:]
    extra_data = data['extra_data'][start:]

    map = LidarMap(lidar_config)
    detector = Detector()

    OBSERVE = "cluster_map"
    v = Visualizer(OBSERVE, detector_config["map_size"], smooth=True, max_size=1000)

    if args.fps == -1:
        args.fps = None
    fps_timer = FPSTimer(force_fps=args.fps)

    l2i = build_location_to_index(lidar_config)

    save_data = []
    stop = False
    pressed_key = -1
    target = None

    for l, e in zip(lidar_data, extra_data):
        with fps_timer:
            ret = map.update(l, e)
            object_d = detector.update(ret)
            avaliables = detector.availiable_obejct_keys
            if pressed_key in avaliables:
                target = avaliables[pressed_key]
            elif pressed_key == ord("q"):
                target = None
            elif target in object_d.keys():
                pass
            else:
                target = None

            if args.save:
                save_data.append(ret)
            if args.render:
                pressed_key = v.draw(ret["map_n"], objects=object_d,
                                     target=target or list(avaliables.values()))
                if pressed_key == ESC:
                    break

    if args.save:
        import pickle

        try:
            pickle.dump(save_data, open(args.path.replace(".h5", ".pkl"), "wb"))
        except Exception as e:
            os.remove(args.path.replace(".h5", ".pkl"))
            raise e
