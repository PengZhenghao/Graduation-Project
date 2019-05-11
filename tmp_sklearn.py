import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类

# import matplotlib.pyplot as plt  # 可视化绘图



if __name__ == '__main__':

    from utils import read_data, setup_logger, FPSTimer, Visualizer
    from lidar_map import LidarMap, lidar_config
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", '-r', action="store_true")
    parser.add_argument("--path", '-p', required=True, help="The path of target h5 file.")
    parser.add_argument("--save", '-s', action="store_true")
    args = parser.parse_args()

    assert args.path.endswith(".h5")
    args.save = args.save and (not os.path.exists(args.path.replace(".h5", ".pkl")))
    setup_logger()

    data = read_data(args.path)
    lidar_data = data['lidar_data']
    extra_data = data['extra_data']

    map = LidarMap(lidar_config)
    v = Visualizer("map_n", lidar_config["map_size"], zoom=1)
    fpstimer = FPSTimer()

    save_data = []

    # db = skc.DBSCAN(eps=1.5, min_samples=3)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法
    db = skc.DBSCAN(eps=0.5, min_samples=6)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法


    def fit(points, weight=None):
        if len(points) < 2:
            return None
        labels = db.fit_predict(points, sample_weight=weight)
        # ratio = len(labels[labels[:] == -1]) / len(labels)
        return labels


    # from lidar_map import lidar_config
    # def index_to_location(d):
    #     return d * lidar_config["grid_size"] - lidar_config["map_size"]/2
    #

    from utils import build_index_to_location

    # TODO remove lidar_config
    index_to_location = build_index_to_location(lidar_config)

    if args.render:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.show()

    for l, e in zip(lidar_data, extra_data):
        with fpstimer:
            ret = map.update(l, e)
            data = ret["points"]
            labels = fit(data, ret["weight"])
            if args.save:
                save_data.append(ret)
            if args.render:
                stop = v.draw(ret["map_n"])
                plt.clf()
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                rge, counts = np.unique(labels, return_counts=True)
                for i in range(labels.max() + 1):
                    cluster = data[labels == i]
                    plt.scatter(cluster[:, 0], cluster[:, 1], s=1)

                if stop:
                    break

    if args.save:
        import pickle

        try:
            pickle.dump(save_data, open(args.path.replace(".h5", ".pkl"), "wb"))
        except Exception as e:
            os.remove(args.path.replace(".h5", ".pkl"))
            raise e



            #
            # # if obstacles have good confidence level in local map or exist in chart,
            # # they will be marked with a very large cluster number 10000
            # fast_cluster = np.zeros((num_grid, num_grid))
            # fast_cluster[self.local_confidence_map >= confidence_min] = 10000
            # # self.local_confidence_map >= confidence_min && self.local_chart ==1 are very special!!!
            # fast_cluster[self.local_chart == 1] = 10000
            #
            # # DBSCAN density set to mindensity_fast,which is larger than mindensity_normal, to find cluster 10000
            # fast_ctest = (fast_cluster > 0) * 1  # ctest mark all place that
            # fast_cstate = fast_ctest * 1
            #
            # self.cluster_identifier = 0
            # for p in range(num_grid):
            #     for q in range(num_grid):
            #         # p, q represent a CELL in SLAM map.
            #
            #         if self.image_binary[p, q] == 1 and (
            #                 not (fast_ctest[p, q] == 1 and fast_cstate[p, q] == 0)):
            #             # if that place is believe to have cluster.
            #
            #
            #             neighbor_xmin = max(0, p - neighborsize)
            #             neighbor_xmax = min(num_grid, p + neighborsize + 1)
            #             neighbor_ymin = max(0, q - neighborsize)
            #             neighbor_ymax = min(num_grid, q + neighborsize + 1)
            #
            #
            #             neighbor_num = count_neighbours(self.image_binary, p, q)
            #
            #             # Seems to use for marking that p, q has been tested!
            #             fast_ctest[p, q] = 1
            #
            #             # Too many neighbor. Consider they all belong to a cluster.
            #             if neighbor_num >= mindensity_fast:
            #
            #                 # Mark p,q to be (?????)
            #                 fast_cstate[p, q] = 1
            #
            #                 max_clusternum = (
            #                     fast_cluster[neighbor_xmin:neighbor_xmax,
            #                     neighbor_ymin:neighbor_ymax] * fast_cstate[
            #                                                    neighbor_xmin:neighbor_xmax,
            #                                                    neighbor_ymin:neighbor_ymax]).max()  # max cluster number of fast core points in neighbor
            #                 # no fast core in  belong to existing cluster
            #                 if max_clusternum == 0:
            #                     self.cluster_identifier += 1
            #                     fast_cluster[p, q] = self.cluster_identifier
            #                     for m in range(neighbor_xmin, neighbor_xmax):
            #                         for n in range(neighbor_ymin, neighbor_ymax):
            #                             if self.image_binary[m, n] == 1:
            #                                 fast_cluster[m, n] = self.cluster_identifier
            #                                 if fast_ctest[m, n] == 0:
            #                                     mn_neighbor_num = count_neighbours(self.image_binary,
            #                                                                        m, n)
            #                                     fast_ctest[m, n] = 1
            #                                     if mn_neighbor_num >= mindensity_fast:
            #                                         fast_cstate[m, n] = 1
            #                 else:
            #
            #
            #                     fast_cluster[p, q] = max_clusternum
            #                     for m in range(neighbor_xmin, neighbor_xmax):  #
            #                         for n in range(neighbor_ymin, neighbor_ymax):  # 此时进来，是位置p，q附近的一小范围。
            #                             if self.image_binary[m, n] == 1:  # 是否有东西
            #                                 if fast_cluster[m, n] == 0:  # 附近有没人
            #                                     fast_cluster[m, n] = max_clusternum  #
            #                                     if fast_ctest[m, n] == 0:  #
            #                                         mn_neighbor_num = count_neighbours(self.image_binary, m, n)
            #                                         fast_ctest[m, n] = 1  #
            #                                         if mn_neighbor_num >= mindensity_fast:
            #                                             fast_cstate[m, n] = 1
            #                                 elif fast_cluster[m, n] != max_clusternum:  #
            #                                     if fast_cstate[m, n] == 1:  # # only core points can combine their cluster with others
            #                                         fast_cluster[fast_cluster == fast_cluster[
            #                                             m, n]] = max_clusternum
            #
            # fast_cluster[np.where(fast_cluster != 10000)] = 0
            # # find normal cluster without disturbance of cluster 10000
            # self.image_binary[np.where(fast_cluster == 10000)] = 0
            #
            # # DBSCAN density set to mindensity_normal to find normal cluster
            # normal_cluster = np.zeros((num_grid, num_grid))
            # normal_ctest = np.zeros((num_grid, num_grid))
            # normal_cstate = np.zeros((num_grid, num_grid))
            # self.cluster_identifier = 0
            # for p in range(num_grid):
            #     for q in range(num_grid):
            #         if self.image_binary[p, q] == 1 and (
            #                 not (normal_ctest[p, q] == 1 and normal_cstate[p, q] == 0)):
            #             neighbor_xmin = max(0, p - neighborsize)
            #             neighbor_xmax = min(num_grid, p + neighborsize + 1)
            #             neighbor_ymin = max(0, q - neighborsize)
            #             neighbor_ymax = min(num_grid, q + neighborsize + 1)
            #             neighbor_binary = self.image_binary[neighbor_xmin:neighbor_xmax,
            #                               neighbor_ymin:neighbor_ymax]
            #             neighbor_num = neighbor_binary.sum()
            #             normal_ctest[p, q] = 1
            #             if neighbor_num >= mindensity_normal:
            #                 normal_cstate[p, q] = 1
            #                 max_clusternum = (
            #                     normal_cluster[neighbor_xmin:neighbor_xmax,
            #                     neighbor_ymin:neighbor_ymax] * normal_cstate[
            #                                                    neighbor_xmin:neighbor_xmax,
            #                                                    neighbor_ymin:neighbor_ymax]).max()  # max cluster number in neighbor
            #                 # no neighbor belong to existing cluster
            #                 if max_clusternum == 0:
            #                     self.cluster_identifier += 1
            #                     normal_cluster[p, q] = self.cluster_identifier
            #                     for m in range(neighbor_xmin, neighbor_xmax):
            #                         for n in range(neighbor_ymin, neighbor_ymax):
            #                             if self.image_binary[m, n] == 1:
            #                                 normal_cluster[m, n] = self.cluster_identifier
            #                                 if normal_ctest[m, n] == 0:
            #                                     mn_neighbor_num = count_neighbours(self.image_binary, m,
            #                                                                        n)
            #                                     normal_ctest[m, n] = 1
            #                                     if mn_neighbor_num >= mindensity_normal:
            #                                         normal_cstate[m, n] = 1
            #                 else:
            #                     normal_cluster[p, q] = max_clusternum
            #                     for m in range(neighbor_xmin, neighbor_xmax):
            #                         for n in range(neighbor_ymin, neighbor_ymax):
            #                             if self.image_binary[m, n] == 1:
            #                                 if normal_cluster[m, n] == 0:
            #                                     normal_cluster[m, n] = max_clusternum
            #                                     if normal_ctest[m, n] == 0:
            #                                         mn_neighbor_num = count_neighbours(
            #                                             self.image_binary, m, n)
            #                                         normal_ctest[m, n] = 1
            #                                         if mn_neighbor_num >= mindensity_normal:
            #                                             normal_cstate[m, n] = 1
            #                                 elif normal_cluster[
            #                                     m, n] != max_clusternum:  # only core points can combine their cluster with others
            #                                     if normal_cstate[m, n] == 1:
            #                                         normal_cluster[
            #                                             np.where(normal_cluster == normal_cluster[
            #                                                 m, n])] = max_clusternum
            # self._cluster_map = fast_cluster + normal_cluster
            # # remove noise
            # self.image_binary[np.where(self._cluster_map == 0)] = 0


            # def analyze_cluster(, posx, posy):
            #
            #     ### What is the format of self._cluster_map?
            #     # Its shape is [250, 250]
            #     # self.cluster_identifier is a scalar!
            #
            #
            #     self.property_cluster = np.zeros((0, 10))
            #     for k in range(1, self.cluster_identifier + 1):
            #         location = np.where(self._cluster_map == k)
            #         area_cluster = len(location[0])
            #         if area_cluster < minarea_cluster:  # some clusters were absorbed by others, some are too small to be an object
            #             self._cluster_map[location] = 0
            #             self.cluster_identifier -= 1
            #             self.image_binary[location] = 1
            #         else:  # calculate basic properties of each cluster
            #             h_cluster = self.image_high[location].max()
            #             low_cluster = self.image_low[location].min()
            #             coor_cluster = np.column_stack((location[0], location[1]))
            #             pca_cluster, heading_cluster = pca(coor_cluster)
            #             l_cluster = (pca_cluster[:, 1].max() - pca_cluster[:, 1].min()) * grid_size
            #             b_cluster = (pca_cluster[:, 0].max() - pca_cluster[:, 0].min()) * grid_size
            #             if l_cluster > maxlen_obj:
            #                 self._cluster_map[location] = 0
            #                 self.cluster_identifier -= 1
            #                 self.image_binary[location] = ceil(confidence_min / 4)
            #             elif low_cluster > maxlow_obj:
            #                 self._cluster_map[location] = 0
            #                 self.cluster_identifier -= 1
            #                 self.image_binary[location] = 0
            #             else:
            #                 self.property_cluster = np.row_stack((self.property_cluster, np.array(
            #                     [area_cluster, # area
            #                      (location[0].min() - num_grid / 2) * grid_size + posx, # x min
            #                      (location[0].max() - num_grid / 2) * grid_size + posx, # x max
            #                      (location[1].min() - num_grid / 2) * grid_size + posy, # y min
            #                      (location[1].max() - num_grid / 2) * grid_size + posy, # y max
            #                      l_cluster, # l
            #                      b_cluster, # b
            #                      heading_cluster, # heading
            #                      h_cluster,  # h
            #                      k #
            #                      ])))
