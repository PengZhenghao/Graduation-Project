from recorder import Recorder

config = {"exp_name": "0419-ob", "save_dir": "experiments"}
r = Recorder(config)

data = r.read()
lidar_data = data["lidar_data"][:]
# frames = data["frame"][:]
extra_data = data["extra_data"][:]
print("lidar_data contains {} and its shape is {}.".format(lidar_data, lidar_data.shape))
print("frame means for each datapoint {}.".format(lidar_data.mean(1)))
m = lidar_data.mean(1)
print(m)
