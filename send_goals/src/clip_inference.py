#!/usr/bin/env python3

import rospy
import csv
import torch
import numpy as np
import clip
from std_msgs.msg import String
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import math

from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from visualization_msgs.msg import Marker
from nav_msgs.srv import GetMap

filename = "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/data_clip.csv"
pgm_file = '/home/ubuntu/Desktop/catkin_turtlebot3/src/turtlebot3_sim_test/maps/map2.pgm'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


class TextSubscriber:
    def __init__(self):
        rospy.init_node('text_subscriber', anonymous=True)
        rospy.Subscriber("/text_topic", String, self.text_callback)

        self.dict_list = self.read_csv_to_dict(filename)

        width, height, pgm_data = self.read_pgm(pgm_file)
        rotated_pgm = np.flipud(pgm_data)
        self.pgm = rotated_pgm

        self.map_service = rospy.ServiceProxy('/static_map', GetMap)
        self.map_data = None
        self.resolution = None
        self.origin = None

        rospy.wait_for_service('/static_map')
        try:
            map_resp = self.map_service()
            self.map_data = np.array(map_resp.map.data).reshape((map_resp.map.info.height, map_resp.map.info.width))
            self.resolution = map_resp.map.info.resolution
            self.origin = map_resp.map.info.origin

            print(self.resolution, self.origin)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

        # 发布目标点用于rviz显示
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

        rospy.loginfo("ready for command")

    def text_callback(self, Command):
        # Clip 编码文本
        rospy.loginfo("Clip Text: %s", Command)
        self.search_target_item(Command.data)
        # print("Label probs:", probs)

    def search_target_item(self, command):
        text = clip.tokenize([command]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        features = []
        index = []
        label = []
        x = []
        y = []
        z = []
        for data_dict in self.dict_list:
            features.append([float(num_str) for num_str in data_dict["features"].split(',')])
            index.append(data_dict["index"])
            label.append(data_dict["label"])
            x.append(data_dict["x"])
            y.append(data_dict["y"])
            z.append(data_dict["z"])

        image_features = torch.tensor(features, dtype=torch.float16).to(device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        print("Shape of tensor:", text_features.cpu().shape, image_features.T.cpu().shape)
        similarity = (100.0 * text_features @ image_features.T).cpu().float().tolist()
        # for i, value in enumerate(similarity[0]):
        #     print("index:", i, "sim:", value, "actually:", label[i])
        # max_similarity = max(similarity[0])
        # max_index = similarity[0].index(max_similarity)
        # print("max:", max_similarity, "result:", label[max_index])

        plt.subplot(1, 3, 1)

        x = [int((i - self.origin.position.x) / self.resolution) for i in x]
        y = [int((i - self.origin.position.y) / self.resolution) for i in y]
        plt.axis('off')  # 关闭坐标轴

        plt.imshow(self.pgm, cmap='gray')

        plt.scatter(x, y, c=similarity[0], cmap='coolwarm')

        # 根据 sim 列表中的元素排序并对齐其他四个列表
        sorted_lists = sorted(zip(similarity[0], index, x, y, z))

        # 解压缩排序后的列表
        sorted_sim, sorted_index, sorted_x, sorted_y, sorted_z = map(list, zip(*sorted_lists))

        # 创建一个新的图形
        plt.subplot(1, 3, 2)

        # 绘制折线图
        plt.plot(range(len(sorted_sim)), sorted_sim, marker='o', linestyle='-')

        plt.xlabel('i')
        plt.ylabel('Similarity')

        # 计算前 % 的元素个数
        percent = 0.03
        num_elements = int(len(sorted_sim) * percent)

        # 取出前 20% 的元素
        sorted_sim_top = sorted_sim[-num_elements:]
        sorted_x_top = sorted_x[-num_elements:]
        sorted_y_top = sorted_y[-num_elements:]

        # indexes = [i for i, x in enumerate(sorted_sim) if x > 28]
        # sorted_sim_top = [sorted_sim[i] for i in indexes]
        # sorted_x_top = [sorted_x[i] for i in indexes]
        # sorted_y_top = [sorted_y[i] for i in indexes]

        plt.subplot(1, 3, 3)
        plt.imshow(self.pgm, cmap='gray')
        plt.scatter(sorted_x_top, sorted_y_top, c=sorted_sim_top, cmap='coolwarm')
        plt.axis('off')
        plt.show()

        cluster_centers, target = self.dbscan_clustering(sorted_x_top, sorted_y_top, sorted_sim_top, eps=10,
                                                         min_samples=5)
        cluster_centers = np.array(cluster_centers)

        if cluster_centers.any():
            plt.imshow(self.pgm, cmap='gray')
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100)

            plt.show()

        else:
            rospy.loginfo("can not find center")

        radius = 15

        threshold = 15

        loc, qua = self.generate_quaternion_on_circle(target, radius, threshold)

        print(loc, qua)

        # loc = [(x / self.scale - self.bias) for x in loc]
        loc[0] = loc[0] * self.resolution + self.origin.position.x
        loc[1] = loc[1] * self.resolution + self.origin.position.y

        self.creat_marker(loc)

        self.send_goal(loc, qua)

        rospy.loginfo("ready for another command")

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    def dbscan_clustering(self, x, y, sim, eps, min_samples):

        xy_val_dict = {(x[i], y[i]): sim[i] for i in range(len(x))}

        X = np.column_stack((x, y))

        # DBSCAN 聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(np.column_stack((x, y)))

        # 获取聚类结果
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # 获取聚类种类数（不包括噪声点）
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # 输出聚类种类数
        print("Number of clusters:", n_clusters_)

        # 输出离群点
        outliers = [(x[i], y[i]) for i in range(len(labels)) if labels[i] == -1]
        print("Outliers:", outliers)

        # 计算聚类的中心坐标和相似度平均值
        cluster_centers = []
        cluster_average_sim = []
        for cluster_label in range(n_clusters_):
            cluster_indices = [i for i in range(len(labels)) if labels[i] == cluster_label]
            cluster_points = [(x[i], y[i]) for i in cluster_indices]

            # 计算相似度平均值
            sim_sum = 0
            for cluster_point in cluster_points:
                sim_sum = sim_sum + xy_val_dict[(cluster_point[0], cluster_point[1])]
            average_sim = sim_sum / len(cluster_points)

            cluster_average_sim.append(average_sim)
            cluster_centers.append(np.mean(cluster_points, axis=0))

        # 对 cluster_average_sim 列表进行排序
        sorted_indices = np.argsort(cluster_average_sim)[::-1]
        sorted_cluster_centers = [cluster_centers[i] for i in sorted_indices]
        sorted_cluster_average_sim = [cluster_average_sim[i] for i in sorted_indices]

        # 绘制结果
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        plt.imshow(self.pgm, cmap='gray')

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0.5, 0.5, 0.5, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='r', markersize=6)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='r', markersize=6)

        plt.title('Number of clusters: %d' % n_clusters_)
        plt.axis('off')
        plt.show()

        print(sorted_cluster_centers, sorted_cluster_average_sim)

        return sorted_cluster_centers, sorted_cluster_centers[0]

    def read_csv_to_dict(self, filename):
        dict_list = []
        data_dict = {}  # 创建一个空字典用于存储CSV数据

        try:
            with open(filename, 'r') as csv_file:  # 打开CSV文件进行读取
                csv_reader = csv.DictReader(csv_file)  # 创建CSV字典读取器

                for row in csv_reader:  # 逐行读取CSV文件
                    for key, value in row.items():  # 遍历每行的键值对
                        if key == "index":
                            data_dict[key] = int(value)  # 将键值对添加到字典中
                        elif key == "x" or key == "y" or key == "z" or key == "conf":
                            data_dict[key] = float(value)  # 将键值对添加到字典中
                        elif key == "label" or key == "features":
                            data_dict[key] = value  # 将键值对添加到字典中
                    dict_list.append(data_dict)
                    data_dict = {}

        except IOError:
            rospy.logerr("无法打开文件: %s", filename)

        return dict_list
        # 读取PGM文件

    def read_pgm(self, filename):
        with open(filename, 'rb') as f:

            f.readline()
            # 跳过注释行
            for line in f:
                if not line.startswith(b'#'):
                    break

            width, height = map(int, line.split())

            f.readline()

            # 读取图像数据
            image_data = np.fromfile(f, dtype=np.uint8)

        return width, height, image_data.reshape((height, width))

    def send_goal(self, loc, qua):
        # 创建move_base客户端
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()

        # 设置目标位置和姿态信息
        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.pose.position.x = loc[0]  # 目标位置x坐标
        target_pose.pose.position.y = loc[1]  # 目标位置y坐标
        # target_pose.pose.position.z = loc[2]
        target_pose.pose.orientation.x = qua[0]
        target_pose.pose.orientation.y = qua[1]
        target_pose.pose.orientation.z = qua[2]
        target_pose.pose.orientation.w = qua[3]

        # 创建目标位姿
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = target_pose.pose

        # 发送目标位置和姿态信息
        client.send_goal(goal)
        client.wait_for_result()

    def creat_marker(self, loc):
        marker = Marker()  # 创建目标点
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = loc[0]  # 目标位置x坐标
        marker.pose.position.y = loc[1]  # 目标位置y坐标
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        # marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # 发布目标点
        self.marker_pub.publish(marker)
        rospy.loginfo("发布目标点完毕")

    def generate_quaternion_on_circle(self, target_point, radius, threshold):

        i = 0
        print("generate_quaternion_on_circle")
        while (True):
            # Generate a random angle
            angle = np.random.uniform(0, 2 * np.pi)

            # Calculate random point on the circle
            x = target_point[0] + radius * np.cos(angle)
            y = target_point[1] + radius * np.sin(angle)

            # Calculate vector pointing towards target point
            vec_to_target = np.array(target_point) - np.array([x, y])

            # Normalize vector
            vec_to_target /= np.linalg.norm(vec_to_target)

            # Create rotation matrix to align with target vector
            rot_matrix = np.eye(3)
            rot_matrix[:2, :2] = np.array([[vec_to_target[0], -vec_to_target[1]],
                                           [vec_to_target[1], vec_to_target[0]]])

            # Convert rotation matrix to quaternion
            quaternion = Rotation.from_matrix(rot_matrix).as_quat()

            if self.is_min_distance_to_gray_area_above_threshold(x, y, threshold):
                self.plot_circle_and_target(target_point, radius, [x, y], threshold)
                break

            i = i + 1
            print(i)

            if i == 20:
                radius = int(radius * 1.5)
                print(radius)
                i = 0

        return [x, y, 0], quaternion

    def is_min_distance_to_gray_area_above_threshold(self, x, y, threshold):
        if self.map_data is None:
            rospy.logerr("Map data not available.")
            return False

        map_x = int(x)
        map_y = int(y)

        # 检查目标点是否在地图范围内
        if not (0 <= map_x < self.map_data.shape[1] and 0 <= map_y < self.map_data.shape[0]):
            rospy.logerr("Target point is out of map bounds.")
            return False

        min_distance = float('inf')  # 初始化最小距离为正无穷

        # 遍历地图上的每个单元格
        for j in range(self.map_data.shape[0]):
            for i in range(self.map_data.shape[1]):
                # 检查单元格是否为灰色区域
                if self.map_data[j, i] == -1:
                    # 计算目标点与当前灰色区域单元格的距离
                    distance = math.sqrt((i - map_x) ** 2 + (j - map_y) ** 2)
                    # 更新最小距离
                    min_distance = min(min_distance, distance)

        # 检查最小距离是否大于阈值
        if min_distance > threshold:
            return True
        else:
            return False

    def plot_circle_and_target(self, target_point, radius, position, threshold):
        plt.imshow(self.pgm, cmap='gray')
        circle1 = plt.Circle(target_point, radius, color='blue', fill=False)
        plt.gca().add_patch(circle1)
        plt.plot(target_point[0], target_point[1], 'ro', label='Target Point')
        plt.plot(position[0], position[1], 'go', label='Random Point on Circle')
        circle = plt.Circle(position, threshold, color='blue', fill=False)
        plt.gca().add_patch(circle)
        plt.quiver(position[0], position[1], target_point[0] - position[0], target_point[1] - position[1], angles='xy',
                   scale_units='xy', scale=1, color='green', label='Direction to Target')
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Random Point on Circle and Direction to Target')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    textsubscriber = TextSubscriber()
    rospy.spin()
