#!/usr/bin/env python3

import rospy

import cv2
import json
import sys
import os
import numpy as np
import torch

from send_goals.msg import detect
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf import TransformListener

# import some common detectron2 utilities
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import read_image

# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建出 centernet 包所在的路径

centernet_path = os.path.join(current_dir, 'third_party', 'CenterNet2')

# 将 centernet 包的路径添加到 Python 搜索路径中
sys.path.append(centernet_path)

from centernet.config import add_centernet_config

from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.predictor import AsyncPredictor

BUILDIN_CLASSIFIER = {
    'lvis': '/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/Deticmain/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}


class myVisualizationDemo(object):
    def __init__(self, cfg,
                 instance_mode=ColorMode.IMAGE, parallel=False):

        self.metadata = MetadataCatalog.get(
            BUILDIN_METADATA_PATH['lvis'])
        classifier = BUILDIN_CLASSIFIER['lvis']

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        vis_output = None
        predictions = self.predictor(image)

        detections = []

        # 将类别ID映射到类别名称的字典
        class_names = self.metadata.get("thing_classes", None)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

            for i in range(len(instances)):
                # 获取检测框的对角坐标
                bbox = instances.pred_boxes.tensor[i].tolist()

                # 获取类别ID
                class_id = instances.pred_classes[i].item()

                # 获取类别名称
                class_name = class_names[class_id] if class_names else str(class_id)

                # 获取置信度
                confidence = instances.scores[i].item()

                # 添加到检测结果列表中
                detections.append({
                    "bbox": bbox,
                    "class_name": class_name,
                    "confidence": confidence
                })

        # 可视化图像输出
        image = image[:, :, ::-1]  # Convert image from OpenCV BGR format to Matplotlib RGB format
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            elif "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return detections, vis_output


class RGBDepthSubscriber:
    def __init__(self):

        rospy.init_node('rgb_subscriber', anonymous=True)
        print(torch.cuda.is_available())
        self.cv_bridge = CvBridge()
        # ctypes.CDLL("libX11.so.6").XInitThreads()

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.label = ""
        self.conf = 0
        self.depth_image = None
        self.depth_image_process = None
        self.fx = 554.254691
        self.fy = 554.254691
        self.cx = 320.5
        self.cy = 240.5
        self.u = 320
        self.v = 240
        self.x = 0
        self.y = 0
        self.z = 0

        self.current_time = rospy.Time.now()

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)

        cfg.merge_from_file(
            "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/Deticmain/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = "/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/Deticmain/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

        cfg.MODEL.PANOPTIC_FPN.ENABLED = False
        cfg.MODEL.SEM_SEG_HEAD.ENABLED = False

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (False)
        cfg.freeze()
        self.demo = myVisualizationDemo(cfg)

        self.velocity_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        self.is_stopped = True

        self.rgb_subscriber = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback, queue_size=1,
                                               buff_size=52428800)
        self.depth_subscriber = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.listener = TransformListener()
        self.world_point_pub = rospy.Publisher('/yolo_point', PointStamped, queue_size=10)
        self.yolo_info_pub = rospy.Publisher('/yolo_info', detect, queue_size=10)

        self.pub1 = rospy.Publisher('/view1', Image, queue_size=10)
        self.pub2 = rospy.Publisher('/view2', Image, queue_size=10)

    def velocity_callback(self, msg):
        # Check if the linear and angular velocities are both zero
        if msg.linear.x == 0.0 and msg.angular.z == 0.0:
            self.is_stopped = True
        else:
            self.is_stopped = False

    def depth_callback(self, depth_data):
        if self.is_stopped:
            try:
                self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_data)
                # print("depth_image_shape")
                # print(self.depth_image.shape)
                normalized_img = cv2.normalize(self.depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_8U)
                cv2.circle(normalized_img, (int(self.u), int(self.v)), 2, (255, 255, 255), 2)
                cv2.imshow("Depth Image", normalized_img)
                cv2.waitKey(1)  # This is necessary for imshow to work properly
            except Exception as e:
                rospy.logerr("depth image: %s", str(e))
        else:
            pass

    def rgb_callback(self, rgb_data):
        if self.is_stopped:
            try:
                rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
                color_image = rgb_image
                self.depth_image_process = self.depth_image
                self.current_time = rospy.Time.now()
                detections, vis_output = self.demo.run_on_image(color_image)
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    self.x1 = int(x1)
                    self.y1 = int(y1)
                    self.x2 = int(x2)
                    self.y2 = int(y2)
                    self.label = detection['class_name']
                    self.conf = detection['confidence']

                    (self.u, self.v) = (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2, ))

                    ispub = self.calculate()
                    print("ispub:")
                    print(ispub)
                    if ispub:
                        roi = color_image[self.y1:self.y2, self.x1:self.x2]
                        roi_imgsmg = self.cv_bridge.cv2_to_imgmsg(roi, encoding="bgr8")
                        detect_msg = detect()
                        detect_msg.x = self.x
                        detect_msg.y = self.y
                        detect_msg.z = self.z
                        detect_msg.x1 = self.x1
                        detect_msg.y1 = self.y1
                        detect_msg.x2 = self.x2
                        detect_msg.y2 = self.y2
                        detect_msg.roi_image = roi_imgsmg
                        detect_msg.conf = self.conf
                        detect_msg.label = self.label
                        self.yolo_info_pub.publish(detect_msg)

                        ros_image1 = self.cv_bridge.cv2_to_imgmsg(vis_output.get_image()[:, :, ::-1], encoding="bgr8")
                        ros_image2 = self.cv_bridge.cv2_to_imgmsg(roi, encoding="bgr8")
                        self.pub1.publish(ros_image1)
                        self.pub2.publish(ros_image2)
                    else:
                        ros_image3 = self.cv_bridge.cv2_to_imgmsg(vis_output.get_image()[:, :, ::-1], encoding="bgr8")
                        self.pub1.publish(ros_image3)

            except Exception as e:
                rospy.logerr("RGB&DETIC: %s", str(e))
        else:
            pass

    def calculate(self):
        try:
            if self.depth_image_process is not None and self.u != 0 and self.v != 0:
                depth_value = self.depth_image_process[self.v, self.u]
                print("depth_value:")
                print(depth_value)
                if depth_value > 0 and depth_value < 10:
                    camera_z = depth_value
                    camera_x = (self.u - self.cx) / self.fx * depth_value
                    camera_y = (self.v - self.cy) / self.fy * depth_value
                    print("Depth value: {:.2f} m".format(depth_value))

                    camera_point_msg = PointStamped()
                    camera_point_msg.header.stamp = self.current_time
                    camera_point_msg.header.frame_id = "camera_frame_optical"  # 假设相机坐标系为 "camera_frame"
                    camera_point_msg.point.x = camera_x
                    camera_point_msg.point.y = camera_y
                    camera_point_msg.point.z = camera_z
                    try:
                        # 使用 TransformListener 获取相机坐标系到世界坐标系的变换
                        self.listener.waitForTransform("map", "camera_frame_optical", self.current_time,
                                                       rospy.Duration(1.0))
                        world_point_msg = self.listener.transformPoint("map", camera_point_msg)
                        self.x = world_point_msg.point.x
                        self.y = world_point_msg.point.y
                        self.z = world_point_msg.point.z
                        # 发布世界坐标系中的sl点
                        self.world_point_pub.publish(world_point_msg)
                        return True
                    except Exception as e:
                        rospy.logwarn("Failed to transform point: %s", str(e))
                        return False
        except Exception as e:
            rospy.logerr("calculate: %s", str(e))
            return False


if __name__ == '__main__':
    rgb_depth_subscriber = RGBDepthSubscriber()
    rospy.spin()
