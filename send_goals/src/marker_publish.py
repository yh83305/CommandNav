#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from send_goals.msg import detect
from visualization_msgs.msg import Marker, MarkerArray


class RGBDepthSubscriber:
    def __init__(self):
        rospy.init_node('marker_publisher', anonymous=True)
        self.cv_bridge = CvBridge()
        self.marker_pub = rospy.Publisher('/yolo_marker', MarkerArray, queue_size=10)
        self.detect_subscriber = rospy.Subscriber("/yolo_info", detect, self.yolo_info_callback)
        self.marker_array = MarkerArray()
        self.label = "p"
        self.i = 0

    def yolo_info_callback(self, msg):
        self.marker_append(msg)
        self.marker_pub.publish(self.marker_array)

    def marker_append(self, msg):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "basic_shapes"
        marker.id = self.i
        self.i = self.i + 1
        if self.i >= 100:
            self.i = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = msg.x
        marker.pose.position.y = msg.y
        marker.pose.position.z = msg.z
        marker.text = msg.label
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.a = msg.conf
        self.marker_array.markers.append(marker)


if __name__ == '__main__':
    rgb_depth_subscriber = RGBDepthSubscriber()
    rospy.spin()
