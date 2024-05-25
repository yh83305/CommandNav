#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

def publish_point():
    rospy.init_node('point_publisher', anonymous=True)
    pub = rospy.Publisher('/point_stamped', PointStamped, queue_size=10)
    rate = rospy.Rate(1)  # 发布频率设置为1Hz

    while not rospy.is_shutdown():
        point = PointStamped()
        point.header = Header(stamp=rospy.Time.now(), frame_id="map")
        point.point.x = 1.0  # 设置点的x坐标
        point.point.y = 2.0  # 设置点的y坐标
        point.point.z = 3.0  # 设置点的z坐标

        pub.publish(point)  # 发布PointStamped消息
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_point()
    except rospy.ROSInterruptException:
        pass

