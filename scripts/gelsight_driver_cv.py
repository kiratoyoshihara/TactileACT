#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GelSightNode:
    def __init__(self):
        rospy.init_node('gelsight_driver', anonymous=True)
        self.device_path = rospy.get_param('~device_path', '/dev/video0')
        self.topic_name = rospy.get_param('~topic_name', 'gelsight/image_raw')

        self.resize_width = rospy.get_param('~width', 640)
        self.resize_height = rospy.get_param('~height', 480)

        self.cap = cv2.VideoCapture(self.device_path)

        # set res to native one
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2464)

        # MJPEG 
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        real_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        rospy.loginfo(f"GelSight Connected: {self.device_path}")
        rospy.loginfo(f"Native Resolution: {real_w}x{real_h} (Outputting: {self.resize_width}x{self.resize_height})")

        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.topic_name, Image, queue_size=1)
        self.rate = rospy.Rate(30)

    def run(self):

        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if ret:
                resized_frame = cv2.resize(frame, (self.resize_width, self.resize_height))

                msg = self.bridge.cv2_to_imgmsg(resized_frame, encoding="bgr8")
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "gelsight_link"

                self.pub.publish(msg)

            else:
                rospy.logwarn("Failed to capture frame from GelSight")

            self.rate.sleep()

        self.cap.release()

if __name__ == '__main__':
    try:
        node = GelSightNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
