import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilities.gelsightmini import GelSightMini, Camera

class GelSightROSData:
    def __init__(self):
        rospy.init_node('gelsight_driver', anonymous=True)

        self.device_id = rospy.get_param('~device_id', 0)
        self.topic_name = rospy.get_param('~topic_name', 'image_raw')
        
        # Image Config (GelSight Mini Default)
        self.width = 640  
        self.height = 480
        self.border_frac = 0.15 

        rospy.loginfo(f"Initializing GelSight Mini on device {self.device_id}...")
        
        self.cam = GelSightMini(
            target_width=self.width,
            target_height=self.height,
            border_fraction=self.border_frac
        )


        self.cam.camera = Camera(self.device_id)

        # Setup ROS
        self.pub = rospy.Publisher(self.topic_name, Image, queue_size=1)
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30) # 30Hz

    def start_camera(self):
        try:
            self.cam.camera.open()
            
            self.cam.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam.target_width)
            self.cam.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam.target_height)
            
            real_w = self.cam.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            real_h = self.cam.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            rospy.loginfo(f"Camera opened: {real_w}x{real_h}")
            
            self.cam.recording = False
            self.cam.frame_count = 0
            
            return True
        except Exception as e:
            rospy.logerr(f"Failed to open GelSight on {self.device_id}: {e}")
            return False

    def run(self):
        if not self.start_camera():
            return

        rospy.loginfo(f"Publishing to {rospy.get_namespace()}{self.topic_name}")

        while not rospy.is_shutdown():
            try:
                # SDKの update() を呼ぶ
                # 内部で crop_and_resize が行われ、RGB画像が返ってきます
                img = self.cam.update(1.0) 

                if img is not None and img.shape[0] > 0:
                    # SDKは RGB で返すので encoding="rgb8"
                    msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "gelsight_link"
                    
                    self.pub.publish(msg)
                
            except Exception as e:
                # カメラが抜けた場合などのエラーハンドリング
                rospy.logwarn_throttle(1, f"Capture error: {e}")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = GelSightROSData()
        node.run()
    except rospy.ROSInterruptException:
        pass