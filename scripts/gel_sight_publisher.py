import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class GelSightPublisher(Node):
    def __init__(self):
        super().__init__('gelsight_driver')

        # ls/dev/video*
        self.declare_parameter('device_id', 0) # 0 for default
        self.declare_parameter('width', 0) 
        self.declare_parameter('height', 0) 
        self.declare_parameter('fps', 30)

        self.dev_id = self.get_parameter('device_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.target_fps = self.get_parameter('fps').value

        self.pub = self.create_publisher(Image, '/gelsight/image_raw', 30)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(self.dev_id)

        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open video device {self.dev_id}")
            raise RuntimeError("Camera open failed")

        if self.width > 0 and self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"GelSight connected on /dev/video{self.dev_id}")
        self.get_logger().info(f"Resolution: {int(actual_w)}x{int(actual_h)}")

        period = 1.0 / self.target_fps
        self.timer = self.create_timer(period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
        
        # add timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gelsight_link"

        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GelSightPublisher()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()