import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
import time

class HandEDriverNode(Node):
    def __init__(self):
        super().__init__('hande_driver')

        # TODO: ls /dev/ttyUSB*
        self.declare_parameter('device', '/dev/ttyUSB0')
        self.device = self.get_parameter('device').value

        self.driver = None
        try:
            self.get_logger().info(f"Connecting to Hand-E on {self.device}...")
            self.driver = RobotiqModbusRtuDriver(self.device)
            self.driver.connect()
            self.driver.reset()
            self.driver.activate()
            time.sleep(1.0) # wait a sec
            self.get_logger().info("Hand-E Connected & Activated!")
        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            return

        # Publisher
        self.pub_state = self.create_publisher(JointState, '/hande/joint_states', 10)

        # Subscriber 
        self.sub_cmd = self.create_subscription(
            Float64MultiArray, '/hande/command', self.command_callback, 10)

        # 30Hz
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        if self.driver is None: return

        try:
            pos_raw = self.driver.get_position() 
            # TODO: chech the following map 0 (close) -> 0.0 [m], 255 (open) -> 0.05 [m]
            pos_meter = (pos_raw / 255.0) * 0.05

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['hande_finger_joint']
            msg.position = [float(pos_meter)]
            
            self.pub_state.publish(msg)
            
        except Exception as e:
            pass

    def command_callback(self, msg):
        if self.driver is None: return

        try:
            cmd_val = msg.data[0]
            
            # Assume 0.0 (open) 1.0 (close)
            target_pos = int(cmd_val * 255.0)
            target_pos = max(0, min(255, target_pos))

            self.driver.move(pos=target_pos, speed=100, force=10)
            
        except Exception as e:
            self.get_logger().error(f"Command Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HandEDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()