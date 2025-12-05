import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
import time

class HandEDriverNode(Node):
    def __init__(self):
        super().__init__('hande_driver')

        # TODO: check /dev/ttyUSB*
        self.declare_parameter('device', '/dev/ttyUSB0')
        self.device = self.get_parameter('device').value

        # manual: 0x00(0) = Open (50mm), 0xFF(255) = Closed (0mm)
        self.MAX_WIDTH_M = 0.05  # 50mm
        self.MIN_WIDTH_M = 0.00  # 0mm

        self.driver = None
        try:
            self.get_logger().info(f"Connecting to Hand-E on {self.device}...")
            self.driver = RobotiqModbusRtuDriver(self.device)
            self.driver.connect()
            self.driver.reset()
            self.driver.activate()
            time.sleep(1.0) 
            self.get_logger().info("Hand-E Connected & Activated!")
        except Exception as e:
            self.get_logger().error(f"Connection Failed: {e}")
            self.driver = None

        # meter
        self.pub_state = self.create_publisher(JointState, '/hande/joint_states', 10)
        
        # Subscriber [m]
        self.sub_cmd = self.create_subscription(
            Float64MultiArray, '/hande/command', self.command_callback, 10)

        # 30Hz
        self.timer = self.create_timer(0.033, self.timer_callback)

    def byte_to_meter(self, byte_val):
        """
        [0,255] to [0.00,0.05]
        """
        ratio_closed = byte_val / 255.0
        
        return self.MAX_WIDTH_M * (1.0 - ratio_closed)

    def meter_to_byte(self, meter_val):
        """
        [0.00, 0.05] to [0, 255]
        """
        meter_val = max(self.MIN_WIDTH_M, min(self.MAX_WIDTH_M, meter_val))
        
        # the biggr, the more open
        ratio_open = meter_val / self.MAX_WIDTH_M
        
        byte_val = 255.0 * (1.0 - ratio_open)
        
        return int(byte_val)

    def timer_callback(self):
        if self.driver is None: return

        try:
            pos_raw = self.driver.get_position() 
            
            pos_meter = self.byte_to_meter(pos_raw)

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['hande_finger_joint']
            msg.position = [float(pos_meter)]
            
            self.pub_state.publish(msg)
            
        except Exception:
            pass

    def command_callback(self, msg):
        if self.driver is None: return

        try:
            cmd_meter = msg.data[0]
            
            # 0=Open, 255=Close
            target_pos_byte = self.meter_to_byte(cmd_meter)
            self.driver.move(pos=target_pos_byte, speed=128, force=10)
            
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