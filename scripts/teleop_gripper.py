import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import termios
import tty
import select

class GripperTeleop(Node):
    def __init__(self):
        super().__init__('gripper_teleop')
        
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, '/hande/command', 10)
        
        self.is_open = True
        self.print_usage()
        
        # check every 0.1 sec
        self.timer = self.create_timer(0.1, self.check_key)

    def print_usage(self):
        print("\n" + "="*50)
        print("  Gripper Teleop for Data Collection")
        print("  [SPACE] : Toggle Open/Close")
        print("  [Ctrl+C]: Quit")
        print("="*50 + "\n")

    def get_key(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def check_key(self):
        key = self.get_key()
        
        if key == ' ':  # Space key
            self.toggle_gripper()
        elif key == '\x03': # Ctrl+C
            rclpy.shutdown()

    def toggle_gripper(self):
        msg = Float64MultiArray()
        
        if self.is_open:
            target = 0.00
            print("Closing Gripper (0.00m)")
            self.is_open = False
        else:
            target = 0.05
            print("Opening Gripper (0.05m)")
            self.is_open = True
            
        msg.data = [target]
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GripperTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()