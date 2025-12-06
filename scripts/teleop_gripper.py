import rospy
from std_msgs.msg import Float64MultiArray
import sys, termios, tty, select

class GripperTeleop:
    def __init__(self):
        rospy.init_node('gripper_teleop')
        self.pub_cmd = rospy.Publisher('/hande/command', Float64MultiArray, queue_size=10)
        self.is_open = True
        
        print("\n" + "="*50)
        print("  Gripper Teleop for Data Collection (ROS 1)")
        print("  [SPACE] : Toggle Open/Close")
        print("  [Ctrl+C]: Quit")
        print("="*50 + "\n")
        
        self.rate = rospy.Rate(10) # 10Hz check

    def get_key(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def run(self):
        while not rospy.is_shutdown():
            key = self.get_key()
            if key == ' ':
                self.toggle_gripper()
            elif key == '\x03': # Ctrl+C
                break
            self.rate.sleep()

    def toggle_gripper(self):
        msg = Float64MultiArray()
        if self.is_open:
            # Close (0.00m)
            target = 0.00
            print("Closing Gripper")
            self.is_open = False
        else:
            # Open (0.05m)
            target = 0.05
            print("Opening Gripper")
            self.is_open = True
            
        msg.data = [target]
        self.pub_cmd.publish(msg)

if __name__ == '__main__':
    try:
        node = GripperTeleop()
        node.run()
    except rospy.ROSInterruptException:
        pass