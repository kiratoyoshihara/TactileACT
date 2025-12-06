import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
import time

class HandEDriverNode:
    def __init__(self):
        rospy.init_node('hande_driver', anonymous=True)

        # TODO: check ls /dev/ttyUSB*
        self.device = rospy.get_param('~device', '/dev/ttyUSB0')

        self.MAX_WIDTH_M = 0.05
        self.MIN_WIDTH_M = 0.00
        
        self.driver = None
        try:
            rospy.loginfo(f"Connecting to Hand-E on {self.device}...")
            self.driver = RobotiqModbusRtuDriver(self.device)
            self.driver.connect()
            self.driver.reset()
            self.driver.activate()
            time.sleep(1.0) 
            rospy.loginfo("Hand-E Connected & Activated!")
        except Exception as e:
            rospy.logerr(f"Connection Failed: {e}")
            self.driver = None

        self.pub_state = rospy.Publisher('/hande/joint_states', JointState, queue_size=10)
        rospy.Subscriber('/hande/command', Float64MultiArray, self.command_callback)

        self.rate = rospy.Rate(30) # 30Hz

    def byte_to_meter(self, byte_val):
        ratio_closed = byte_val / 255.0
        return self.MAX_WIDTH_M * (1.0 - ratio_closed)

    def meter_to_byte(self, meter_val):
        meter_val = max(self.MIN_WIDTH_M, min(self.MAX_WIDTH_M, meter_val))
        ratio_open = meter_val / self.MAX_WIDTH_M
        byte_val = 255.0 * (1.0 - ratio_open)
        return int(byte_val)

    def loop(self):
        while not rospy.is_shutdown():
            if self.driver:
                try:
                    pos_raw = self.driver.get_position() 
                    pos_meter = self.byte_to_meter(pos_raw)

                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.name = ['hande_finger_joint']
                    msg.position = [float(pos_meter)]
                    
                    self.pub_state.publish(msg)
                except Exception:
                    pass
            self.rate.sleep()

    def command_callback(self, msg):
        if self.driver is None: return
        try:
            cmd_meter = msg.data[0]
            target_pos_byte = self.meter_to_byte(cmd_meter)
            self.driver.move(pos=target_pos_byte, speed=128, force=10)
        except Exception as e:
            rospy.logerr(f"Command Error: {e}")

if __name__ == '__main__':
    try:
        node = HandEDriverNode()
        node.loop()
    except rospy.ROSInterruptException:
        pass