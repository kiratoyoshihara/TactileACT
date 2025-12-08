import rospy
from lbr_fri_msgs.msg import LBRState, LBRPositionCommand
from sensor_msgs.msg import JointState

class KukaTeachingMode:
    def __init__(self):
        rospy.init_node('kuka_teaching_mode')

        self.curr_pos = None

        self.sub = rospy.Subscriber('/lbr_state', LBRState, self.callback)
        
        # Publisher
        self.pub = rospy.Publisher('/lbr_command', LBRPositionCommand, queue_size=1)

        rospy.loginfo("Ghost Mode Started: The robot will follow your hand.")
    
        self.rate = rospy.Rate(100) 

    def callback(self, msg):
        # LBRState
        self.curr_pos = msg.measured_joint_position
        # JointState
        # self.curr_pos = msg.position

    def run(self):
        while not rospy.is_shutdown():
            if self.curr_pos is not None:
                cmd = LBRPositionCommand()
                cmd.joint_position = self.curr_pos
                self.pub.publish(cmd)
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = KukaTeachingMode()
        node.run()
    except rospy.ROSInterruptException:
        pass