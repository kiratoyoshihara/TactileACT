import rospy
import rosbag
import time
import argparse
import numpy as np
from std_msgs.msg import Float64MultiArray
# TODO: check correct message type
from lbr_fri_msgs.msg import LBRPositionCommand, LBRState


class MotionReplayer:
    def __init__(self, bag_path):
        rospy.init_node('motion_replayer', anonymous=True)
        
        self.bag_path = bag_path
        
        # Publishers
        # KUKA Command
        self.pub_kuka = rospy.Publisher('/lbr_command', LBRPositionCommand, queue_size=1)
        # Hand-E Command
        self.pub_hande = rospy.Publisher('/hande/command', Float64MultiArray, queue_size=1)
        
        # Subscribers (for safety check / current state)
        self.current_kuka_pos = None
        rospy.Subscriber('/lbr_state', LBRState, self.kuka_cb)
        
        rospy.loginfo(f"Loading bag: {bag_path}")
        self.trajectory, self.gripper_cmds = self.load_bag()
        rospy.loginfo(f"Loaded {len(self.trajectory)} KUKA points and {len(self.gripper_cmds)} Gripper commands.")

    def kuka_cb(self, msg):
        self.current_kuka_pos = np.array(msg.measured_joint_position)

    def load_bag(self):
        """
        Extract KUKA joint states and Gripper commands from rosbag.
        Returns:
            trajectory: list of (timestamp, joint_positions)
            gripper_cmds: list of (timestamp, command_value)
        """
        bag = rosbag.Bag(self.bag_path)
        
        kuka_traj = []
        hand_cmds = []
        
        start_time = None

        # Topics to read
        # Note: Ensure these match your record_episode.sh
        topic_kuka = '/lbr_state'
        topic_hand = '/hande/command' # Replay the COMMANDS sent by human
        
        for topic, msg, t in bag.read_messages(topics=[topic_kuka, topic_hand]):
            if start_time is None:
                start_time = t.to_sec()
            
            rel_time = t.to_sec() - start_time
            
            if topic == topic_kuka:
                kuka_traj.append((rel_time, msg.measured_joint_position))
                
            elif topic == topic_hand:
                # msg is Float64MultiArray
                hand_cmds.append((rel_time, msg.data[0]))
                
        bag.close()
        return kuka_traj, hand_cmds

    def move_to_start(self):
        """
        Safely move the robot to the first point of the trajectory.
        """
        if not self.trajectory:
            rospy.logerr("No trajectory found!")
            return False

        target_pos = np.array(self.trajectory[0][1])
        rospy.loginfo("Moving to start position...")
        
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.current_kuka_pos is None:
                rate.sleep()
                continue
            
            # Simple P-Control to move to start
            error = target_pos - self.current_kuka_pos
            if np.max(np.abs(error)) < 0.02: # Threshold 0.02 rad
                rospy.loginfo("Reached start position.")
                break
            
            # Calculate step (Speed limit)
            step = error * 0.05 # Gain
            cmd_pos = self.current_kuka_pos + step
            
            # Publish
            msg = LBRPositionCommand()
            msg.joint_position = cmd_pos.tolist()
            self.pub_kuka.publish(msg)
            
            rate.sleep()
        
        # Hold for a moment
        rospy.sleep(1.0)
        return True

    def run(self):
        if not self.move_to_start():
            return

        rospy.loginfo("Starting Replay...")
        start_wall_time = rospy.get_time()
        
        # Indices to keep track of playback
        idx_kuka = 0
        idx_hand = 0
        
        rate = rospy.Rate(100) # Replay loop rate (higher than recording)
        
        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_wall_time
            
            # 1. Update KUKA Command
            # Find the latest point in trajectory that matches current elapsed time
            while idx_kuka < len(self.trajectory) - 1:
                if self.trajectory[idx_kuka+1][0] > elapsed:
                    break
                idx_kuka += 1
            
            if idx_kuka < len(self.trajectory):
                target_joints = self.trajectory[idx_kuka][1]
                msg = LBRPositionCommand()
                msg.joint_position = target_joints
                self.pub_kuka.publish(msg)
            
            # 2. Update Gripper Command
            # Find the latest command
            while idx_hand < len(self.gripper_cmds) - 1:
                if self.gripper_cmds[idx_hand+1][0] > elapsed:
                    break
                idx_hand += 1
                
            if idx_hand < len(self.gripper_cmds) and len(self.gripper_cmds) > 0:
                target_grip = self.gripper_cmds[idx_hand][1]
                msg = Float64MultiArray()
                msg.data = [target_grip]
                self.pub_hande.publish(msg)

            # Check if finished
            if idx_kuka >= len(self.trajectory) - 1:
                rospy.loginfo("Replay finished.")
                break
                
            rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Path to input bag file')
    args = parser.parse_args(rospy.myargv()[1:])
    
    replayer = MotionReplayer(args.bag)
    replayer.run()