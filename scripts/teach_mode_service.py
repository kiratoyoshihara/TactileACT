#!/usr/bin/env python3
import rospy
from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest
from iiwa_msgs.msg import ControlMode

def enable_gravity_compensation():
    rospy.init_node('set_teach_mode')
    service_name = '/iiwa/configuration/ConfigureControlMode'
    rospy.loginfo(f"Waiting for service: {service_name}")

    rospy.wait_for_service(service_name)
    
    try:
        set_mode = rospy.ServiceProxy(service_name, ConfigureControlMode)
        req = ConfigureControlModeRequest()
		req.control_mode = ControlMode.JOINT_IMPEDANCE

        req.joint_impedance.joint_stiffness.a1 = 100.0
        req.joint_impedance.joint_stiffness.a2 = 100.0
        req.joint_impedance.joint_stiffness.a3 = 100.0
        req.joint_impedance.joint_stiffness.a4 = 100.0
        req.joint_impedance.joint_stiffness.a5 = 100.0
        req.joint_impedance.joint_stiffness.a6 = 100.0
        req.joint_impedance.joint_stiffness.a7 = 100.0

        req.joint_impedance.joint_damping.a1 = 0.7
        req.joint_impedance.joint_damping.a2 = 0.7
        req.joint_impedance.joint_damping.a3 = 0.7
        req.joint_impedance.joint_damping.a4 = 0.7
        req.joint_impedance.joint_damping.a5 = 0.7
        req.joint_impedance.joint_damping.a6 = 0.7
        req.joint_impedance.joint_damping.a7 = 0.7
        
        resp = set_mode(req)

        if resp.success:
            rospy.loginfo("SUCCESS: KUKA is now in Gravity Compensation Mode! (Stiffness=0)")

        else:
            rospy.logerr(f"FAILED: {resp.error}")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":

    enable_gravity_compensation()
