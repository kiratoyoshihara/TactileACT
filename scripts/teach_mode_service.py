#!/usr/bin/env python3
import rospy
from iiwa_msgs.msg import ControlMode
from iiwa_msgs.srv import ConfigureControlMode, ConfigureControlModeRequest

def activate_cartesian_impedance():
    rospy.init_node('kuka_impedance_service')
    service_name = '/iiwa/configuration/ConfigureControlMode'
    rospy.loginfo(f"Waiting for service: {service_name}")
    rospy.wait_for_service(service_name)
    
  try:
        configure_mode = rospy.ServiceProxy(service_name, ConfigureControlMode)
        req = ConfigureControlModeRequest()
        req.control_mode = ControlMode.CARTESIAN_IMPEDANCE
    
        req.cartesian_impedance.cartesian_stiffness.x = 50.0
        req.cartesian_impedance.cartesian_stiffness.y = 50.0
        req.cartesian_impedance.cartesian_stiffness.z = 50.0
        req.cartesian_impedance.cartesian_stiffness.a = 30.0
        req.cartesian_impedance.cartesian_stiffness.b = 30.0
        req.cartesian_impedance.cartesian_stiffness.c = 30.0
    
        req.cartesian_impedance.cartesian_damping.x = 1.0
        req.cartesian_impedance.cartesian_damping.y = 1.0
        req.cartesian_impedance.cartesian_damping.z = 1.0
        req.cartesian_impedance.cartesian_damping.a = 1.0
        req.cartesian_impedance.cartesian_damping.b = 1.0
        req.cartesian_impedance.cartesian_damping.c = 1.0
        req.cartesian_impedance.nullspace_stiffness = 0.7
        req.cartesian_impedance.nullspace_damping = 0.7
        
        resp = configure_mode(req)

      if resp.success:
            rospy.loginfo("SUCCESS: Cartesian Impedance Mode Configured.")
            rospy.loginfo("Note: Stiffness is set high (X/Y=1500). Robot will hold position.")
      else:
            rospy.logerr(f"FAILED: {resp.error}")
      except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    activate_cartesian_impedance()










