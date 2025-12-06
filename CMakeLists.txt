cmake_minimum_required(VERSION 3.0.2)
project(tactile_act)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tactile_act
#  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts (Optional: allows running via 'rosrun tactile_act ...')
catkin_install_python(PROGRAMS
  scripts/hande_driver_node.py
  scripts/teleop_gripper.py
  inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)