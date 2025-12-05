from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            'scripts/launch_sensors.py' 
        ])
    )

    hande_driver = Node(
        package='tactile_act', 
        executable='python3',
        arguments=['scripts/hande_driver_node.py'],
        name='hande_driver',
        output='screen'
    )

    return LaunchDescription([
        sensors_launch,
        hande_driver
    ])