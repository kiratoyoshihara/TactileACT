from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            'scripts/sensors.launch.py' 
        ])
    )

    hande_driver = ExecuteProcess(
        cmd=[
            'python3', 
            'scripts/hande_driver_node.py'
        ],
        output='screen'
    )

    return LaunchDescription([
        sensors_launch,
        hande_driver
    ])