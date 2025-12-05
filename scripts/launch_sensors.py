from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # RealSense
    realsense_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='realsense_driver',
        namespace='camera',  # /camera/image_raw
        output='screen',
        parameters=[{
            'video_device': '/dev/video6',   # TODO: check here
            'framerate': 30.0,
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'yuyv',         
            'brightness': 50,
            'auto_white_balance': True,
            'autoexposure': True,
        }],
        remappings=[('image_raw', 'image_raw')]
    )

    # GelSight
    gelsight_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='gelsight_driver',
        namespace='gelsight', # /gelsight/image_raw
        output='screen',
        parameters=[{
            'video_device': '/dev/video4', # TODO: check device ID 
            'framerate': 30.0,
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'mjpeg2rgb',    
            'brightness': 50,
            'auto_white_balance': False,
        }],
        remappings=[('image_raw', 'image_raw')]
    )

    return LaunchDescription([
        realsense_node,
        gelsight_node
    ])