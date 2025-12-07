from setuptools import setup
import os
from glob import glob

package_name = 'tactile_act'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.models', f'{package_name}.utils', f'{package_name}.configs', f'{package_name}.data'],
    package_dir={'': '.'}, 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('scripts/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kiratoyoshihara',
    maintainer_email='kiratoyoshihara@gmail.com',
    description='Tactile-ACT package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = inference_node:main',
            'hande_driver = scripts.hande_driver_node:main',
            'teleop_gripper = scripts.teleop_gripper:main',
        ],
    },
)