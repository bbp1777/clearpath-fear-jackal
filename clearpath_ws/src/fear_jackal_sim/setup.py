from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'fear_jackal_sim'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)],
        ),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sting',
    maintainer_email='sting@example.com',
    description='Custom Jackal sidewalk simulation scaffold for fear-based RL experiments.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fear_sensor_subscriber = fear_jackal_sim.fear_sensor_subscriber:main',
            'fear_trainer = fear_jackal_sim.trainer:main',
            'fear_goal_monitor = fear_jackal_sim.goal_monitor:main',
            'fear_dataset_exporter = fear_jackal_sim.fear_dataset_exporter:main',
            'train_smann = fear_jackal_sim.train_smann:main',
            'train_smann_grid = fear_jackal_sim.train_smann_grid:main',
            'fear_manual_capture = fear_jackal_sim.fear_manual_capture:main',
            'command_timeout_overrides = fear_jackal_sim.command_timeout_overrides:main',
            'train_memory_fear = fear_jackal_sim.train_memory_fear:main',
        ],
    },
)


