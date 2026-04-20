# This imports glob so launch, config, and world files can be installed with the package.
from glob import glob
# This imports os so we can build install paths cleanly.
import os

# This imports the setuptools helpers used by ament_python packages.
from setuptools import find_packages, setup


# This stores the ROS package name in one place.
package_name = 'jackal_smann_eval'


# This registers the package metadata, data files, and console scripts.
setup(
    # This sets the package name.
    name=package_name,
    # This sets the initial package version.
    version='0.1.0',
    # This finds the Python modules that belong to this package.
    packages=find_packages(exclude=['test']),
    # This installs ROS resource files, launch files, configs, and worlds.
    data_files=[
        # This registers the package with the ament index.
        (
            'share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)],
        ),
        # This installs the package manifest.
        (os.path.join('share', package_name), ['package.xml']),
        # This installs the launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # This installs the config files.
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # This installs the world files.
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    # This keeps the Python dependency list minimal.
    install_requires=['setuptools'],
    # This marks the package as zip-safe.
    zip_safe=True,
    # This records the maintainer name.
    maintainer='sting',
    # This records the maintainer email.
    maintainer_email='sting@example.com',
    # This describes the purpose of the package.
    description='Minimal frozen-evaluation Jackal package for offline-trained SMANN experiments.',
    # This records the package license.
    license='Apache-2.0',
    # This exposes the ROS console entrypoints.
    entry_points={
        # This lists the installed ROS executables.
        'console_scripts': [
            # This starts the evaluator node.
            'jackal_smann_evaluator = jackal_smann_eval.evaluator:main',
            # This starts the goal monitor node.
            'jackal_goal_monitor = jackal_smann_eval.goal_monitor:main',
        ],
    },
)
