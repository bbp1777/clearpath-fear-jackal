# This imports os so the packaged world path can be built.
import os

# This imports get_package_share_directory so the package world file can be located.
from ament_index_python.packages import get_package_share_directory
# This imports LaunchDescription for the ROS launch entrypoint.
from launch import LaunchDescription
# This imports launch arguments.
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# This imports the Python launch source wrapper.
from launch.launch_description_sources import PythonLaunchDescriptionSource
# This imports LaunchConfiguration so arguments can be reused.
from launch.substitutions import LaunchConfiguration
# This imports the Node action for starting ROS nodes.
from launch_ros.actions import Node
# This imports FindPackageShare for package-relative launch includes.
from launch_ros.substitutions import FindPackageShare


# This builds the simulator-only launch description used by the evaluator child process.
def generate_launch_description():
    """Launch Gazebo, spawn the Clearpath robot, and bridge the simulation clock."""

    # This locates the package share directory.
    package_share = get_package_share_directory('jackal_smann_eval')
    # This declares the namespace argument.
    namespace_arg = DeclareLaunchArgument('namespace', default_value='jackal_sidewalk')
    # This declares the setup path argument.
    setup_path_arg = DeclareLaunchArgument('setup_path', default_value='/workspaces/clearpath_docker/sim_setup')
    # This declares the sim-time argument.
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    # This declares the spawn x argument.
    x_arg = DeclareLaunchArgument('x', default_value='-14.0')
    # This declares the spawn y argument.
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    # This declares the spawn z argument.
    z_arg = DeclareLaunchArgument('z', default_value='0.20')
    # This declares the spawn yaw argument.
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')
    # This declares the world name argument.
    world_name_arg = DeclareLaunchArgument('world_name', default_value='mini_sidewalk')

    # This builds the packaged world path.
    world_path = os.path.join(package_share, 'worlds', 'mini_sidewalk.sdf')

    # This includes Gazebo with the packaged world file.
    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']), launch_arguments={'gz_args': [world_path, ' -r -v 4']}.items())
    # This includes the Clearpath robot spawn launch.
    jackal = IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare('clearpath_gz'), '/launch/robot_spawn.launch.py']), launch_arguments={'setup_path': LaunchConfiguration('setup_path'), 'namespace': LaunchConfiguration('namespace'), 'world': LaunchConfiguration('world_name'), 'rviz': 'false', 'x': LaunchConfiguration('x'), 'y': LaunchConfiguration('y'), 'z': LaunchConfiguration('z'), 'yaw': LaunchConfiguration('yaw')}.items())
    # This bridges the simulation clock into ROS.
    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge', arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'], output='screen')

    # This returns the full launch description.
    return LaunchDescription([namespace_arg, setup_path_arg, use_sim_time_arg, x_arg, y_arg, z_arg, yaw_arg, world_name_arg, gazebo, jackal, clock_bridge])
