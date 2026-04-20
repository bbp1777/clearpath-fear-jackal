# This imports os so the default config path can be built.
import os

# This imports get_package_share_directory so the default config file can be located.
from ament_index_python.packages import get_package_share_directory
# This imports LaunchDescription for the ROS launch entrypoint.
from launch import LaunchDescription
# This imports launch arguments and launch includes.
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# This imports the Python launch source wrapper.
from launch.launch_description_sources import PythonLaunchDescriptionSource
# This imports LaunchConfiguration so arguments can be reused.
from launch.substitutions import LaunchConfiguration
# This imports the Node action for starting ROS nodes.
from launch_ros.actions import Node


# This builds the top-level frozen-evaluation launch description.
def generate_launch_description():
    """Launch the simulator once, then run the goal monitor and single-episode evaluator."""

    # This locates the package share directory.
    package_share = get_package_share_directory('jackal_smann_eval')
    # This builds the default parameter file path.
    default_config = os.path.join(package_share, 'config', 'evaluator.yaml')

    # This declares the parameter file argument.
    config_arg = DeclareLaunchArgument('config', default_value=default_config)
    # This declares the reward mode sweep argument.
    reward_mode_arg = DeclareLaunchArgument('reward_mode', default_value='combined')
    # This declares the fear threshold sweep argument.
    fear_threshold_arg = DeclareLaunchArgument('smann_fear_threshold', default_value='0.50')
    # This declares the checkpoint override argument.
    checkpoint_arg = DeclareLaunchArgument('smann_checkpoint', default_value='/workspaces/clearpath_docker/clearpath_ws/logs/rodney_training/jackal_mann_independent/weights')

    # This includes the simulator-only launch once at startup.
    simulator = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(package_share, 'launch', 'sidewalk_sim.launch.py')))
    # This starts the goal monitor node.
    goal_monitor = Node(package='jackal_smann_eval', executable='jackal_goal_monitor', name='jackal_goal_monitor', output='screen', parameters=[LaunchConfiguration('config')])
    # This starts the evaluator node.
    evaluator = Node(package='jackal_smann_eval', executable='jackal_smann_evaluator', name='jackal_smann_evaluator', output='screen', parameters=[LaunchConfiguration('config'), {'reward_mode': LaunchConfiguration('reward_mode'), 'smann_fear_threshold': LaunchConfiguration('smann_fear_threshold'), 'smann_checkpoint': LaunchConfiguration('smann_checkpoint')}])

    # This returns the top-level launch description.
    return LaunchDescription([config_arg, reward_mode_arg, fear_threshold_arg, checkpoint_arg, simulator, goal_monitor, evaluator])
