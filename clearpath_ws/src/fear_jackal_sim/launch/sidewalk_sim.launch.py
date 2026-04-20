"""
Launch description for the simulator-only stack, including Gazebo, Jackal spawn, bridges,
and diagnostics.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    Build and return the simulator-only launch description.
    """
    package_share = get_package_share_directory('fear_jackal_sim')
    world_name = 'mini_sidewalk'
    default_world_sdf = os.path.join(package_share, 'worlds', f'{world_name}.sdf')
    terminal_contact_topics = [
        '/jackal_sidewalk/sim/contacts/grass_left',
        '/jackal_sidewalk/sim/contacts/grass_right',
        '/jackal_sidewalk/sim/contacts/box_01',
        '/jackal_sidewalk/sim/contacts/box_02',
        '/jackal_sidewalk/sim/contacts/box_03',
        '/jackal_sidewalk/sim/contacts/box_04',
    ]
    terminal_touch_topics = [
        '/jackal_sidewalk/sim/touched/grass_left/touched',
        '/jackal_sidewalk/sim/touched/grass_right/touched',
        '/jackal_sidewalk/sim/touched/box_01/touched',
        '/jackal_sidewalk/sim/touched/box_02/touched',
        '/jackal_sidewalk/sim/touched/box_03/touched',
        '/jackal_sidewalk/sim/touched/box_04/touched',
    ]

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='jackal_sidewalk',
        description='ROS namespace used for the simulated Jackal.',
    )
    setup_path_arg = DeclareLaunchArgument(
        'setup_path',
        default_value='/workspaces/clearpath_docker/sim_setup',
        description='Directory containing Clearpath robot.yaml.',
    )
    world_sdf_arg = DeclareLaunchArgument(
        'world_sdf',
        default_value=default_world_sdf,
        description='Absolute path to the Gazebo world SDF file.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock.',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Start RViz alongside the simulator.',
    )
    enable_audio_arg = DeclareLaunchArgument(
        'enable_audio',
        default_value='false',
        description='Whether to bridge the logical microphone topics into ROS.',
    )
    x_arg = DeclareLaunchArgument('x', default_value='-14.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.20')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')

    color_topic_arg = DeclareLaunchArgument(
        'color_topic',
        default_value=PythonExpression(
            ["'/' + '", LaunchConfiguration('namespace'), "' + '/sensors/camera_0/color/image'"]
        ),
        description='ROS topic for the simulated RealSense color image.',
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value=PythonExpression(
            ["'/' + '", LaunchConfiguration('namespace'), "' + '/sensors/camera_0/depth/image'"]
        ),
        description='ROS topic for the simulated RealSense depth image.',
    )
    collision_topic_arg = DeclareLaunchArgument(
        'collision_topic',
        default_value=PythonExpression(["'/' + '", LaunchConfiguration('namespace'), "' + '/collision'"]),
        description='ROS topic used for terminal collision events.',
    )
    audio_ros_topic_arg = DeclareLaunchArgument(
        'audio_ros_topic',
        default_value=PythonExpression(
            ["'/' + '", LaunchConfiguration('namespace'), "' + '/sensors/audio/fear_level'"]
        ),
        description='ROS topic exposed for the logical microphone.',
    )
    audio_gz_topic_arg = DeclareLaunchArgument(
        'audio_gz_topic',
        default_value=PythonExpression(
            [f"'/world/{world_name}/model/' + '", LaunchConfiguration('namespace'), "' + '/robot/mic_0/detection'"]
        ),
        description='Gazebo topic produced by the logical microphone plugin.',
    )

    # This launch file is intentionally simulator-only so it can be used both for
    # manual dataset capture and as the child process restarted between episodes.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']
        ),
        launch_arguments={
            'gz_args': [LaunchConfiguration('world_sdf'), ' -r -v 4'],
        }.items(),
    )

    jackal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('clearpath_gz'), '/launch/robot_spawn.launch.py']
        ),
        launch_arguments={
            'setup_path': LaunchConfiguration('setup_path'),
            'namespace': LaunchConfiguration('namespace'),
            'world': world_name,
            'rviz': LaunchConfiguration('rviz'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'yaw': LaunchConfiguration('yaw'),
        }.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    terminal_contact_bridges = [
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'terminal_contact_bridge_{topic.rsplit("/", 1)[-1]}',
            arguments=[f'{topic}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'],
            output='screen',
        )
        for topic in terminal_contact_topics
    ]

    terminal_touch_bridges = [
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'terminal_touch_bridge_{topic.split("/")[-2]}',
            arguments=[f'{topic}@std_msgs/msg/Bool[gz.msgs.Boolean'],
            output='screen',
        )
        for topic in terminal_touch_topics
    ]

    audio_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='logical_audio_bridge',
        condition=IfCondition(LaunchConfiguration('enable_audio')),
        arguments=[
            PythonExpression(
                ["'", LaunchConfiguration('audio_gz_topic'), "@std_msgs/msg/Float64[gz.msgs.Double'"]
            )
        ],
        remappings=[
            (LaunchConfiguration('audio_gz_topic'), LaunchConfiguration('audio_ros_topic'))
        ],
        output='screen',
    )

    subscriber = Node(
        package='fear_jackal_sim',
        executable='fear_sensor_subscriber',
        name='fear_sensor_subscriber',
        output='screen',
        parameters=[
            {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'color_topic': LaunchConfiguration('color_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'collision_topic': LaunchConfiguration('collision_topic'),
                'audio_enabled': ParameterValue(LaunchConfiguration('enable_audio'), value_type=bool),
                'audio_topic': LaunchConfiguration('audio_ros_topic'),
            }
        ],
    )

    return LaunchDescription(
        [
            namespace_arg,
            setup_path_arg,
            world_sdf_arg,
            use_sim_time_arg,
            rviz_arg,
            enable_audio_arg,
            x_arg,
            y_arg,
            z_arg,
            yaw_arg,
            color_topic_arg,
            depth_topic_arg,
            collision_topic_arg,
            audio_ros_topic_arg,
            audio_gz_topic_arg,
            gazebo,
            jackal,
            clock_bridge,
            *terminal_contact_bridges,
            *terminal_touch_bridges,
            audio_bridge,
            subscriber,
        ]
    )
