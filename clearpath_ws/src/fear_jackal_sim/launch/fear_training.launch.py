"""
Launch description for the training stack, including the goal monitor, trainer, and optional
simulator bringup arguments.
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """
    Build and return the full training launch description.
    """
    package_share = get_package_share_directory('fear_jackal_sim')
    default_trainer_config = os.path.join(package_share, 'config', 'fear_trainer.yaml')
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
    trainer_config_arg = DeclareLaunchArgument(
        'trainer_config',
        default_value=default_trainer_config,
        description='Parameter file for the fear trainer node.',
    )
    start_sim_arg = DeclareLaunchArgument(
        'start_sim',
        default_value='true',
        description='Whether to start the simulator bringup alongside the trainer.',
    )
    manage_sim_process_arg = DeclareLaunchArgument(
        'manage_sim_process',
        default_value='true',
        description='Let the trainer own and fully relaunch the simulation stack between episodes.',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Forwarded to the simulator launch.',
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock for trainer-side nodes.',
    )
    enable_audio_arg = DeclareLaunchArgument(
        'enable_audio',
        default_value='false',
        description='Keep the logical microphone disabled for this RGB-D pass.',
    )
    fear_model_mode_arg = DeclareLaunchArgument(
        'fear_model_mode',
        default_value='none',
        description='Select none for base PPO, smann for Sanchez-style intrinsic fear, or memory_similarity for the older manual-bank path.',
    )
    manual_memory_dataset_dir_arg = DeclareLaunchArgument(
        'manual_memory_dataset_dir',
        default_value='',
        description='Optional manual three-step dataset directory used to build the fear bank on the fly.',
    )
    manual_memory_bank_path_arg = DeclareLaunchArgument(
        'manual_memory_bank_path',
        default_value='',
        description='Offline memory bank built from manual three-step samples.',
    )
    smann_checkpoint_arg = DeclareLaunchArgument(
        'smann_checkpoint',
        default_value='/workspaces/clearpath_docker/clearpath_ws/logs/smann_training/smann_grid/final_selected/weights',
        description='Checkpoint directory containing the Jackal SMANN weights saved from offline training.',
    )
    smann_dataset_dir_arg = DeclareLaunchArgument(
        'smann_dataset_dir',
        default_value='/workspaces/clearpath_docker/clearpath_ws/logs/manual_dataset',
        description='Manual low-shot Jackal SMANN dataset used for run provenance logging.',
    )
    # This argument makes reward-mode sweeps easy to run from the launch command line.
    reward_mode_arg = DeclareLaunchArgument(
        'reward_mode',
        default_value='external_only',
        description='Choose external_only for base PPO or combined to add an intrinsic fear penalty.',
    )
    # This argument makes the SMANN code source explicit so CarRacingTesting is used live.
    fear_repo_path_arg = DeclareLaunchArgument(
        'fear_repo_path',
        default_value='/workspaces/clearpath_docker/Behavior-Intrinsic-Fear-main/CarRacingTesting',
        description='Behavior-Intrinsic-Fear source directory used for live SMANN imports.',
    )
    sanchez_upstream_repo_arg = DeclareLaunchArgument(
        'sanchez_upstream_repo',
        default_value='https://github.com/ras8047/Behavior-Intrinsic-Fear',
        description='Canonical Sanchez source repository used for source-parity audits.',
    )
    sanchez_upstream_commit_arg = DeclareLaunchArgument(
        'sanchez_upstream_commit',
        default_value='',
        description='Optional pinned Sanchez upstream commit recorded in TensorBoard metadata.',
    )
    # This argument exposes the paper-threshold sweep directly at launch time.
    smann_fear_threshold_arg = DeclareLaunchArgument(
        'smann_fear_threshold',
        default_value='0.50',
        description='Fear score cutoff used for SMANN reward gating; tune from validation distributions.',
    )
    max_episode_steps_arg = DeclareLaunchArgument(
        'max_episode_steps',
        default_value='750',
        description='Maximum control steps per episode for the default training runs.',
    )
    goal_completion_threshold_arg = DeclareLaunchArgument(
        'goal_completion_threshold',
        default_value='0.50',
        description='RGB goal-block coverage threshold required for sparse success reward.',
    )
    # This argument switches between training and frozen evaluation runs.
    evaluation_only_arg = DeclareLaunchArgument(
        'evaluation_only',
        default_value='false',
        description='Disable all online learning updates when true; keep false for PPO training.',
    )
    enable_online_smann_updates_arg = DeclareLaunchArgument(
        'enable_online_smann_updates',
        default_value='false',
        description='Enable online SMANN vicarious-conditioning updates. Keep false for the main paper comparison.',
    )
    # This argument keeps the action source configurable for ablation experiments.
    use_policy_network_arg = DeclareLaunchArgument(
        'use_policy_network',
        default_value='true',
        description='Use the PPO action head. Set false only for the reactive fear-only ablation.',
    )
    # This argument makes it easy to separate tensorboard runs for different thresholds.
    run_name_arg = DeclareLaunchArgument(
        'run_name',
        default_value='ppo_rgbdcnn_base',
        description='Run name used for TensorBoard and experiment bookkeeping.',
    )
    max_episodes_arg = DeclareLaunchArgument(
        'max_episodes',
        default_value='50',
        description='Stop the trainer cleanly after this many completed episodes. Set 0 to run until interrupted.',
    )
    random_seed_arg = DeclareLaunchArgument(
        'random_seed',
        default_value='0',
        description='Random seed used for PPO initialization and stochastic action sampling.',
    )
    run_artifact_dir_arg = DeclareLaunchArgument(
        'run_artifact_dir',
        default_value='/workspaces/clearpath_docker/clearpath_ws/logs/paper_runs_rgbdcnn',
        description='Root directory for per-run summaries, PPO checkpoints, and other generated outputs.',
    )
    fear_reactive_policy_arg = DeclareLaunchArgument(
        'fear_reactive_policy',
        default_value='false',
        description='Enable the reactive fear-only ablation action override. Keep false for PPO plus fear reward runs.',
    )
    control_period_s_arg = DeclareLaunchArgument(
        'control_period_s',
        default_value='0.5',
        description='Policy control period in seconds. The final paper setup uses 0.5 s.',
    )
    episode_archive_dir_arg = DeclareLaunchArgument(
        'episode_archive_dir',
        default_value='/workspaces/clearpath_docker/clearpath_ws/logs/episode_archives',
        description='Directory where episode replay archives are stored for offline fear-dataset export.',
    )
    x_arg = DeclareLaunchArgument('x', default_value='-14.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.20')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value=PythonExpression(["'", LaunchConfiguration('namespace'), "' + '/robot'"]),
        description='Gazebo entity name used when resetting the robot pose.',
    )
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
    cmd_vel_topic_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value=PythonExpression(["'/' + '", LaunchConfiguration('namespace'), "' + '/cmd_vel'"]),
        description='ROS topic used for trainer velocity commands.',
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value=PythonExpression(["'/' + '", LaunchConfiguration('namespace'), "' + '/platform/odom'"]),
        description='ROS odometry topic used for stuck detection.',
    )
    goal_topic_arg = DeclareLaunchArgument(
        'goal_coverage_topic',
        default_value=PythonExpression(["'/' + '", LaunchConfiguration('namespace'), "' + '/goal/coverage'"]),
        description='Topic that publishes goal visibility coverage.',
    )
    collision_topic_arg = DeclareLaunchArgument(
        'collision_topic',
        default_value=PythonExpression(["'/' + '", LaunchConfiguration('namespace'), "' + '/collision'"]),
        description='Topic that publishes near-collision or collision events.',
    )

    # When the trainer owns simulator relaunches, this include stays disabled and
    # Gazebo is spawned as a managed child process from trainer.py instead.
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('fear_jackal_sim'), '/launch/sidewalk_sim.launch.py']
        ),
        condition=IfCondition(
            PythonExpression([
                "'",
                LaunchConfiguration('start_sim'),
                "' == 'true' and '",
                LaunchConfiguration('manage_sim_process'),
                "' != 'true'",
            ])
        ),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'setup_path': LaunchConfiguration('setup_path'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'rviz': LaunchConfiguration('rviz'),
            'enable_audio': LaunchConfiguration('enable_audio'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'yaw': LaunchConfiguration('yaw'),
            'color_topic': LaunchConfiguration('color_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'collision_topic': LaunchConfiguration('collision_topic'),
        }.items(),
    )

    goal_monitor = Node(
        package='fear_jackal_sim',
        executable='fear_goal_monitor',
        name='fear_goal_monitor',
        output='screen',
        parameters=[
            {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'color_topic': LaunchConfiguration('color_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'goal_coverage_topic': LaunchConfiguration('goal_coverage_topic'),
                'collision_topic': LaunchConfiguration('collision_topic'),
                'terminal_contact_topics': terminal_contact_topics,
                'terminal_touch_topics': terminal_touch_topics,
                'use_odometry_collision_fallback': False,
            }
        ],
    )

    trainer = Node(
        package='fear_jackal_sim',
        executable='fear_trainer',
        name='fear_trainer',
        output='screen',
        parameters=[
            LaunchConfiguration('trainer_config'),
            {
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'namespace': LaunchConfiguration('namespace'),
                'color_topic': LaunchConfiguration('color_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'enable_audio': ParameterValue(LaunchConfiguration('enable_audio'), value_type=bool),
                'audio_topic': '',
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'goal_coverage_topic': LaunchConfiguration('goal_coverage_topic'),
                'collision_topic': LaunchConfiguration('collision_topic'),
                'world_name': 'mini_sidewalk',
                'model_name': LaunchConfiguration('model_name'),
                'setup_path': LaunchConfiguration('setup_path'),
                'rviz': ParameterValue(LaunchConfiguration('rviz'), value_type=bool),
                'manage_sim_process': ParameterValue(LaunchConfiguration('manage_sim_process'), value_type=bool),
                'spawn_x': ParameterValue(LaunchConfiguration('x'), value_type=float),
                'spawn_y': ParameterValue(LaunchConfiguration('y'), value_type=float),
                'spawn_z': ParameterValue(LaunchConfiguration('z'), value_type=float),
                'spawn_yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float),
                'max_episode_steps': ParameterValue(LaunchConfiguration('max_episode_steps'), value_type=int),
                'goal_completion_threshold': ParameterValue(LaunchConfiguration('goal_completion_threshold'), value_type=float),
                'fear_model_mode': LaunchConfiguration('fear_model_mode'),
                'manual_memory_dataset_dir': LaunchConfiguration('manual_memory_dataset_dir'),
                'manual_memory_bank_path': LaunchConfiguration('manual_memory_bank_path'),
                'reward_mode': LaunchConfiguration('reward_mode'),
                'smann_checkpoint': LaunchConfiguration('smann_checkpoint'),
                'smann_dataset_dir': LaunchConfiguration('smann_dataset_dir'),
                'fear_repo_path': LaunchConfiguration('fear_repo_path'),
                'sanchez_upstream_repo': LaunchConfiguration('sanchez_upstream_repo'),
                'sanchez_upstream_commit': LaunchConfiguration('sanchez_upstream_commit'),
                'smann_fear_threshold': ParameterValue(LaunchConfiguration('smann_fear_threshold'), value_type=float),
                'evaluation_only': ParameterValue(LaunchConfiguration('evaluation_only'), value_type=bool),
                'enable_online_smann_updates': ParameterValue(LaunchConfiguration('enable_online_smann_updates'), value_type=bool),
                'use_policy_network': ParameterValue(LaunchConfiguration('use_policy_network'), value_type=bool),
                'run_name': LaunchConfiguration('run_name'),
                'max_episodes': ParameterValue(LaunchConfiguration('max_episodes'), value_type=int),
                'random_seed': ParameterValue(LaunchConfiguration('random_seed'), value_type=int),
                'run_artifact_dir': LaunchConfiguration('run_artifact_dir'),
                'fear_reactive_policy': ParameterValue(LaunchConfiguration('fear_reactive_policy'), value_type=bool),
                'control_period_s': ParameterValue(LaunchConfiguration('control_period_s'), value_type=float),
                'episode_archive_dir': LaunchConfiguration('episode_archive_dir'),
            },
        ],
    )

    return LaunchDescription(
        [
            namespace_arg,
            setup_path_arg,
            trainer_config_arg,
            start_sim_arg,
            manage_sim_process_arg,
            rviz_arg,
            use_sim_time_arg,
            enable_audio_arg,
            fear_model_mode_arg,
            manual_memory_dataset_dir_arg,
            manual_memory_bank_path_arg,
            smann_checkpoint_arg,
            smann_dataset_dir_arg,
            reward_mode_arg,
            fear_repo_path_arg,
            sanchez_upstream_repo_arg,
            sanchez_upstream_commit_arg,
            smann_fear_threshold_arg,
            max_episode_steps_arg,
            goal_completion_threshold_arg,
            evaluation_only_arg,
            enable_online_smann_updates_arg,
            use_policy_network_arg,
            run_name_arg,
            max_episodes_arg,
            random_seed_arg,
            run_artifact_dir_arg,
            fear_reactive_policy_arg,
            control_period_s_arg,
            episode_archive_dir_arg,
            x_arg,
            y_arg,
            z_arg,
            yaw_arg,
            model_name_arg,
            color_topic_arg,
            depth_topic_arg,
            cmd_vel_topic_arg,
            odom_topic_arg,
            goal_topic_arg,
            collision_topic_arg,
            sim,
            goal_monitor,
            trainer,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=trainer,
                    on_exit=[Shutdown(reason='fear_trainer completed or exited')],
                )
            ),
        ]
    )
