# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports Callable for the reset hook type hint.
from collections.abc import Callable

# This imports TwistStamped for robot motion commands.
from geometry_msgs.msg import TwistStamped
# This imports Node for the ROS node interface.
from rclpy.node import Node
# This imports Image for RGB and depth subscriptions.
from sensor_msgs.msg import Image
# This imports Bool for the collision subscription.
from std_msgs.msg import Bool
# This imports Float32 for the goal coverage subscription.
from std_msgs.msg import Float32

# This imports the shared action and observation containers.
from jackal_smann_eval.types import AgentAction, EnvironmentConfig, ObservationBundle


# This wraps ROS topics into a small environment-style interface.
class Environment:
    """Cache the latest Jackal sensor state and publish stamped velocity commands."""

    # This builds the publishers, subscriptions, and reset bookkeeping.
    def __init__(self, node: Node, config: EnvironmentConfig) -> None:
        """Store the node and config, then subscribe to the topics used by evaluation."""

        # This stores the ROS node reference.
        self._node = node
        # This stores the environment config.
        self.config = config
        # This stores the current episode step count.
        self._episode_steps = 0
        # This stores the latest color message.
        self._latest_color = None
        # This stores the latest depth message.
        self._latest_depth = None
        # This stores the latest goal coverage value.
        self._latest_goal_coverage = 0.0
        # This stores the latest collision flag.
        self._latest_collision = False
        # This stores the collision latch for the current episode.
        self._collision_latched = False
        # This stores whether the next episode is still waiting on fresh frames.
        self._awaiting_reset_frames = False
        # This stores the reset hook used by the evaluator.
        self._reset_callback = None
        # This stores the last commanded action for keepalive republishing.
        self._last_action = AgentAction()

        # This creates the stamped velocity publisher.
        self._cmd_pub = self._node.create_publisher(TwistStamped, self.config.cmd_vel_topic, 10)
        # This subscribes to the color topic.
        self._node.create_subscription(Image, self.config.color_topic, self._on_color, 10)
        # This subscribes to the depth topic.
        self._node.create_subscription(Image, self.config.depth_topic, self._on_depth, 10)
        # This subscribes to the goal coverage topic.
        self._node.create_subscription(Float32, self.config.goal_coverage_topic, self._on_goal_coverage, 10)
        # This subscribes to the collision topic.
        self._node.create_subscription(Bool, self.config.collision_topic, self._on_collision, 10)

    # This stores the evaluator-owned reset callback.
    def register_reset_callback(self, callback: Callable[[], None]) -> None:
        """Store the simulator reset callback so the environment can request clean episodes."""

        # This stores the reset callback.
        self._reset_callback = callback

    # This stores the latest usable color frame.
    def _on_color(self, msg: Image) -> None:
        """Cache the newest color frame and release reset gating when fresh frames arrive."""

        # This stores the latest color frame.
        self._latest_color = msg
        # This clears the reset wait when both cameras are ready and collision is clear.
        if self._latest_depth is not None and not self._latest_collision:
            # This marks the environment as ready again.
            self._awaiting_reset_frames = False

    # This stores the latest usable depth frame.
    def _on_depth(self, msg: Image) -> None:
        """Cache the newest depth frame and release reset gating when fresh frames arrive."""

        # This stores the latest depth frame.
        self._latest_depth = msg
        # This clears the reset wait when both cameras are ready and collision is clear.
        if self._latest_color is not None and not self._latest_collision:
            # This marks the environment as ready again.
            self._awaiting_reset_frames = False

    # This stores the latest goal coverage scalar.
    def _on_goal_coverage(self, msg: Float32) -> None:
        """Cache the newest goal coverage value from the goal monitor."""

        # This stores the newest goal coverage value.
        self._latest_goal_coverage = float(msg.data)

    # This stores the latest collision flag and latches terminal collisions.
    def _on_collision(self, msg: Bool) -> None:
        """Cache the newest collision flag and latch terminal collisions for the episode."""

        # This stores the raw collision value.
        collision = bool(msg.data)
        # This stores the latest collision value.
        self._latest_collision = collision

        # This latches terminal collisions.
        if collision:
            # This stores the terminal collision latch.
            self._collision_latched = True
        else:
            # This clears the reset wait when both cameras are ready after a reset.
            if self._awaiting_reset_frames and self._latest_color is not None and self._latest_depth is not None:
                # This marks the environment as ready again.
                self._awaiting_reset_frames = False

    # This reports whether the evaluator can safely step the episode.
    def is_ready(self) -> bool:
        """Return true when fresh RGB-D data exists and the reset gate has cleared."""

        # This returns whether the latest camera frames exist and reset gating has cleared.
        return self._latest_color is not None and self._latest_depth is not None and not self._awaiting_reset_frames

    # This publishes one stamped Twist command.
    def _publish_twist(self, action: AgentAction) -> None:
        """Build one stamped Twist message and publish it to the Jackal controller."""

        # This creates the stamped command message.
        cmd = TwistStamped()
        # This stamps the command with the current ROS time.
        cmd.header.stamp = self._node.get_clock().now().to_msg()
        # This tags the command with the robot base frame.
        cmd.header.frame_id = 'base_link'
        # This fills the forward velocity field.
        cmd.twist.linear.x = float(action.linear_x)
        # This fills the yaw rate field.
        cmd.twist.angular.z = float(action.angular_z)
        # This publishes the command.
        self._cmd_pub.publish(cmd)

    # This stores and publishes the latest action.
    def publish_action(self, action: AgentAction) -> None:
        """Cache the latest action and publish it immediately."""

        # This stores the latest action for later keepalive use.
        self._last_action = AgentAction(linear_x=float(action.linear_x), angular_z=float(action.angular_z))
        # This publishes the latest action right away.
        self._publish_twist(self._last_action)

    # This republishes the last action with a fresh timestamp.
    def republish_last_action(self) -> None:
        """Republish the last command with a fresh timestamp when needed."""

        # This republishes the cached action.
        self._publish_twist(self._last_action)

    # This increments the current episode step count.
    def increment_step(self) -> None:
        """Advance the episode step counter by one."""

        # This increments the step count.
        self._episode_steps += 1

    # This builds the latest observation bundle.
    def build_observation(self) -> ObservationBundle:
        """Package the current RGB-D messages and task flags into one observation object."""

        # This checks whether the goal completion threshold was met.
        goal_reached = self._latest_goal_coverage >= self.config.goal_completion_threshold
        # This checks whether the step limit was reached.
        truncated = self._episode_steps >= self.config.max_episode_steps
        # This returns the observation bundle.
        return ObservationBundle(color_msg=self._latest_color, depth_msg=self._latest_depth, goal_coverage=float(self._latest_goal_coverage), collision=bool(self._collision_latched), goal_reached=bool(goal_reached), step_index=int(self._episode_steps), terminal=bool(self._collision_latched), truncated=bool(truncated))

    # This resets episode-local state and requests a simulator reset when available.
    def reset(self) -> ObservationBundle:
        """Clear episode state, request a simulator reset, and wait for fresh post-reset data."""

        # This clears the episode step counter.
        self._episode_steps = 0
        # This clears the latest color frame.
        self._latest_color = None
        # This clears the latest depth frame.
        self._latest_depth = None
        # This clears the latest goal coverage value.
        self._latest_goal_coverage = 0.0
        # This clears the latest collision value.
        self._latest_collision = False
        # This clears the collision latch.
        self._collision_latched = False
        # This clears the cached action.
        self._last_action = AgentAction()
        # This enables the reset gating until fresh frames arrive.
        self._awaiting_reset_frames = True

        # This requests a simulator reset when a callback exists.
        if self._reset_callback is not None:
            # This calls the evaluator-owned reset function.
            self._reset_callback()

        # This returns the cleared observation state.
        return self.build_observation()
