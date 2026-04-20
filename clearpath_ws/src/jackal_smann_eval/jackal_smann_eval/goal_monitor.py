# This imports future annotations so forward type hints stay simple.
from __future__ import annotations

# This imports rclpy for the ROS node entrypoint.
import rclpy
# This imports odometry for off-sidewalk checks.
from nav_msgs.msg import Odometry
# This imports Duration for hold timers.
from rclpy.duration import Duration
# This imports Node for the ROS node base class.
from rclpy.node import Node
# This imports Time for sim-time jump handling.
from rclpy.time import Time
# This imports Image for the camera subscriptions.
from sensor_msgs.msg import Image
# This imports Bool for the collision topic.
from std_msgs.msg import Bool
# This imports Float32 for the goal coverage topic.
from std_msgs.msg import Float32

# This imports the shared image helpers.
from jackal_smann_eval.vision_utils import ImageDecodingError, compute_green_goal_coverage, rgb_image_to_numpy


# This turns RGB, depth, and odometry into task signals.
class JackalGoalMonitor(Node):
    """Publish green goal coverage and terminal collision flags for the evaluator."""

    # This declares parameters and starts the subscriptions.
    def __init__(self) -> None:
        """Read thresholds, set cached state, and subscribe to the required topics."""

        # This names the ROS node.
        super().__init__('jackal_goal_monitor')

        # This declares the color topic.
        self.declare_parameter('color_topic', '/jackal_sidewalk/sensors/camera_0/color/image')
        # This declares the depth topic.
        self.declare_parameter('depth_topic', '/jackal_sidewalk/sensors/camera_0/depth/image')
        # This declares the odometry topic.
        self.declare_parameter('odom_topic', '/jackal_sidewalk/platform/odom')
        # This declares the goal coverage output topic.
        self.declare_parameter('goal_coverage_topic', '/jackal_sidewalk/goal/coverage')
        # This declares the collision output topic.
        self.declare_parameter('collision_topic', '/jackal_sidewalk/collision')
        # This declares the green threshold.
        self.declare_parameter('goal_min_green', 160)
        # This declares the red threshold.
        self.declare_parameter('goal_max_red', 140)
        # This declares the blue threshold.
        self.declare_parameter('goal_max_blue', 140)
        # This declares the sidewalk center line.
        self.declare_parameter('sidewalk_center_y', 0.0)
        # This declares the sidewalk half width.
        self.declare_parameter('sidewalk_half_width_m', 1.30)
        # This declares the off-sidewalk margin.
        self.declare_parameter('off_sidewalk_margin_m', 0.02)
        # This declares the off-sidewalk hold time.
        self.declare_parameter('off_sidewalk_hold_s', 0.10)
        # This declares the robot half width.
        self.declare_parameter('robot_half_width_m', 0.24)
        # This declares the robot half length used by simulator-grounded collision checks.
        self.declare_parameter('robot_half_length_m', 0.32)
        # This declares the odometry freshness timeout.
        self.declare_parameter('motion_state_timeout_s', 1.0)
        # This declares the sim-time jump tolerance.
        self.declare_parameter('sim_time_jump_tolerance_s', 1.0)

        # This reads the color topic.
        color_topic = self.get_parameter('color_topic').value
        # This reads the depth topic.
        depth_topic = self.get_parameter('depth_topic').value
        # This reads the odom topic.
        odom_topic = self.get_parameter('odom_topic').value
        # This reads the goal topic.
        goal_topic = self.get_parameter('goal_coverage_topic').value
        # This reads the collision topic.
        collision_topic = self.get_parameter('collision_topic').value

        # This stores the green threshold.
        self._goal_min_green = int(self.get_parameter('goal_min_green').value)
        # This stores the red threshold.
        self._goal_max_red = int(self.get_parameter('goal_max_red').value)
        # This stores the blue threshold.
        self._goal_max_blue = int(self.get_parameter('goal_max_blue').value)
        # This stores the sidewalk center.
        self._sidewalk_center_y = float(self.get_parameter('sidewalk_center_y').value)
        # This stores the sidewalk half width.
        self._sidewalk_half_width = float(self.get_parameter('sidewalk_half_width_m').value)
        # This stores the sidewalk margin.
        self._off_sidewalk_margin = float(self.get_parameter('off_sidewalk_margin_m').value)
        # This stores the off-sidewalk hold duration.
        self._off_sidewalk_hold_duration = Duration(seconds=float(self.get_parameter('off_sidewalk_hold_s').value))
        # This stores the robot half width.
        self._robot_half_width = float(self.get_parameter('robot_half_width_m').value)
        # This stores the robot half length.
        self._robot_half_length = float(self.get_parameter('robot_half_length_m').value)
        # This stores the odometry freshness timeout.
        self._motion_state_timeout = Duration(seconds=float(self.get_parameter('motion_state_timeout_s').value))
        # This stores the sim-time jump tolerance.
        self._sim_time_jump_tolerance = Duration(seconds=float(self.get_parameter('sim_time_jump_tolerance_s').value))

        # This stores the known obstacle footprints from the packaged mini_sidewalk world.
        self._obstacle_boxes = [
            (-8.0, 0.55, 0.7, 0.7),
            (-2.5, -0.45, 0.8, 0.8),
            (3.5, 0.0, 0.7, 0.7),
            (8.5, -0.55, 0.7, 0.7),
        ]
        # This stores the latest forward position.
        self._latest_odom_x = 0.0
        # This stores the latest lateral position.
        self._latest_odom_y = self._sidewalk_center_y
        # This stores the latest odometry stamp.
        self._latest_odom_stamp = None
        # This stores the last seen sim stamp.
        self._last_sim_stamp = None
        # This stores when off-sidewalk contact began.
        self._off_sidewalk_since = None
        # This tracks the active collision state.
        self._collision_active = False

        # This creates the goal coverage publisher.
        self._coverage_pub = self.create_publisher(Float32, goal_topic, 10)
        # This creates the collision publisher.
        self._collision_pub = self.create_publisher(Bool, collision_topic, 10)
        # This subscribes to color images.
        self.create_subscription(Image, color_topic, self._on_color, 10)
        # This subscribes to depth images.
        self.create_subscription(Image, depth_topic, self._on_depth, 10)
        # This subscribes to odometry.
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)

        # This reports the active task topics.
        self.get_logger().info(f'Goal monitor listening on color={color_topic} depth={depth_topic} and publishing coverage={goal_topic} collision={collision_topic}')

    # This converts one color frame into goal coverage.
    def _on_color(self, msg: Image) -> None:
        """Decode the RGB frame and publish green goal coverage."""

        # This updates the sim-time cache.
        self._handle_sim_time_stamp(Time.from_msg(msg.header.stamp))

        try:
            # This decodes the RGB image.
            image = rgb_image_to_numpy(msg)
        except ImageDecodingError as exc:
            # This reports the image decoding problem.
            self.get_logger().warning(str(exc))
            # This stops processing this frame.
            return

        # This computes the green goal coverage.
        coverage = compute_green_goal_coverage(image, min_green=self._goal_min_green, max_red=self._goal_max_red, max_blue=self._goal_max_blue)
        # This publishes the goal coverage value.
        self._coverage_pub.publish(Float32(data=float(coverage)))

    # This caches the latest odometry needed for sidewalk checks.
    def _on_odom(self, msg: Odometry) -> None:
        """Cache the robot lateral position for off-sidewalk detection."""

        # This converts the message stamp into a ROS time.
        stamp = Time.from_msg(msg.header.stamp)
        # This updates the sim-time cache.
        self._handle_sim_time_stamp(stamp)
        # This stores the latest x position.
        self._latest_odom_x = float(msg.pose.pose.position.x)
        # This stores the latest y position.
        self._latest_odom_y = float(msg.pose.pose.position.y)
        # This stores the latest odometry stamp.
        self._latest_odom_stamp = stamp
        # This publishes a simulator-grounded collision state from the latest pose.
        self._publish_collision_state(stamp)

    # This uses depth frame timing to refresh collision state without reading depth distances.
    def _on_depth(self, msg: Image) -> None:
        """Refresh terminal collision checks without using depth as a terminal signal."""

        # This converts the message stamp into a ROS time.
        stamp = Time.from_msg(msg.header.stamp)
        # This updates the sim-time cache.
        self._handle_sim_time_stamp(stamp)
        # This publishes the collision state from odometry and known simulator geometry.
        self._publish_collision_state(stamp)

    # This computes and publishes the current simulator-grounded collision state.
    def _publish_collision_state(self, stamp: Time) -> None:
        """Publish collision when the robot footprint overlaps an obstacle or leaves the sidewalk."""

        # This checks for overlap with the known obstacle footprints.
        obstacle_collision = self._compute_obstacle_collision(stamp)
        # This checks whether the robot has moved off the sidewalk long enough.
        off_sidewalk_collision = self._compute_off_sidewalk_collision(stamp)
        # This merges both terminal conditions.
        collision = bool(obstacle_collision or off_sidewalk_collision)

        # This keeps terminal collisions latched until a true reset/relaunch
        # clears the latch through the sim-time rewind path below.
        if self._collision_active:
            self._collision_pub.publish(Bool(data=True))
            return

        # This reports the first collision event.
        if collision and not self._collision_active:
            # This builds the reason list.
            reasons = []
            # This records an obstacle collision reason.
            if obstacle_collision:
                reasons.append('obstacle')
            # This records an off-sidewalk collision reason.
            if off_sidewalk_collision:
                reasons.append('off_sidewalk')
            # This logs the collision event.
            self.get_logger().info(f'Publishing terminal collision event reasons={"+".join(reasons)} odom_x={self._latest_odom_x:.3f} odom_y={self._latest_odom_y:.3f}')

        # This stores the current collision state.
        self._collision_active = collision
        # This publishes the collision state.
        self._collision_pub.publish(Bool(data=collision))

    # This clears cached motion state after a backwards sim-time jump.
    def _handle_sim_time_stamp(self, stamp: Time) -> None:
        """Clear cached motion state when Gazebo time jumps backward after a relaunch."""

        # This checks whether a previous sim stamp exists.
        if self._last_sim_stamp is not None:
            # This reads the jump tolerance in nanoseconds.
            tolerance_ns = self._sim_time_jump_tolerance.nanoseconds
            # This clears cached state when time moved backward enough.
            if stamp.nanoseconds + tolerance_ns < self._last_sim_stamp.nanoseconds:
                # This resets the cached forward position.
                self._latest_odom_x = 0.0
                # This resets the cached lateral position.
                self._latest_odom_y = self._sidewalk_center_y
                # This clears the odometry stamp.
                self._latest_odom_stamp = None
                # This clears the off-sidewalk timer.
                self._off_sidewalk_since = None
                # This clears the collision state.
                self._collision_active = False
                # This republishes a cleared collision state.
                self._collision_pub.publish(Bool(data=False))

        # This stores the latest sim stamp.
        self._last_sim_stamp = stamp

    # This detects overlap with the known obstacle boxes in the packaged world.
    def _compute_obstacle_collision(self, now: Time) -> bool:
        """Return true when the robot footprint overlaps any configured obstacle footprint."""

        # This clears obstacle collision when odometry is stale.
        if not self._motion_state_is_fresh(self._latest_odom_stamp, now):
            # This reports no collision.
            return False

        # This checks robot footprint overlap against each obstacle footprint.
        for center_x, center_y, size_x, size_y in self._obstacle_boxes:
            # This checks longitudinal overlap.
            overlaps_x = abs(self._latest_odom_x - center_x) <= ((size_x / 2.0) + self._robot_half_length)
            # This checks lateral overlap.
            overlaps_y = abs(self._latest_odom_y - center_y) <= ((size_y / 2.0) + self._robot_half_width)
            # This reports a collision when both footprint axes overlap.
            if overlaps_x and overlaps_y:
                # This reports obstacle contact.
                return True

        # This reports no obstacle contact.
        return False

    # This checks whether the cached odometry is recent enough to trust.
    def _motion_state_is_fresh(self, stamp: Time | None, now: Time) -> bool:
        """Return whether the latest odometry is fresh enough for sidewalk logic."""

        # This returns false when no odometry stamp exists.
        if stamp is None:
            # This reports stale motion state.
            return False

        # This checks whether the odometry age is within the allowed timeout.
        return bool(now.nanoseconds - stamp.nanoseconds <= self._motion_state_timeout.nanoseconds)

    # This detects sustained contact with non-sidewalk terrain.
    def _compute_off_sidewalk_collision(self, now: Time) -> bool:
        """Return true when the robot footprint has stayed off the sidewalk long enough."""

        # This clears the timer when odometry is stale.
        if not self._motion_state_is_fresh(self._latest_odom_stamp, now):
            # This clears the off-sidewalk timer.
            self._off_sidewalk_since = None
            # This reports no collision.
            return False

        # This computes the absolute lateral offset.
        lateral_offset = abs(self._latest_odom_y - self._sidewalk_center_y)
        # This checks whether the robot footprint is touching non-sidewalk terrain.
        touching_non_sidewalk = lateral_offset + self._robot_half_width >= (self._sidewalk_half_width - self._off_sidewalk_margin)

        # This clears the timer when the robot is back on the sidewalk.
        if not touching_non_sidewalk:
            # This clears the off-sidewalk timer.
            self._off_sidewalk_since = None
            # This reports no collision.
            return False

        # This starts the timer on the first off-sidewalk frame.
        if self._off_sidewalk_since is None:
            # This stores the first off-sidewalk time.
            self._off_sidewalk_since = now
            # This reports no collision yet.
            return False

        # This returns whether the hold time has elapsed.
        return bool(now - self._off_sidewalk_since >= self._off_sidewalk_hold_duration)


# This starts and spins the goal monitor node.
def main(args=None) -> None:
    """Create the goal monitor node and spin it until shutdown."""

    # This starts the ROS client library.
    rclpy.init(args=args)
    # This creates the goal monitor node.
    node = JackalGoalMonitor()

    try:
        # This spins the node until shutdown.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This ignores Ctrl-C shutdowns.
        pass
    finally:
        try:
            # This destroys the node cleanly.
            node.destroy_node()
        except Exception:
            # This ignores shutdown cleanup errors.
            pass
        try:
            # This shuts ROS down cleanly.
            rclpy.shutdown()
        except Exception:
            # This ignores shutdown cleanup errors.
            pass
