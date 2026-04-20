"""
Interactive capture tool for building manual three-step RGB-D sequences one state at a time.
"""
from __future__ import annotations

import argparse
import threading
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from fear_jackal_sim.manual_sequence_dataset import save_manual_sequence_sample
from fear_jackal_sim.vision_utils import depth_image_to_numpy, resize_nearest, rgb_image_to_numpy


class ManualSequenceCapture(Node):
    """
    ROS node that saves a fixed-length RGB-D snippet for offline fear training.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize buffers, subscriptions, and capture mode.
        """
        super().__init__('fear_manual_capture')
        self.args = args
        self.lookback = max(int(args.lookback), 1)
        self.image_size = max(int(args.image_size), 1)
        self.capture_interval_s = max(float(args.capture_interval_s), 0.01)
        self.capture_mode = str(args.capture_mode).strip().lower()
        self.rgb_window: deque = deque(maxlen=self.lookback)
        self.depth_window: deque = deque(maxlen=self.lookback)
        self._latest_color: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._latest_color_stamp_ns = -1
        self._latest_depth_stamp_ns = -1
        self._last_used_color_stamp_ns = -1
        self._last_used_depth_stamp_ns = -1
        self._last_capture_time = self.get_clock().now()
        self._capture_requested = False
        self._sample_saved = False

        self.create_subscription(Image, args.color_topic, self._on_color, 10)
        self.create_subscription(Image, args.depth_topic, self._on_depth, 10)
        self.create_timer(0.05, self._capture_tick)
        self.get_logger().info(
            f"Waiting to capture {self.lookback} steps from color={args.color_topic} depth={args.depth_topic} into {args.dataset_dir} as label={args.label}."
        )

        if self.capture_mode == 'enter':
            # This matches Rodney Sanchez's fixed short lookback idea, but lets you
            # decide exactly when each of the three states is committed to the sample.
            self.get_logger().info(
                'Enter-triggered capture is active. Reposition the robot, then press Enter once per step.'
            )
            threading.Thread(target=self._wait_for_enter, daemon=True).start()
        else:
            self.get_logger().info(
                f'Timed capture is active. A new step will be sampled every {self.capture_interval_s:.2f}s once frames are fresh.'
            )

    def _wait_for_enter(self) -> None:
        """
        Request one capture each time the operator presses Enter.
        """
        while rclpy.ok() and not self._sample_saved:
            try:
                input()
            except EOFError:
                self.get_logger().warning(
                    'Standard input is unavailable, so enter-triggered capture cannot continue. '
                    'Restart with --capture-mode timed if you want automatic sampling.'
                )
                return
            except KeyboardInterrupt:
                return
            self._capture_requested = True

    def _on_color(self, msg: Image) -> None:
        """
        Cache the most recent color frame.
        """
        self._latest_color = msg
        self._latest_color_stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)

    def _on_depth(self, msg: Image) -> None:
        """
        Cache the most recent depth frame.
        """
        self._latest_depth = msg
        self._latest_depth_stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)

    def _capture_tick(self) -> None:
        """
        Capture one step when possible and save the sample once the lookback is full.
        """
        if self._sample_saved:
            return
        if self._latest_color is None or self._latest_depth is None:
            return

        now = self.get_clock().now()
        if self.capture_mode == 'timed':
            elapsed = (now.nanoseconds - self._last_capture_time.nanoseconds) / 1e9
            if elapsed < self.capture_interval_s:
                return
        elif not self._capture_requested:
            return

        if self._latest_color_stamp_ns <= self._last_used_color_stamp_ns:
            return
        if self._latest_depth_stamp_ns <= self._last_used_depth_stamp_ns:
            return

        try:
            rgb = resize_nearest(rgb_image_to_numpy(self._latest_color), self.image_size, self.image_size)
            depth = resize_nearest(depth_image_to_numpy(self._latest_depth), self.image_size, self.image_size)
        except Exception as exc:
            self.get_logger().warning(f'Unable to decode capture frame: {exc}')
            return

        self.rgb_window.append(rgb.transpose(2, 0, 1).astype('uint8', copy=False))
        self.depth_window.append(depth.astype('float32', copy=False))
        self._last_used_color_stamp_ns = self._latest_color_stamp_ns
        self._last_used_depth_stamp_ns = self._latest_depth_stamp_ns
        self._last_capture_time = now
        self._capture_requested = False

        self.get_logger().info(f'Captured step {len(self.rgb_window)}/{self.lookback}.')
        if len(self.rgb_window) < self.lookback:
            if self.capture_mode == 'enter':
                self.get_logger().info('Move the robot to the next state, then press Enter again.')
            return

        # Each saved sample is one Sanchez-style short memory snippet: a fixed lookback
        # of three consecutive RGB-D observations with a known reward label.
        metadata = save_manual_sequence_sample(
            output_dir=self.args.dataset_dir,
            rgb_window=list(self.rgb_window),
            depth_window=list(self.depth_window),
            label=self.args.label,
            reward=self.args.reward,
            note=self.args.note,
        )
        self._sample_saved = True
        self.get_logger().info(
            f"Saved manual sequence sample path={metadata['path']} label={metadata['label']} reward={metadata['reward']:.3f}"
        )
        rclpy.shutdown()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """
    Parse CLI arguments for manual sequence capture.
    """
    parser = argparse.ArgumentParser(description='Capture a manual 3-step RGB+depth memory sample from the running Jackal sim.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--label', choices=['unsafe', 'safe'], required=True)
    parser.add_argument('--reward', type=float, default=None)
    parser.add_argument('--lookback', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--capture-mode', choices=['enter', 'timed'], default='enter')
    parser.add_argument('--capture-interval-s', type=float, default=0.25)
    parser.add_argument('--note', default='')
    parser.add_argument('--color-topic', default='/jackal_sidewalk/sensors/camera_0/color/image')
    parser.add_argument('--depth-topic', default='/jackal_sidewalk/sensors/camera_0/depth/image')
    return parser.parse_known_args()


def main(args=None) -> None:
    """
    ROS entrypoint that spins the manual capture node.
    """
    parsed_args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = ManualSequenceCapture(parsed_args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
