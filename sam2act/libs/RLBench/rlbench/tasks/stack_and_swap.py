# stack_and_swap.py: MemoryBench task -- stack two blocks then swap them back
# stack_and_swap.py: to their original patches, with a button press as memory separator.

import math
from typing import List

from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    Condition, ConditionSet, DetectedCondition, NothingGrasped
)

NUM_TARGETS = 2
GRIPPER_DOWN = [math.pi, 0, math.pi]

# Z heights measured from rearrange_block.ttm in world frame.
Z_GRASP = 0.7975
Z_APPROACH = 0.9475
Z_LIFT = 1.0135
Z_BUTTON_APPROACH = 0.9145
Z_BUTTON_PRESS = 0.7705
BLOCK_HEIGHT = 0.06
Z_STACK_GRASP = Z_GRASP + BLOCK_HEIGHT
Z_STACK_APPROACH = Z_APPROACH + BLOCK_HEIGHT


class JointTriggerCondition(Condition):
    def __init__(self, joint: Joint, position: float):
        self._joint = joint
        self._original_pos = joint.get_joint_position()
        self._pos = position
        self._done = False

    def condition_met(self):
        met = math.fabs(
            self._joint.get_joint_position() - self._original_pos) > self._pos
        if met:
            self._done = True
        return self._done, False


class DetectedTriggerCondition(Condition):
    def __init__(self, obj: Object, detector: ProximitySensor,
                 negated: bool = False):
        self._obj = obj
        self._detector = detector
        self._negated = negated
        self._done = False

    def condition_met(self):
        met = self._detector.is_detected(self._obj)
        if self._negated:
            met = not met
        if met:
            self._done = True
        return self._done, False


class StackedAtCenterCondition(Condition):
    """Latches true when upper block is stacked on lower block at center."""
    def __init__(self, lower: Shape, upper: Shape,
                 center_xy, xy_tol=0.025, z_tol=0.02):
        self._lower = lower
        self._upper = upper
        self._center_xy = center_xy
        self._xy_tol = xy_tol
        self._z_tol = z_tol
        self._done = False

    def condition_met(self):
        ap = self._lower.get_position()
        bp = self._upper.get_position()
        cx, cy = self._center_xy
        tol = self._xy_tol
        zt = self._z_tol

        # Whichever block is at center is the lower block.
        # The other block must be stacked on top of it.
        for lower, upper in [(ap, bp), (bp, ap)]:
            lower_at_center = (abs(lower[0] - cx) <= tol and
                               abs(lower[1] - cy) <= tol)
            if not lower_at_center:
                continue
            stacked = (abs(upper[0] - lower[0]) <= tol and
                       abs(upper[1] - lower[1]) <= tol and
                       abs((upper[2] - lower[2]) - BLOCK_HEIGHT) <= zt)
            if stacked:
                self._done = True
                return self._done, False

        return self._done, False


class StackAndSwap(Task):

    def init_task(self) -> None:
        self._detectors = [
            ProximitySensor(f"success{i+1}") for i in range(NUM_TARGETS)
        ]
        self._targets = [Shape(f"target{i+1}") for i in range(NUM_TARGETS)]

        self._block_a = Shape("block1")
        self._block_b = Shape("block2")
        self._center_detector = ProximitySensor("success0")

        self._button = Shape("push_buttons_target1")
        self._button_joint = Joint("target_button_joint1")

        self.spawn_boundary = SpawnBoundary([Shape("boundary")])
        self.register_graspable_objects([self._block_a, self._block_b])

        self._cx = self._center_detector.get_position()[0]
        self._cy = self._center_detector.get_position()[1]

        # Delete legacy waypoints (they have baked extension strings).
        i = 0
        while Object.exists(f'waypoint{i}'):
            Dummy(f'waypoint{i}').remove()
            i += 1

        # Create clean waypoints with no extension strings.
        task_base = self.get_base()
        for idx in range(24):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_DOWN)
            wp.set_parent(task_base)

        self.register_waypoint_ability_end(1, self._make_close(self._block_a))
        self.register_waypoint_ability_end(4, self._open)
        self.register_waypoint_ability_end(6, self._make_close(self._block_b))
        self.register_waypoint_ability_end(9, self._open)
        self.register_waypoint_ability_end(11, self._close_empty)
        self.register_waypoint_ability_end(13, self._open)
        self.register_waypoint_ability_end(15, self._make_close(self._block_b))
        self.register_waypoint_ability_end(18, self._open)
        self.register_waypoint_ability_end(20, self._make_close(self._block_a))
        self.register_waypoint_ability_end(23, self._open)

        for idx in range(24):
            self.register_waypoint_ability_start(idx, self._skip_collisions)

        self.goal_conditions = []

    def _make_close(self, target: Shape):
        def _fn(_waypoint):
            gripper = self.robot.gripper
            done = False
            while not done:
                done = gripper.actuate(0.0, velocity=0.04)
                self.pyrep.step()
            gripper.grasp(target)
        return _fn

    def _close_empty(self, _waypoint):
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()

    def _open(self, _waypoint):
        gripper = self.robot.gripper
        gripper.release()
        done = False
        while not done:
            done = gripper.actuate(1.0, velocity=0.04)
            self.pyrep.step()

    def _skip_collisions(self, waypoint):
        waypoint._ignore_collisions = True

    def init_episode(self, index: int) -> List[str]:
        patch_a = self._targets[index]
        patch_b = self._targets[NUM_TARGETS - index - 1]
        detector_a = self._detectors[index]
        detector_b = self._detectors[NUM_TARGETS - index - 1]

        self.spawn_boundary.clear()
        self.spawn_boundary.sample(self._button, min_distance=0.05)

        x_a, y_a, _ = patch_a.get_position()
        _, _, z = self._block_a.get_position()
        self._block_a.set_position([x_a, y_a, z])

        x_b, y_b, _ = patch_b.get_position()
        _, _, z = self._block_b.get_position()
        self._block_b.set_position([x_b, y_b, z])

        bx, by, _ = self._button.get_position()
        cx, cy = self._cx, self._cy

        # Phase 1: pick block A from patch, place at center
        self._wp(0,  x_a, y_a, Z_APPROACH)
        self._wp(1,  x_a, y_a, Z_GRASP)
        self._wp(2,  x_a, y_a, Z_LIFT)
        self._wp(3,  cx,  cy,  Z_APPROACH)
        self._wp(4,  cx,  cy,  Z_GRASP)

        # Phase 2: pick block B from patch, stack on A at center
        self._wp(5,  x_b, y_b, Z_APPROACH)
        self._wp(6,  x_b, y_b, Z_GRASP)
        self._wp(7,  x_b, y_b, Z_LIFT)
        self._wp(8,  cx,  cy,  Z_STACK_APPROACH)
        self._wp(9,  cx,  cy,  Z_STACK_GRASP)
        self._wp(10, cx,  cy,  Z_LIFT)

        # Phase 3: press button
        self._wp(11, bx,  by,  Z_BUTTON_APPROACH)
        self._wp(12, bx,  by,  Z_BUTTON_PRESS)
        self._wp(13, bx,  by,  Z_LIFT)

        # Phase 4: unstack B, place on patch A
        self._wp(14, cx,  cy,  Z_STACK_APPROACH)
        self._wp(15, cx,  cy,  Z_STACK_GRASP)
        self._wp(16, cx,  cy,  Z_LIFT)
        self._wp(17, x_a, y_a, Z_APPROACH)
        self._wp(18, x_a, y_a, Z_GRASP)

        # Phase 5: pick A from center, place on patch B
        self._wp(19, cx,  cy,  Z_APPROACH)
        self._wp(20, cx,  cy,  Z_GRASP)
        self._wp(21, cx,  cy,  Z_LIFT)
        self._wp(22, x_b, y_b, Z_APPROACH)
        self._wp(23, x_b, y_b, Z_GRASP)

        self.goal_conditions = [
            DetectedTriggerCondition(self._block_a, detector_a, negated=True),
            StackedAtCenterCondition(self._block_a, self._block_b, (cx, cy)),
            JointTriggerCondition(self._button_joint, 0.003),
            DetectedCondition(self._block_b, detector_a),
            DetectedCondition(self._block_a, detector_b),
            NothingGrasped(self.robot.gripper),
        ]
        condition_set = ConditionSet(self.goal_conditions, False)
        self.register_success_conditions([condition_set])

        return [
            'Stack the blocks at the center, press the button, '
            'then swap each block to the other\'s original patch'
        ]

    def _wp(self, idx, x, y, z):
        wp = Dummy(f'waypoint{idx}')
        wp.set_position([x, y, z])
        wp.set_orientation(GRIPPER_DOWN)

    def step(self) -> None:
        pass

    def variation_count(self) -> int:
        return 2

    def is_static_workspace(self) -> bool:
        return True
