# blocks_in_drawers.py: MemoryBench task -- uses reopen_drawer scene layout.
# blocks_in_drawers.py: Open one closed drawer, pick one block, place inside.

import os
from itertools import permutations
from typing import List

import numpy as np

# Debug prints gated on BID_DEBUG=1. Logs block/sensor/drawer state at every
# key callback so failure modes (placement drift, late drawer re-open, passive
# drawer drift) can be diagnosed from text alone.
_BID_DEBUG = os.environ.get('BID_DEBUG', '0') not in ('0', '', 'false', 'False')


def _bid_log(msg):
    if _BID_DEBUG:
        print(f'[bid-debug] {msg}', flush=True)

from pyrep.const import PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    Condition, ConditionSet, DetectedCondition, NothingGrasped
)


class DrawerClosedCondition(Condition):
    """Succeeds when the drawer SHAPE is within `threshold` of its
    per-episode closed-y baseline.

    Reads the drawer shape's y directly rather than the joint position,
    because `gripper.grasp()` reparents the drawer shape off the joint's
    kinematic chain (shape becomes a child of the gripper). After grasp,
    the joint's internal DOF no longer tracks the shape and
    `get_joint_position()` returns stale ~0 even when the drawer is wide
    open. Shape-y is the ground-truth drawer position.

    The `closed_y` baseline is captured per-episode (at init, right after
    `set_joint_position(0.0)` puts the drawer in its closed state). This
    is frame-agnostic: any offset between task/world/parent frames
    cancels out in `y - closed_y`.
    """
    def __init__(self, drawer_shape: Shape, closed_y: float,
                 threshold: float = 0.03):
        self._shape = drawer_shape
        self._closed_y = closed_y
        self._threshold = threshold

    def condition_met(self):
        y = self._shape.get_position()[1]
        met = abs(y - self._closed_y) < self._threshold
        return met, False


class BlocksInDifferentDrawers(Condition):
    """Task goal: each block sits inside SOME drawer, and the two blocks
    are in two DIFFERENT drawers. `d1`/`d2` are demo-generation scaffolding,
    not part of the task identity — the language goal says "put each block
    in a different drawer" with no specific drawer requirement.
    """
    def __init__(self, block1: Shape, block2: Shape,
                 sensors: List[ProximitySensor]):
        self._b1, self._b2 = block1, block2
        self._sensors = list(sensors)

    def condition_met(self):
        b1_hits = {i for i, s in enumerate(self._sensors) if s.is_detected(self._b1)}
        b2_hits = {i for i, s in enumerate(self._sensors) if s.is_detected(self._b2)}
        if not b1_hits or not b2_hits:
            return False, False
        # Each block must be in some drawer, and the two sets must be
        # disjoint. This forbids the case where block2 sits in one of the
        # drawers block1 also occupies (multi-sensor hit on overlapping
        # geometry). Stricter than `any(i!=j)`.
        met = b1_hits.isdisjoint(b2_hits)
        return met, False

DRAWER_NAMES = ['bottom', 'middle', 'top']
VARIATIONS = list(permutations(DRAWER_NAMES, 2))  # 6 ordered pairs

# Gripper orientations.
# HANDLE: tool points along +Y (toward drawer face). Fingers close around
# the handle knob. Used for open-drawer phase. Matches the anchor ori.
# ABOVE: tool points straight down (fingers toward -Z). Used for block
# pick and placement inside the drawer cavity.
GRIPPER_HANDLE = [-1.5705, 0.0, -3.1412]
GRIPPER_ABOVE = [-3.1416, 0.0, 1.5708]

# Scene geometry — measured from blocks_in_drawers.ttm via inspect_scene.py.
# Drawer slides along -Y: closed handle at Y=+0.096, open handle at Y=-0.114.
DRAWER_TRAVEL = -0.21            # closed -> open, in -Y
HANDLE_APPROACH_DY = -0.042      # pre-approach offset from closed handle

# Joint motor force used to hold inactive drawers against arm-brush drift.
# Enabled in init_task. Tuning knob: too low -> drawers still drift; too
# high -> the robot cannot pull the target drawer open. Starting at 3 N.
DRAWER_BRAKE_FORCE = 3.0

# Block properties. Blocks are identical red cubes; spawn positions are
# sampled per-episode from zones derived from the `boundary` shape.
BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_HALF = BLOCK_SIZE[2] / 2
BLOCK_COLOR = [1.0, 0.0, 0.0]
BLOCK_MASS = 0.05
Z_TABLE = 0.752
Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15

# Fallback spawn positions used at shape creation time; overwritten by the
# zone sampler in init_episode.
BLOCK1_XY = (0.15, -0.35)
BLOCK2_XY = (0.30, -0.35)

# Transit height when moving laterally with the block.
Z_TRANSIT = 1.28

# Placement inside the drawer cavity: above the success sensor.
Z_INTERIOR_CLEARANCE = 0.10

# Per-block spawn margin (see blocks_in_drawers_hard.py). Each block center
# stays at least (block_half + BLOCK_SPAWN_MARGIN) from its spawn-zone edge.
BLOCK_SPAWN_MARGIN = 0.02



class RelWaypoint:
    """Waypoint defined as (anchor_obj, local_offset, orientation)."""
    def __init__(self, anchor, offset, orientation):
        self.anchor = anchor
        self.offset = list(offset)
        self.orientation = orientation

    def world_position(self):
        p = self.anchor.get_position()
        return [p[0] + self.offset[0],
                p[1] + self.offset[1],
                p[2] + self.offset[2]]

    def world_orientation(self):
        if self.orientation == 'anchor':
            return list(self.anchor.get_orientation())
        return list(self.orientation)


class BlocksInDrawers(Task):

    def init_task(self) -> None:
        task_base = self.get_base()

        self._drawer_joints = {
            name: Joint(f'drawer_joint_{name}') for name in DRAWER_NAMES
        }
        self._drawer_shapes = {
            name: Shape(f'drawer_{name}') for name in DRAWER_NAMES
        }

        # Turn each drawer joint into a soft-braked velocity-locked motor so
        # inactive drawers resist drift from arm brushes. Probed defaults from
        # the .ttm are FORCE mode, motor disabled — so the joint was purely
        # passive and any contact force moved it freely. With the motor on,
        # target velocity 0, and locked-at-zero-velocity, the joint actively
        # resists motion up to DRAWER_BRAKE_FORCE. A firm grasp pull exceeds
        # this and opens the drawer; a passing arm brush does not.
        for j in self._drawer_joints.values():
            j.set_motor_enabled(True)
            j.set_control_loop_enabled(False)
            j.set_joint_target_velocity(0.0)
            j.set_motor_locked_at_zero_velocity(True)
            j.set_joint_force(DRAWER_BRAKE_FORCE)
        self._drawer_sensors = {
            name: ProximitySensor(f'success_{name}') for name in DRAWER_NAMES
        }
        self._drawer_anchors = {
            name: Dummy(f'waypoint_anchor_{name}') for name in DRAWER_NAMES
        }

        # Create two red blocks at runtime.
        for name in ('block1', 'block2'):
            if Object.exists(name):
                Shape(name).remove()
        self._block1 = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block1.set_name('block1')
        self._block1.set_color(BLOCK_COLOR)
        self._block1.set_parent(task_base)
        self._block1.set_position([BLOCK1_XY[0], BLOCK1_XY[1], Z_BLOCK_GRASP])

        self._block2 = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block2.set_name('block2')
        self._block2.set_color(BLOCK_COLOR)
        self._block2.set_parent(task_base)
        self._block2.set_position([BLOCK2_XY[0], BLOCK2_XY[1], Z_BLOCK_GRASP])

        self.register_graspable_objects(
            [self._block1, self._block2]
            + list(self._drawer_shapes.values()))

        # Compute spawn zones from the `boundary` shape: X axis split into 2
        # equal strips, one per block; Y spans the full boundary. A margin of
        # (block_half + BLOCK_SPAWN_MARGIN) per zone edge keeps blocks clear.
        boundary = Shape('boundary')
        bpos = boundary.get_position()
        bbox = boundary.get_bounding_box()
        x_min = bpos[0] + bbox[0]
        x_max = bpos[0] + bbox[1]
        y_min = bpos[1] + bbox[2]
        y_max = bpos[1] + bbox[3]
        pad = BLOCK_HALF + BLOCK_SPAWN_MARGIN
        n = 2
        step = (x_max - x_min) / n
        self._spawn_zones = [
            (x_min + i * step + pad,
             x_min + (i + 1) * step - pad,
             y_min + pad,
             y_max - pad)
            for i in range(n)
        ]

        # Delete any legacy waypoints left in the scene.
        for i in range(100):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        # 30 waypoints total. Each phase: 3 open + 4 pick + 1 lift +
        # 4 place + 3 close = 15. Two phases back-to-back.
        self._n_waypoints = 30
        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_ABOVE)
            wp.set_parent(task_base)

        # Per-phase waypoint index layout (base + offset):
        #   +0..+2   open drawer
        #   +3..+6   pick block
        #   +7       vertical lift after grasp
        #   +8..+11  place block
        #   +12..+14 close drawer
        # Callbacks:
        #   +1  grip drawer handle (open)
        #   +2  release after pull
        #   +6  grasp block
        #   +10 release block in drawer
        #   +13 grip drawer handle (close)
        #   +14 release after push
        # Linear paths: +2 (pull open) and +14 (push closed).
        callbacks = {
            1: self._close_grip_drawer1,
            2: self._open,
            6: self._make_close(self._block1),
            10: self._open,
            13: self._close_grip_drawer1,
            14: self._open,
            # Phase 2 offsets: add 15.
            16: self._close_grip_drawer2,
            17: self._open,
            21: self._make_close(self._block2),
            25: self._open,
            28: self._close_grip_drawer2,
            29: self._open,
        }
        for wp_idx, cb in callbacks.items():
            self.register_waypoint_ability_end(wp_idx, cb)

        # Ignore collisions for every waypoint during planning.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        # Linear-only paths:
        #   +2, +14 : drawer open pull, drawer close push
        #   +10     : final vertical drop to release block inside drawer
        # Phase-2 offsets: add 15.
        for idx in (2, 10, 14, 17, 25, 29):
            self.register_waypoint_ability_start(idx, self._set_linear)

        self.goal_conditions = []

    # -- gripper callbacks ---------------------------------------------------

    def _dump_drawer_joints(self, tag):
        if not _BID_DEBUG:
            return
        parts = [f'{n}={self._drawer_joints[n].get_joint_position():+.4f}'
                 for n in DRAWER_NAMES]
        _bid_log(f'{tag} drawer_joints: ' + ' '.join(parts))
        # Also dump drawer SHAPE positions to catch desync between joint
        # reading (pinned by set_joint_interval) and physical shape motion.
        try:
            shape_parts = []
            for n in DRAWER_NAMES:
                sp = self._drawer_shapes[n].get_position()
                shape_parts.append(f'{n}=y{sp[1]:+.4f}')
            _bid_log(f'{tag} drawer_shapes: ' + ' '.join(shape_parts))
        except Exception as e:
            _bid_log(f'{tag} shape-dump err: {e}')

    def _dump_block_vs_sensor(self, tag, block, drawer_name):
        if not _BID_DEBUG:
            return
        bp = np.array(block.get_position())
        sensor = self._drawer_sensors[drawer_name]
        sp = np.array(sensor.get_position())
        bbox = sensor.get_bounding_box()  # local frame [xmn,xmx,ymn,ymx,zmn,zmx]
        detected = sensor.is_detected(block)
        dx, dy, dz = (bp - sp).tolist()
        _bid_log(
            f'{tag} block={block.get_name()} drawer={drawer_name} '
            f'block_pos=[{bp[0]:.4f},{bp[1]:.4f},{bp[2]:.4f}] '
            f'sensor_pos=[{sp[0]:.4f},{sp[1]:.4f},{sp[2]:.4f}] '
            f'delta=[{dx:+.4f},{dy:+.4f},{dz:+.4f}] '
            f'bbox_half=[x<={max(abs(bbox[0]),abs(bbox[1])):.4f},'
            f'y<={max(abs(bbox[2]),abs(bbox[3])):.4f},'
            f'z<={max(abs(bbox[4]),abs(bbox[5])):.4f}] '
            f'is_detected={detected}'
        )

    def _make_close(self, target: Shape):
        def _fn(_waypoint):
            gripper = self.robot.gripper
            _bid_log(f'GRIP-CLOSE target={target.get_name()} '
                     f'target_pos={[round(v,4) for v in target.get_position()]} '
                     f'ee_pos={[round(v,4) for v in gripper.get_position()]}')
            done = False
            while not done:
                done = gripper.actuate(0.0, velocity=0.04)
                self.pyrep.step()
            gripper.grasp(target)
            _bid_log(f'GRIP-CLOSE done target={target.get_name()} '
                     f'target_pos_after={[round(v,4) for v in target.get_position()]}')
        return _fn

    def _close_grip_drawer1(self, _waypoint):
        self._close_grip_drawer_by_name(self._d1)

    def _close_grip_drawer2(self, _waypoint):
        self._close_grip_drawer_by_name(self._d2)

    def _close_grip_drawer_by_name(self, drawer_name):
        gripper = self.robot.gripper
        _bid_log(f'DRAWER-GRIP drawer={drawer_name} '
                 f'ee_pos={[round(v,4) for v in gripper.get_position()]}')
        self._dump_drawer_joints('DRAWER-GRIP before-actuate')
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        self._dump_drawer_joints('DRAWER-GRIP after-actuate')
        gripper.grasp(self._drawer_shapes[drawer_name])
        # Pin every non-active drawer's joint by shrinking its range to zero
        # at its current position. Prevents contact drag from the active
        # drawer. The active drawer keeps its full range of motion.
        self._active_drawer_name = drawer_name
        for name, j in self._drawer_joints.items():
            if name == drawer_name:
                continue
            pos = j.get_joint_position()
            j.set_joint_interval(False, [pos, 0.0])
        _bid_log(f'DRAWER-GRIP locked_inactive active={drawer_name}')
        self._dump_drawer_joints('DRAWER-GRIP after-lock')

    def _open(self, _waypoint):
        gripper = self.robot.gripper
        # At release time, log block-vs-sensor state for BOTH blocks so we can
        # tell whether a block just placed is inside the target sensor volume.
        _bid_log('RELEASE fired')
        self._dump_drawer_joints('RELEASE pre-release')
        try:
            self._dump_block_vs_sensor('RELEASE(block1)', self._block1, self._d1)
            self._dump_block_vs_sensor('RELEASE(block2)', self._block2, self._d2)
        except Exception as e:
            _bid_log(f'RELEASE state-dump err: {e}')
        gripper.release()
        done = False
        while not done:
            done = gripper.actuate(1.0, velocity=0.04)
            self.pyrep.step()
        # Restore the full [0, 0.21] range on every drawer so the next grip
        # callback can freely open or close any of them.
        for j in self._drawer_joints.values():
            j.set_joint_interval(False, [0.0, 0.21])
        self._active_drawer_name = None
        self._dump_drawer_joints('RELEASE after')
        try:
            self._dump_block_vs_sensor('RELEASE(block1) after', self._block1, self._d1)
            self._dump_block_vs_sensor('RELEASE(block2) after', self._block2, self._d2)
        except Exception as e:
            _bid_log(f'RELEASE after state-dump err: {e}')

    def _skip_collisions(self, waypoint):
        waypoint._ignore_collisions = True
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[blocks_in_drawers] >>> START path to {name}')

    def _set_linear(self, waypoint):
        waypoint._linear_only = True
        waypoint._ignore_collisions = True

    # -- episode setup -------------------------------------------------------

    def init_episode(self, index: int) -> List[str]:
        d1, d2 = VARIATIONS[index]
        self._d1, self._d2 = d1, d2
        self._active_drawer_name = None

        # Reset every drawer to closed with its full [0, 0.21] range. Ranges
        # shrink to zero on inactive drawers during each grip callback and
        # are restored on release.
        for name in DRAWER_NAMES:
            j = self._drawer_joints[name]
            j.set_joint_interval(False, [0.0, 0.21])
            j.set_joint_position(0.0, disable_dynamics=True)

        # Capture the closed-y baseline for DrawerClosedCondition. All three
        # drawer shapes are identical and just reset to closed, so one
        # sample is enough. Using this baseline makes the condition
        # frame-agnostic: any task/world/parent offset cancels in the
        # (y - baseline) delta.
        self._closed_y = float(
            self._drawer_shapes[DRAWER_NAMES[0]].get_position()[1])
        _bid_log(f'INIT closed_y baseline={self._closed_y:+.4f}')

        # Randomize block positions within fixed spawn zones. Each block is
        # assigned to its own X-strip; block-to-strip assignment is shuffled.
        zones = list(self._spawn_zones)
        np.random.shuffle(zones)
        for block, zone in zip((self._block1, self._block2), zones):
            x_lo, x_hi, y_lo, y_hi = zone
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            block.set_position([x, y, Z_BLOCK_GRASP])

        rel_specs = (
            self._drawer_phase_specs(d1, self._block1, base_idx=0)
            + self._drawer_phase_specs(d2, self._block2, base_idx=15)
        )

        _bid_log(f'INIT variation index={index} d1={d1} d2={d2}')
        _bid_log(f'INIT block1_pos={[round(v,4) for v in self._block1.get_position()]} '
                 f'block2_pos={[round(v,4) for v in self._block2.get_position()]}')
        for name in DRAWER_NAMES:
            s = self._drawer_sensors[name]
            _bid_log(f'INIT sensor {name} pos={[round(v,4) for v in s.get_position()]} '
                     f'bbox={[round(v,4) for v in s.get_bounding_box()]}')
        self._dump_drawer_joints('INIT')

        print(f'[blocks_in_drawers] Setting {self._n_waypoints} waypoints '
              f'(block1->{d1}, block2->{d2}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, '
                  f'{pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # Success: each block ends up inside some drawer and the two blocks
        # occupy DIFFERENT drawers. d1/d2 are demo-generation scaffolding,
        # not part of task identity — the language goal doesn't name drawers.
        # All three drawers must be closed at the end (no drawer left open),
        # and nothing grasped.
        self.goal_conditions = [
            BlocksInDifferentDrawers(
                self._block1, self._block2,
                list(self._drawer_sensors.values())),
            DrawerClosedCondition(self._drawer_shapes['bottom'], self._closed_y),
            DrawerClosedCondition(self._drawer_shapes['middle'], self._closed_y),
            DrawerClosedCondition(self._drawer_shapes['top'], self._closed_y),
            NothingGrasped(self.robot.gripper),
        ]
        self.register_success_conditions(
            [ConditionSet(self.goal_conditions, order_matters=False)])

        return [
            'put each block in a different drawer and close the drawers',
            'store the two blocks in two separate drawers',
            'place the blocks in different drawers and close each one',
        ]

    def _drawer_phase_specs(self, drawer_name, block, base_idx):
        """Build 15-waypoint (open + pick + lift + place + close) spec list
        starting at base_idx."""
        anchor = self._drawer_anchors[drawer_name]
        sensor = self._drawer_sensors[drawer_name]

        block_z = block.get_position()[2]
        z_transit_from_block = Z_TRANSIT - block_z
        sensor_z = sensor.get_position()[2]
        z_transit_from_sensor = Z_TRANSIT - sensor_z

        i = base_idx
        return [
            # Open drawer (anchor = closed-handle pose).
            # +0: pre-approach in -Y, handle orientation
            (i + 0, RelWaypoint(anchor,
                [0.0, HANDLE_APPROACH_DY, 0.0], 'anchor')),
            # +1: at handle (callback: close + grasp drawer shape)
            (i + 1, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            # +2: linear pull -Y (drawer opens, callback releases)
            (i + 2, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),

            # Pick block.
            # +3: transit above block (combined lateral + reorient to ABOVE)
            (i + 3, RelWaypoint(block,
                [0.0, 0.0, z_transit_from_block], GRIPPER_ABOVE)),
            # +4: descend to approach Z
            (i + 4, RelWaypoint(block,
                [0.0, 0.0, Z_BLOCK_APPROACH - block_z], GRIPPER_ABOVE)),
            # +5: descend to grasp Z (pre-grasp pose)
            (i + 5, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            # +6: at grasp Z (callback: close + grasp block)
            (i + 6, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            # +7: pure vertical lift straight up from the block position
            #     to transit Z. Keeps the block clear of any open drawer
            #     body before the lateral move begins.
            (i + 7, RelWaypoint(block,
                [0.0, 0.0, z_transit_from_block], GRIPPER_ABOVE)),

            # Place inside open drawer. Sensor lives on the drawer body, so
            # at init_episode time (drawer closed) its position is the
            # closed-sensor pose. Shift -Y by DRAWER_TRAVEL to target the
            # open-drawer interior.
            # +8: lateral transit above open drawer interior (at transit Z)
            (i + 8, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),
            # +9: descend to just above interior
            (i + 9, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, Z_INTERIOR_CLEARANCE], GRIPPER_ABOVE)),
            # +10: at sensor position (callback: release block)
            (i + 10, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, 0.0], GRIPPER_ABOVE)),
            # +11: retreat up
            (i + 11, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),

            # Close drawer (anchor = closed-handle pose).
            # Open-handle position is anchor + [0, DRAWER_TRAVEL, 0].
            # Pre-approach the open handle from -Y side, handle orientation.
            # +12: pre-approach at open handle Y minus HANDLE_APPROACH_DY
            (i + 12, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL + HANDLE_APPROACH_DY, 0.0], 'anchor')),
            # +13: at open handle (callback: close + grasp drawer shape)
            (i + 13, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            # +14: linear push back to closed handle (callback: release)
            (i + 14, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
        ]

    def _wp(self, idx, x, y, z, orientation):
        wp = Dummy(f'waypoint{idx}')
        wp.set_position([x, y, z])
        wp.set_orientation(orientation)

    def step(self) -> None:
        # Per-step drift tracker (BID_DEBUG_STEP=1) — noisy, off by default.
        if _BID_DEBUG and os.environ.get('BID_DEBUG_STEP', '0') not in ('0', '', 'false', 'False'):
            self._dump_drawer_joints('STEP')
            try:
                self._dump_block_vs_sensor('STEP(block1)', self._block1, self._d1)
                self._dump_block_vs_sensor('STEP(block2)', self._block2, self._d2)
            except Exception:
                pass

    def variation_count(self) -> int:
        return len(VARIATIONS)

    def is_static_workspace(self) -> bool:
        return True
