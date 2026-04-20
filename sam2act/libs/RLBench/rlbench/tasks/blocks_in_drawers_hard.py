# blocks_in_drawers_hard.py: MemoryBench task -- 3-block, 3-drawer variant.
# blocks_in_drawers_hard.py: Scales blocks_in_drawers to test deeper D4 recall.

from itertools import permutations
from typing import List

import numpy as np

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
    """Succeeds when the drawer joint position is below `threshold` (m)."""
    def __init__(self, joint: Joint, threshold: float = 0.03):
        self._joint = joint
        self._threshold = threshold

    def condition_met(self):
        met = abs(self._joint.get_joint_position()) < self._threshold
        return met, False


DRAWER_NAMES = ['bottom', 'middle', 'top']
VARIATIONS = list(permutations(DRAWER_NAMES, 3))  # 6 ordered triples

# Gripper orientations.
GRIPPER_HANDLE = [-1.5705, 0.0, -3.1412]
GRIPPER_ABOVE = [-3.1416, 0.0, 1.5708]

# Scene geometry (inherited from blocks_in_drawers.ttm; the _hard .ttm is a
# direct copy, so the same constants apply). Drawer slides along -Y.
DRAWER_TRAVEL = -0.21
HANDLE_APPROACH_DY = -0.042

BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_HALF = BLOCK_SIZE[2] / 2
BLOCK_COLOR = [1.0, 0.0, 0.0]
BLOCK_MASS = 0.05
Z_TABLE = 0.752
Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15

# Fallback spawn positions used at shape creation time; overwritten by the
# zone sampler in init_episode.
BLOCK1_XY = (0.10, -0.30)
BLOCK2_XY = (0.10, -0.40)
BLOCK3_XY = (0.18, -0.35)

Z_TRANSIT = 1.28
Z_INTERIOR_CLEARANCE = 0.10

# Per-block spawn margin. Each block center stays at least
# (block_half + BLOCK_SPAWN_MARGIN) away from its spawn-zone edge so the
# worst-case face-to-face gap between blocks in adjacent zones equals one
# block width.
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


class BlocksInDrawersHard(Task):

    def init_task(self) -> None:
        task_base = self.get_base()

        self._drawer_joints = {
            name: Joint(f'drawer_joint_{name}') for name in DRAWER_NAMES
        }
        self._drawer_shapes = {
            name: Shape(f'drawer_{name}') for name in DRAWER_NAMES
        }
        self._drawer_sensors = {
            name: ProximitySensor(f'success_{name}') for name in DRAWER_NAMES
        }
        self._drawer_anchors = {
            name: Dummy(f'waypoint_anchor_{name}') for name in DRAWER_NAMES
        }

        # Create three identical red blocks at runtime.
        for name in ('block1', 'block2', 'block3'):
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

        self._block3 = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block3.set_name('block3')
        self._block3.set_color(BLOCK_COLOR)
        self._block3.set_parent(task_base)
        self._block3.set_position([BLOCK3_XY[0], BLOCK3_XY[1], Z_BLOCK_GRASP])

        self.register_graspable_objects(
            [self._block1, self._block2, self._block3]
            + list(self._drawer_shapes.values()))

        # Compute spawn zones from the `boundary` shape. The boundary's X
        # axis is split into 3 equal strips, one per block, giving each block
        # a dedicated column on the table. Y range spans the full boundary.
        # A margin of (block_half + BLOCK_SPAWN_MARGIN) keeps each block
        # center clear of its zone edge so blocks never touch.
        boundary = Shape('boundary')
        bpos = boundary.get_position()
        bbox = boundary.get_bounding_box()  # [minx, maxx, miny, maxy, ...]
        x_min = bpos[0] + bbox[0]
        x_max = bpos[0] + bbox[1]
        y_min = bpos[1] + bbox[2]
        y_max = bpos[1] + bbox[3]
        pad = BLOCK_HALF + BLOCK_SPAWN_MARGIN
        n = 3  # one zone per block
        step = (x_max - x_min) / n
        self._spawn_zones = [
            (x_min + i * step + pad,     # x_lo
             x_min + (i + 1) * step - pad,  # x_hi
             y_min + pad,                # y_lo
             y_max - pad)                # y_hi
            for i in range(n)
        ]

        # Delete any legacy waypoints left in the scene.
        for i in range(100):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        # 45 waypoints total. Each phase: 3 open + 4 pick + 1 lift +
        # 4 place + 3 close = 15. Three phases back-to-back.
        self._n_waypoints = 45
        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_ABOVE)
            wp.set_parent(task_base)

        # Per-phase waypoint offsets (base + offset):
        #   +0..+2   open drawer
        #   +3..+6   pick block (grasp at +6)
        #   +7       vertical lift after grasp
        #   +8..+11  place block (release at +10)
        #   +12..+14 close drawer (grip at +13, release at +14)
        # Linear-only paths: +2, +10, +14.
        callbacks = {
            # Phase 1 (base_idx=0): drawer 1 + block1
            1: self._close_grip_drawer1,
            2: self._open,
            6: self._make_close(self._block1),
            10: self._open,
            13: self._close_grip_drawer1,
            14: self._open,
            # Phase 2 (base_idx=15): drawer 2 + block2
            16: self._close_grip_drawer2,
            17: self._open,
            21: self._make_close(self._block2),
            25: self._open,
            28: self._close_grip_drawer2,
            29: self._open,
            # Phase 3 (base_idx=30): drawer 3 + block3
            31: self._close_grip_drawer3,
            32: self._open,
            36: self._make_close(self._block3),
            40: self._open,
            43: self._close_grip_drawer3,
            44: self._open,
        }
        for wp_idx, cb in callbacks.items():
            self.register_waypoint_ability_end(wp_idx, cb)

        # Ignore collisions for every waypoint during planning.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        # Linear-only paths: drawer pulls/pushes and final block drops across
        # all three phases.
        for idx in (2, 10, 14, 17, 25, 29, 32, 40, 44):
            self.register_waypoint_ability_start(idx, self._set_linear)

        self.goal_conditions = []

    # -- gripper callbacks ---------------------------------------------------

    def _make_close(self, target: Shape):
        def _fn(_waypoint):
            gripper = self.robot.gripper
            done = False
            while not done:
                done = gripper.actuate(0.0, velocity=0.04)
                self.pyrep.step()
            gripper.grasp(target)
        return _fn

    def _close_grip_drawer1(self, _waypoint):
        self._close_grip_drawer_by_name(self._d1)

    def _close_grip_drawer2(self, _waypoint):
        self._close_grip_drawer_by_name(self._d2)

    def _close_grip_drawer3(self, _waypoint):
        self._close_grip_drawer_by_name(self._d3)

    def _close_grip_drawer_by_name(self, drawer_name):
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        gripper.grasp(self._drawer_shapes[drawer_name])
        # Pin every non-active drawer's joint by shrinking its range to zero
        # at its current position. This hard-locks it against contact drag
        # from the active drawer. The active drawer keeps its full range.
        self._active_drawer_name = drawer_name
        for name, j in self._drawer_joints.items():
            if name == drawer_name:
                continue
            pos = j.get_joint_position()
            j.set_joint_interval(False, [pos, 0.0])

    def _open(self, _waypoint):
        gripper = self.robot.gripper
        gripper.release()
        done = False
        while not done:
            done = gripper.actuate(1.0, velocity=0.04)
            self.pyrep.step()
        # Restore the original [0, 0.21] range on every drawer so the next
        # grip callback can open/close any of them.
        for j in self._drawer_joints.values():
            j.set_joint_interval(False, [0.0, 0.21])
        self._active_drawer_name = None

    def _skip_collisions(self, waypoint):
        waypoint._ignore_collisions = True
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[blocks_in_drawers_hard] >>> START path to {name}')

    def _set_linear(self, waypoint):
        waypoint._linear_only = True
        waypoint._ignore_collisions = True

    # -- episode setup -------------------------------------------------------

    def init_episode(self, index: int) -> List[str]:
        d1, d2, d3 = VARIATIONS[index]
        self._d1, self._d2, self._d3 = d1, d2, d3
        self._active_drawer_name = None

        # Reset every drawer to closed with its full [0, 0.21] range. Ranges
        # get shrunk to zero on inactive drawers during each grip callback
        # (see _close_grip_drawer_by_name) and restored on release (_open).
        for name in DRAWER_NAMES:
            j = self._drawer_joints[name]
            j.set_joint_interval(False, [0.0, 0.21])
            j.set_joint_position(0.0, disable_dynamics=True)

        # Randomize block positions within fixed spawn zones. Each block is
        # assigned to its own X-strip; block-to-strip assignment is shuffled
        # so blocks appear across zones uniformly over episodes. Blocks are
        # identical so the assignment has no functional effect.
        zones = list(self._spawn_zones)
        np.random.shuffle(zones)
        for block, zone in zip(
                (self._block1, self._block2, self._block3), zones):
            x_lo, x_hi, y_lo, y_hi = zone
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            block.set_position([x, y, Z_BLOCK_GRASP])

        rel_specs = (
            self._drawer_phase_specs(d1, self._block1, base_idx=0)
            + self._drawer_phase_specs(d2, self._block2, base_idx=15)
            + self._drawer_phase_specs(d3, self._block3, base_idx=30)
        )

        print(f'[blocks_in_drawers_hard] Setting {self._n_waypoints} '
              f'waypoints (block1->{d1}, block2->{d2}, block3->{d3}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, '
                  f'{pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # Success: all three blocks inside their target drawers, all three
        # drawers closed at the end, and nothing grasped.
        self.goal_conditions = [
            DetectedCondition(self._block1, self._drawer_sensors[d1]),
            DetectedCondition(self._block2, self._drawer_sensors[d2]),
            DetectedCondition(self._block3, self._drawer_sensors[d3]),
            DrawerClosedCondition(self._drawer_joints[d1]),
            DrawerClosedCondition(self._drawer_joints[d2]),
            DrawerClosedCondition(self._drawer_joints[d3]),
            NothingGrasped(self.robot.gripper),
        ]
        self.register_success_conditions(
            [ConditionSet(self.goal_conditions, order_matters=False)])

        return [
            'put each block in a different drawer and close the drawers',
            'store the three blocks in three separate drawers',
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
            # Open drawer.
            (i + 0, RelWaypoint(anchor,
                [0.0, HANDLE_APPROACH_DY, 0.0], 'anchor')),
            (i + 1, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            (i + 2, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),

            # Pick block.
            (i + 3, RelWaypoint(block,
                [0.0, 0.0, z_transit_from_block], GRIPPER_ABOVE)),
            (i + 4, RelWaypoint(block,
                [0.0, 0.0, Z_BLOCK_APPROACH - block_z], GRIPPER_ABOVE)),
            (i + 5, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            (i + 6, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            # Pure vertical lift to transit Z before any lateral move.
            (i + 7, RelWaypoint(block,
                [0.0, 0.0, z_transit_from_block], GRIPPER_ABOVE)),

            # Place inside open drawer. Sensor is read before the drawer
            # opens, so DRAWER_TRAVEL shifts the target to the open interior.
            (i + 8, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),
            (i + 9, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, Z_INTERIOR_CLEARANCE], GRIPPER_ABOVE)),
            (i + 10, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, 0.0], GRIPPER_ABOVE)),
            (i + 11, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),

            # Close drawer.
            (i + 12, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL + HANDLE_APPROACH_DY, 0.0], 'anchor')),
            (i + 13, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 14, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
        ]

    def _wp(self, idx, x, y, z, orientation):
        wp = Dummy(f'waypoint{idx}')
        wp.set_position([x, y, z])
        wp.set_orientation(orientation)

    def step(self) -> None:
        pass

    def variation_count(self) -> int:
        return len(VARIATIONS)

    def is_static_workspace(self) -> bool:
        return True
