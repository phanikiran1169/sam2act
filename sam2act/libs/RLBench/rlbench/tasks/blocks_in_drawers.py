# blocks_in_drawers.py: MemoryBench task -- uses reopen_drawer scene layout.
# blocks_in_drawers.py: Open one closed drawer, pick one block, place inside.

import math
from typing import List

from pyrep.const import PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    ConditionSet, DetectedCondition, NothingGrasped
)

DRAWER_NAMES = ['bottom', 'middle', 'top']

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

# Block spawn (fixed for now, inside the 'boundary' shape at y=-0.316).
BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_HALF = BLOCK_SIZE[2] / 2
BLOCK_COLOR = [1.0, 0.0, 0.0]
BLOCK_MASS = 0.05
Z_TABLE = 0.752
Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15
BLOCK_XY = (0.253, -0.316)       # boundary center

# Transit height when moving laterally with the block.
Z_TRANSIT = 1.20

# Placement inside the drawer cavity: above the success sensor.
Z_INTERIOR_CLEARANCE = 0.10

# Active drawer for the current episode (fixed to bottom for now).
ACTIVE_DRAWER = 'bottom'


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
        self._drawer_sensors = {
            name: ProximitySensor(f'success_{name}') for name in DRAWER_NAMES
        }
        self._drawer_anchors = {
            name: Dummy(f'waypoint_anchor_{name}') for name in DRAWER_NAMES
        }

        # Create one block at runtime.
        if Object.exists('block1'):
            Shape('block1').remove()
        self._block = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block.set_name('block1')
        self._block.set_color(BLOCK_COLOR)
        self._block.set_parent(task_base)
        self._block.set_position([BLOCK_XY[0], BLOCK_XY[1], Z_BLOCK_GRASP])

        # Both block and drawer shapes are graspable. When a waypoint runs
        # with a close_gripper() extension the engine grasps all registered
        # graspables — but we use python callbacks, so we control grasp
        # explicitly via self._close_grip_drawer / _make_close_block.
        self.register_graspable_objects(
            [self._block] + list(self._drawer_shapes.values()))

        # Delete any legacy waypoints left in the scene.
        for i in range(100):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        # 11 waypoints total: 3 open-drawer + 4 pick + 4 place.
        self._n_waypoints = 11
        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_ABOVE)
            wp.set_parent(task_base)

        # Callbacks.
        #   wp1: grip handle (close + grasp drawer shape)
        #   wp2: release (after linear pull opens drawer)
        #   wp6: grasp block
        #   wp9: release block inside drawer
        callbacks = {
            1: self._close_grip_drawer,
            2: self._open,
            6: self._make_close(self._block),
            9: self._open,
        }
        for wp_idx, cb in callbacks.items():
            self.register_waypoint_ability_end(wp_idx, cb)

        # Ignore collisions for every waypoint during planning.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        # Linear-only path for the drawer pull.
        self.register_waypoint_ability_start(2, self._set_linear)

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

    def _close_grip_drawer(self, _waypoint):
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        if getattr(self, '_active_drawer_shape', None) is not None:
            gripper.grasp(self._active_drawer_shape)

    def _open(self, _waypoint):
        gripper = self.robot.gripper
        gripper.release()
        done = False
        while not done:
            done = gripper.actuate(1.0, velocity=0.04)
            self.pyrep.step()

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
        d1 = ACTIVE_DRAWER
        self._active_drawer_shape = self._drawer_shapes[d1]

        # Reset every drawer to closed.
        for name in DRAWER_NAMES:
            self._drawer_joints[name].set_joint_position(
                0.0, disable_dynamics=True)

        # Place block at fixed spawn.
        self._block.set_position([BLOCK_XY[0], BLOCK_XY[1], Z_BLOCK_GRASP])

        anchor = self._drawer_anchors[d1]
        sensor = self._drawer_sensors[d1]
        block = self._block

        # Offsets to reach block/sensor transit height.
        block_z = block.get_position()[2]
        z_transit_from_block = Z_TRANSIT - block_z
        sensor_z = sensor.get_position()[2]
        z_transit_from_sensor = Z_TRANSIT - sensor_z

        rel_specs = [
            # -- Phase 1: open drawer (anchor = closed-handle pose) --
            # wp0: pre-approach in -Y, at handle Z, handle orientation
            (0, RelWaypoint(anchor,
                [0.0, HANDLE_APPROACH_DY, 0.0], 'anchor')),
            # wp1: at handle (callback: close gripper + grasp drawer shape)
            (1, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            # wp2: linear pull along drawer axis -> drawer opens. Callback
            #      releases the drawer shape at the end.
            (2, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),

            # -- Phase 2: pick block --
            # wp3: transit above block (combined lateral + reorient to ABOVE)
            (3, RelWaypoint(block,
                [0.0, 0.0, z_transit_from_block], GRIPPER_ABOVE)),
            # wp4: descend to approach Z above block
            (4, RelWaypoint(block,
                [0.0, 0.0, Z_BLOCK_APPROACH - block_z], GRIPPER_ABOVE)),
            # wp5: descend to grasp Z (no callback — callback fires at wp6)
            (5, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            # wp6: at grasp Z (callback: close gripper + grasp block)
            (6, RelWaypoint(block, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),

            # -- Phase 3: place in open drawer --
            # Sensor moves with drawer_top (the drawer body). At episode
            # init the drawer is closed, so sensor.get_position() gives the
            # closed position. Apply DRAWER_TRAVEL along Y to target the
            # open-drawer interior. Wp positions are computed once and
            # do not auto-update, which is what we want.
            # wp7: lift block to transit, move laterally above open drawer
            (7, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),
            # wp8: descend to just above interior
            (8, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, Z_INTERIOR_CLEARANCE], GRIPPER_ABOVE)),
            # wp9: at sensor position (callback: release block)
            (9, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, 0.0], GRIPPER_ABOVE)),
            # wp10: retreat up
            (10, RelWaypoint(sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_sensor], GRIPPER_ABOVE)),
        ]

        print(f'[blocks_in_drawers] Setting {self._n_waypoints} waypoints '
              f'(drawer={d1}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, '
                  f'{pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # Success: block detected inside the active drawer's sensor.
        self.goal_conditions = [
            DetectedCondition(self._block, self._drawer_sensors[d1]),
            NothingGrasped(self.robot.gripper),
        ]
        self.register_success_conditions(
            [ConditionSet(self.goal_conditions, order_matters=False)])

        return [
            f'put the block in the {d1} drawer',
            f'open the {d1} drawer and place the block inside',
            f'store the block in the {d1} drawer',
        ]

    def _wp(self, idx, x, y, z, orientation):
        wp = Dummy(f'waypoint{idx}')
        wp.set_position([x, y, z])
        wp.set_orientation(orientation)

    def step(self) -> None:
        pass

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self) -> bool:
        return True
