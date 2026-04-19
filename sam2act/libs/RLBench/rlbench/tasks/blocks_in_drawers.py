# blocks_in_drawers.py: MemoryBench task -- INCREMENTAL DEBUG VERSION
# blocks_in_drawers.py: Currently testing only Phase 1 (open one drawer).

import math
from itertools import permutations
from typing import List

from pyrep.const import PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    Condition, ConditionSet, DetectedSeveralCondition, NothingGrasped
)

DRAWER_NAMES = ['bottom', 'middle', 'top']
VARIATIONS = list(permutations(DRAWER_NAMES, 2))

# Gripper orientations.
# ABOVE: tool points down (fingers toward -Z). Matches robot rest pose and
# is used for approach/drop/retreat (vertical motion).
# FORWARD: tool points in +X (fingers toward the drawer handle). Used when
# gripping and pulling the handle — gripper approaches the handle from
# the robot side along +X so the fingers close around the handle knob.
GRIPPER_ABOVE = [3.1413, 0.0004, -1.5708]        # vertical, fingers down
GRIPPER_FORWARD = [1.5708, 0.0, 0.0]             # horizontal, tool toward +X
GRIPPER_DRAWER = GRIPPER_FORWARD                 # alias used below
GRIPPER_PICK = [-3.1416, 0.0, -1.5708]           # vertical for block pick

Z_TABLE = 0.752
BLOCK_HALF = 0.02

# Drawer frame top surface Z (frame center 1.0295 + bbox half 0.1883).
Z_DRAWER_TOP = 1.218

# Blocks sit on the table, one on each side of the robot so they are
# clear of the drawer's motion path. X is pulled toward the robot to
# keep them comfortably reachable; Y=+/-0.30 puts them to the sides.
BLOCK_A_XY = (0.20, 0.30)    # +Y side (robot's left)
BLOCK_B_XY = (0.20, -0.30)   # -Y side (robot's right)

Z_HANDLE = {
    'bottom': 0.9153,
    'middle': 1.0354,
    'top': 1.1554,
}
Y_DRAWER = 0.025
X_HANDLE_CLOSED = 0.4294
X_HANDLE_OPEN = 0.2294

DRAWER_TRAVEL = X_HANDLE_OPEN - X_HANDLE_CLOSED
X_INTERIOR = 0.6019 + DRAWER_TRAVEL
Z_INTERIOR = {
    'bottom': 0.8615,
    'middle': 1.0115,
    'top': 1.106,
}

Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF        # block sits on the table
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15     # approach just above the block
Z_LIFT = 1.10                                 # unused now; kept for phase refs

Z_INTERIOR_CLEARANCE = 0.12

BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_COLOR = [1.0, 0.0, 0.0]
BLOCK_MASS = 0.05

# DEBUG MODE: how many phases to enable. Each phase adds one waypoint
# beyond the previous cumulative end index, so we can isolate failures.
# 1  = Phase 1: open drawer 1 (wp0-wp5)              =>  6 waypoints
# 2  = + wp6:  extra retract in -X (clear airspace)  =>  7 waypoints
# 3  = + wp7:  vertical lift to transit Z             =>  8 waypoints
# 4  = + wp8:  transit to above block + reorient ABOVE=>  9 waypoints
# 5  = + wp9:  descend above block (approach Z)       => 10 waypoints
# 6  = + wp10: grasp block                            => 11 waypoints
# 7  = + wp11: lift block to transit Z                => 12 waypoints
# 8  = + wp12: transit above drawer interior          => 13 waypoints
# 9  = + wp13: descend above placement                => 14 waypoints
# 10 = + wp14: place inside drawer                    => 15 waypoints
# 11 = + wp15: retreat up                             => 16 waypoints
DEBUG_PHASE = 11

# Drawer prismatic joint range is [0.0, 0.3] (verified via inspect_scene.py).
# Use 0.2 as the open target (67% of full travel) — leaves headroom and
# matches the X delta we measured (handle moves -0.2 in X when joint=0.2).
DRAWER_OPEN_JOINT = 0.2

# Vertical clearance above the handle for approach/retreat waypoints. Gives
# the planner room to descend onto the handle from above instead of arriving
# sideways and clipping the drawer face.
HANDLE_APPROACH_DZ = 0.10

# Horizontal offset (toward robot, in -X) for the above-handle approach and
# retreat waypoints. Placing them slightly back from the handle keeps the
# gripper body clear of the drawer front face during descent/ascent.
# After shifting the drawer -0.10 in X the anchor is at X=0.329; using
# HANDLE_APPROACH_DX=0 places the approach at the open-handle X (~0.129)
# which keeps the arm in a comfortable reach without crowding the base.
HANDLE_APPROACH_DX = 0.0

# High transit Z: above the drawer frame top (~1.218) by a safe margin.
# Used for initial descent from rest and any lateral transitions so the
# arm stays well clear of the drawer top surface during horizontal moves.
Z_TRANSIT = 1.35


class DrawerClosedCondition(Condition):
    def __init__(self, joint: Joint, threshold: float = 0.04):
        self._joint = joint
        self._threshold = threshold

    def condition_met(self):
        met = math.fabs(self._joint.get_joint_position()) < self._threshold
        return met, False


class RelWaypoint:
    """A waypoint defined relative to a scene object.

    The world position is the anchor object's current position plus a fixed
    offset. This makes the waypoint auto-follow the anchor if the anchor
    moves (e.g., when we randomize drawer/block positions).

    Orientation is a world-frame Euler [x, y, z]. It can either be fixed
    (passed as a list) or copied from the anchor's current orientation
    (pass orientation='anchor').
    """
    def __init__(self, anchor, offset, orientation):
        self.anchor = anchor            # Shape/Dummy with get_position/orientation
        self.offset = list(offset)      # [dx, dy, dz] added to anchor position
        self.orientation = orientation  # [rx, ry, rz] or 'anchor'

    def world_position(self):
        p = self.anchor.get_position()
        return [p[0] + self.offset[0],
                p[1] + self.offset[1],
                p[2] + self.offset[2]]

    def world_orientation(self):
        if self.orientation == 'anchor':
            return list(self.anchor.get_orientation())
        return list(self.orientation)


# Phase definitions: (start_idx, end_idx_exclusive) per phase.
# Currently only Phase 1 is defined (5 waypoints with safe approach/retreat).
# Subsequent phases will be added once Phase 1 is verified.
PHASE_RANGES = [
    # Phase 1: open drawer 1 (6 waypoints, relative to drawer anchor).
    (0, 6),
    # Phase 2: pick block A, place in open drawer 1.
    #   wp6:  extra retract in -X to safe X
    (6, 7),
    #   wp7:  vertical lift to transit Z at safe X
    (7, 8),
    #   wp8:  transit above block + reorient to ABOVE (combined)
    (8, 9),
    #   wp9:  descend above block (approach Z)
    (9, 10),
    #   wp10: grasp block (callback)
    (10, 11),
    #   wp11: lift to transit Z (block in hand)
    (11, 12),
    #   wp12: transit above drawer interior
    (12, 13),
    #   wp13: descend above placement
    (13, 14),
    #   wp14: place inside drawer (callback)
    (14, 15),
    #   wp15: retreat up
    (15, 16),
]

# Intermediate gripper orientation: 45-deg between vertical-down and the
# handle-facing pose. Built in the same Euler branch as the scene anchor
# (handle_ori = [0, pi/2, pi/2]) so the wrist does not flip across branches
# when interpolating between waypoints — only the Y (pitch) component
# changes smoothly from pi/4 to pi/2.
GRIPPER_TILT45 = [0.0, math.pi / 4, math.pi / 2]

# Intermediate transit Z. Just above the drawer frame top (~1.218) with a
# small margin — keeps the tilted-gripper waypoint inside the comfortable
# arm reach envelope.
Z_TILT = 1.25

# Extra retract X (Phase 2 wp7). After the drawer-close retract leaves
# the gripper at approach X (~0.13), we pull back further along -X to a
# position clear of the drawer's open footprint. From here a vertical
# lift gives the wrist plenty of room to rotate without flipping. Tuned
# to sit just in front of the robot base (X_base ~ -0.31).
X_SAFE_RETRACT = 0.13


class BlocksInDrawers(Task):

    def init_task(self) -> None:
        task_base = self.get_base()

        if Object.exists('item'):
            Shape('item').remove()

        # Create two identical RED blocks at runtime.
        self._block_a = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block_a.set_name('block1')
        self._block_a.set_color(BLOCK_COLOR)
        self._block_a.set_parent(task_base)
        # Place at fixed visible position right away.
        self._block_a.set_position([BLOCK_A_XY[0], BLOCK_A_XY[1], Z_BLOCK_GRASP])

        self._block_b = Shape.create(
            type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
            respondable=True, static=False, mass=BLOCK_MASS)
        self._block_b.set_name('block2')
        self._block_b.set_color(BLOCK_COLOR)
        self._block_b.set_parent(task_base)
        self._block_b.set_position([BLOCK_B_XY[0], BLOCK_B_XY[1], Z_BLOCK_GRASP])

        print(f'[blocks_in_drawers] block1 at {self._block_a.get_position()}')
        print(f'[blocks_in_drawers] block2 at {self._block_b.get_position()}')

        self._drawer_joints = {
            name: Joint(f'drawer_joint_{name}') for name in DRAWER_NAMES
        }
        self._drawer_sensors = {
            name: ProximitySensor(f'success_{name}') for name in DRAWER_NAMES
        }
        self._drawer_anchors = {
            name: Dummy(f'waypoint_anchor_{name}') for name in DRAWER_NAMES
        }

        self._drawer_shapes = {
            name: Shape(f'drawer_{name}') for name in DRAWER_NAMES
        }
        self.register_graspable_objects(
            [self._block_a, self._block_b]
            + list(self._drawer_shapes.values()))

        # Delete legacy waypoints from .ttm (may be non-contiguous).
        for i in range(100):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        # Determine waypoint count from DEBUG_PHASE.
        end_idx = PHASE_RANGES[DEBUG_PHASE - 1][1]
        self._n_waypoints = end_idx
        print(f'[blocks_in_drawers] DEBUG_PHASE={DEBUG_PHASE}, creating {end_idx} waypoints')

        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_PICK)
            wp.set_parent(task_base)

        # Register gripper callbacks only for waypoints we actually create.
        # Phase 1 (wp0-wp5, drawer open):
        #   wp3:  close gripper to grip handle
        #   wp4:  linear pull to open, then release
        # Phase 2 (wp6-wp16, pick block, place in drawer):
        #   wp11: close gripper on block (grasp)
        #   wp15: open gripper (release block in drawer)
        callbacks = {
            3: self._close_grip,                        # grip handle
            4: self._open,                              # release after pull
            10: self._make_close(self._block_a),        # grasp block A
            14: self._open,                             # release block in drawer
        }
        for wp_idx, cb in callbacks.items():
            if wp_idx < self._n_waypoints:
                self.register_waypoint_ability_end(wp_idx, cb)

        # All waypoints: ignore collisions for path planning. The drawer pull
        # waypoint uses a linear path so the handle moves cleanly along its
        # prismatic joint axis.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        linear_indices = [4]   # drawer pull only
        for idx in linear_indices:
            if idx < self._n_waypoints:
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

    def _close_grip(self, _waypoint):
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        # Physics-attach the active drawer shape so the linear pull
        # reliably drags the drawer joint from closed to open.
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
        # Identify which waypoint this is by name for logging.
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[blocks_in_drawers] >>> START path to {name}')

    def _set_linear(self, waypoint):
        waypoint._linear_only = True
        waypoint._ignore_collisions = True

    def _wp_arrived(self, waypoint):
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[blocks_in_drawers] <<< ARRIVED at {name}')

    # -- episode setup -------------------------------------------------------

    def init_episode(self, index: int) -> List[str]:
        d1, d2 = VARIATIONS[index]
        self._active_drawer_shape = self._drawer_shapes[d1]

        for name in DRAWER_NAMES:
            self._drawer_joints[name].set_joint_position(
                0.0, disable_dynamics=True)

        x_a, y_a = BLOCK_A_XY
        x_b, y_b = BLOCK_B_XY
        self._block_a.set_position([x_a, y_a, Z_BLOCK_GRASP])
        self._block_b.set_position([x_b, y_b, Z_BLOCK_GRASP])

        z_h1, z_h2 = Z_HANDLE[d1], Z_HANDLE[d2]
        z_i1, z_i2 = Z_INTERIOR[d1], Z_INTERIOR[d2]

        # Phase 1: Open drawer 1 (6 waypoints), defined relative to the
        # drawer's scene anchor. The anchor sits at the closed-handle position
        # so all offsets are in the anchor's local frame (approximately world-
        # aligned here). When the drawer is randomized, the anchor moves and
        # the waypoints follow automatically.
        #
        # Orientation source: the anchor's own orientation gives the "gripper
        # aligned with handle" pose used for grip/pull. Other waypoints use
        # fixed world-frame orientations (GRIPPER_ABOVE, GRIPPER_TILT45).
        #
        # Key offsets (from the closed-handle anchor position):
        #   DRAWER_TRAVEL (=-0.2)         : handle X when drawer fully open
        #   -HANDLE_APPROACH_DX (=-0.1)   : a bit farther back than the open
        #                                   handle — safe approach/retract X
        anchor_d1 = self._drawer_anchors[d1]
        anchor_d1_z = anchor_d1.get_position()[2]
        z_tilt_offset = Z_TILT - anchor_d1_z                # anchor -> transit Z

        # Interior-sensor anchor: sits inside the drawer cavity. When the
        # drawer is pulled open it moves with the drawer, so it is the right
        # anchor for dropping the block inside.
        sensor_d1 = self._drawer_sensors[d1]
        sensor_d1_z = sensor_d1.get_position()[2]
        z_transit_from_sensor = Z_TRANSIT - sensor_d1_z

        # Block anchor: follows the block in world. Offsets are in world frame.
        block_a = self._block_a
        block_a_z = block_a.get_position()[2]               # = Z_BLOCK_GRASP
        z_transit_from_block = Z_TRANSIT - block_a_z

        rel_specs = [
            # -- Phase 1: open drawer 1 (relative to drawer anchor) --
            # wp0: intermediate transit above approach X, mid Z, 45-deg tilted
            (0, RelWaypoint(anchor_d1,
                [DRAWER_TRAVEL - HANDLE_APPROACH_DX, 0.0, z_tilt_offset],
                GRIPPER_TILT45)),
            # wp1: descent to handle height at approach X, gripper facing drawer
            (1, RelWaypoint(anchor_d1,
                [DRAWER_TRAVEL - HANDLE_APPROACH_DX, 0.0, 0.0],
                'anchor')),
            # wp2: forward slide to closed handle (at anchor exactly)
            (2, RelWaypoint(anchor_d1, [0.0, 0.0, 0.0], 'anchor')),
            # wp3: at closed handle — gripper closes on handle (callback)
            (3, RelWaypoint(anchor_d1, [0.0, 0.0, 0.0], 'anchor')),
            # wp4: at open handle — linear pull along drawer joint axis
            (4, RelWaypoint(anchor_d1,
                [DRAWER_TRAVEL, 0.0, 0.0], 'anchor')),
            # wp5: horizontal retract to approach X (same Z, same ori)
            (5, RelWaypoint(anchor_d1,
                [DRAWER_TRAVEL - HANDLE_APPROACH_DX, 0.0, 0.0],
                'anchor')),

            # -- Phase 2: pick block A, place inside drawer 1 --
            # wp6: extra retract in -X to safe X (keep handle orientation).
            #      Pure horizontal translation — no rotation.
            (6, RelWaypoint(anchor_d1,
                [X_SAFE_RETRACT - anchor_d1.get_position()[0], 0.0, 0.0],
                'anchor')),
            # wp7: pure vertical lift to transit Z, keep handle orientation.
            #      Short single-dominant-joint motion — clean curve up.
            (7, RelWaypoint(anchor_d1,
                [X_SAFE_RETRACT - anchor_d1.get_position()[0], 0.0,
                 z_tilt_offset],
                'anchor')),
            # wp8: lateral transit above block AND reorient to ABOVE in one
            #      planned motion. Starts with handle orientation at safe-X,
            #      ends pointing down over the block at transit Z.
            (8, RelWaypoint(block_a,
                [0.0, 0.0, z_transit_from_block],
                GRIPPER_ABOVE)),
            # wp9: descend to approach Z above block
            (9, RelWaypoint(block_a,
                [0.0, 0.0, Z_BLOCK_APPROACH - block_a_z],
                GRIPPER_ABOVE)),
            # wp10: descend to grasp Z (callback closes gripper on block)
            (10, RelWaypoint(block_a, [0.0, 0.0, 0.0], GRIPPER_ABOVE)),
            # wp11: lift to transit Z with block in hand
            (11, RelWaypoint(block_a,
                [0.0, 0.0, z_transit_from_block],
                GRIPPER_ABOVE)),
            # wp12-15: placement waypoints target the drawer interior *after*
            # the drawer has been pulled open. At validate time the drawer is
            # still closed, so we read the sensor's current (closed) X and
            # add DRAWER_TRAVEL to get where the sensor will be when open.
            # wp12: lateral transit above the (soon-to-be-open) interior.
            #      Use a lower Z than the block transit (sensor_z + 0.3) to
            #      keep the pose inside the Panda's reach envelope — gripper
            #      pointing down at X~0.4 is near the reach limit at Z=1.35.
            (12, RelWaypoint(sensor_d1,
                [DRAWER_TRAVEL, 0.0, 0.30],
                GRIPPER_ABOVE)),
            # wp13: descend to just above interior placement
            (13, RelWaypoint(sensor_d1,
                [DRAWER_TRAVEL, 0.0, Z_INTERIOR_CLEARANCE],
                GRIPPER_ABOVE)),
            # wp14: place block at the opened-sensor position (callback releases)
            (14, RelWaypoint(sensor_d1,
                [DRAWER_TRAVEL, 0.0, 0.0],
                GRIPPER_ABOVE)),
            # wp15: retreat up out of drawer cavity (same lowered Z as wp12)
            (15, RelWaypoint(sensor_d1,
                [DRAWER_TRAVEL, 0.0, 0.30],
                GRIPPER_ABOVE)),
        ]

        # Set each waypoint's world pose from its RelWaypoint.
        print(f'[blocks_in_drawers] Setting {self._n_waypoints} waypoints (variation={index}, d1={d1}, d2={d2}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # Minimal success conditions for debug — just need something registered.
        self.goal_conditions = [
            NothingGrasped(self.robot.gripper),
        ]
        condition_set = ConditionSet(self.goal_conditions, order_matters=False)
        self.register_success_conditions([condition_set])

        return [
            'put each block into a separate drawer',
            'place the blocks into different drawers',
            'store each block in its own drawer',
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
