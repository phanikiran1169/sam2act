# blocks_in_two_cabinets.py: 5-block, 5-drawer (2 cabinets, cabinet 2 bottom skipped).
# blocks_in_two_cabinets.py: Cabinet 1 (near) is filled first, then cabinet 2.

import os
from itertools import permutations
from typing import List

import numpy as np

# Debug prints gated on BID_DEBUG=1.
_BID_DEBUG = os.environ.get('BID_DEBUG', '0') not in ('0', '', 'false', 'False')


def _bid_log(msg):
    if _BID_DEBUG:
        print(f'[bitc-debug] {msg}', flush=True)

from pyrep.const import PrimitiveShape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    Condition, ConditionSet, NothingGrasped
)


class DrawerClosedCondition(Condition):
    """Succeeds when the drawer SHAPE is within `threshold` of its
    per-episode closed-y baseline. See blocks_in_drawers.py for the
    full rationale (shape-y is immune to gripper.grasp() reparenting).
    """
    def __init__(self, drawer_shape: Shape, closed_y: float,
                 threshold: float = 0.03):
        self._shape = drawer_shape
        self._closed_y = closed_y
        self._threshold = threshold

    def condition_met(self):
        y = self._shape.get_position()[1]
        met = abs(y - self._closed_y) < self._threshold
        if _BID_DEBUG and not met:
            _bid_log(f'COND DrawerClosed FAIL shape={self._shape.get_name()} '
                     f'y={y:+.4f} baseline={self._closed_y:+.4f} '
                     f'delta={y - self._closed_y:+.4f} threshold={self._threshold}')
        return met, False


class BlocksInDifferentDrawers(Condition):
    """Each of the six blocks sits inside SOME drawer, and all six occupy
    DIFFERENT drawers (one block per drawer slot across both cabinets).
    """
    def __init__(self, blocks: List[Shape], sensors: List[ProximitySensor]):
        self._blocks = list(blocks)
        self._sensors = list(sensors)

    def condition_met(self):
        hits = [
            {i for i, s in enumerate(self._sensors) if s.is_detected(b)}
            for b in self._blocks
        ]
        if any(not h for h in hits):
            if _BID_DEBUG:
                _bid_log(f'COND BlocksInDiff FAIL — block missing from all drawers. hits={hits}')
            return False, False
        # Pairwise disjoint.
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                if not hits[i].isdisjoint(hits[j]):
                    if _BID_DEBUG:
                        _bid_log(f'COND BlocksInDiff FAIL — blocks {i+1}+{j+1} share a drawer. hits={hits}')
                    return False, False
        return True, False


# Two cabinets × 3 drawers each. Cabinet suffix: _1 (near, x≈0.16),
# _2 (far, x≈0.43). Slot key: (cabinet_idx, drawer_name).
DRAWER_LEVELS = ['bottom', 'middle', 'top']
CABINETS = (1, 2)
SLOTS = [(c, d) for c in CABINETS for d in DRAWER_LEVELS]  # 6 slots

# 12 variations = 6 (cab1 drawer order) x 2 (cab2 drawer order over {middle,top}).
# Cabinet 2 skips the bottom drawer (unreachable open-handle pose), so its order
# is over the 2-element set only. Cabinet 1 always runs first.
_CAB1_ORDERS = list(permutations(DRAWER_LEVELS))            # 6
_CAB2_ORDERS = list(permutations(['middle', 'top']))         # 2
VARIATIONS = [(c1, c2) for c1 in _CAB1_ORDERS for c2 in _CAB2_ORDERS]  # 12

# Gripper orientations.
GRIPPER_HANDLE = [-1.5705, 0.0, -3.1412]
GRIPPER_ABOVE = [-3.1416, 0.0, 1.5708]

# Scene geometry. Both cabinets share orientation; drawers slide along -Y.
DRAWER_TRAVEL = -0.21
HANDLE_APPROACH_DY = -0.10

DRAWER_BRAKE_FORCE = 0.0

BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_HALF = BLOCK_SIZE[2] / 2
BLOCK_COLOR = [1.0, 0.0, 0.0]
BLOCK_MASS = 0.05
Z_TABLE = 0.752
Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15

Z_TRANSIT = 1.28
Z_INTERIOR_CLEARANCE = 0.10

# Per-block spawn margin (see blocks_in_drawers_hard.py).
BLOCK_SPAWN_MARGIN = 0.02

# DEBUG: 5 phases — skip cab2/bottom (unreachable open-handle).
# Cabinet 1: bottom, middle, top (3 phases). Cabinet 2: middle, top (2 phases).
N_BLOCKS = 5
N_PHASES = 5
PHASE_WP = 15           # 3 open + 4 pick + 1 lift + 4 place + 3 close
N_WAYPOINTS = N_PHASES * PHASE_WP   # 75


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


def _slot_key(cabinet_idx, drawer_name):
    return f'{drawer_name}_{cabinet_idx}'


class BlocksInTwoCabinets(Task):

    def init_task(self) -> None:
        task_base = self.get_base()

        self._drawer_joints = {}
        self._drawer_shapes = {}
        self._drawer_sensors = {}
        self._drawer_anchors = {}
        for cab in CABINETS:
            for d in DRAWER_LEVELS:
                key = _slot_key(cab, d)
                self._drawer_joints[key] = Joint(f'drawer_joint_{d}_{cab}')
                self._drawer_shapes[key] = Shape(f'drawer_{d}_{cab}')
                self._drawer_sensors[key] = ProximitySensor(f'success_{d}_{cab}')
                self._drawer_anchors[key] = Dummy(f'waypoint_anchor_{d}_{cab}')

        # Soft-braked motor on every drawer joint (see blocks_in_drawers.py).
        for j in self._drawer_joints.values():
            j.set_motor_enabled(True)
            j.set_control_loop_enabled(False)
            j.set_joint_target_velocity(0.0)
            j.set_motor_locked_at_zero_velocity(True)
            j.set_joint_force(DRAWER_BRAKE_FORCE)

        # Create six identical red blocks at runtime.
        self._blocks = []
        for i in range(N_BLOCKS):
            name = f'block{i+1}'
            if Object.exists(name):
                Shape(name).remove()
            blk = Shape.create(
                type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
                respondable=True, static=False, mass=BLOCK_MASS)
            blk.set_name(name)
            blk.set_color(BLOCK_COLOR)
            blk.set_parent(task_base)
            self._blocks.append(blk)

        self.register_graspable_objects(
            self._blocks + list(self._drawer_shapes.values()))

        # Spawn zones from `boundary` shape, split into a 3-column x 2-row
        # grid (6 cells, one per block).
        boundary = Shape('boundary')
        bpos = boundary.get_position()
        bbox = boundary.get_bounding_box()
        x_min = bpos[0] + bbox[0]
        x_max = bpos[0] + bbox[1]
        y_min = bpos[1] + bbox[2]
        y_max = bpos[1] + bbox[3]
        pad = BLOCK_HALF + BLOCK_SPAWN_MARGIN
        n_cols = 3
        n_rows = (N_BLOCKS + n_cols - 1) // n_cols  # 1 row for 3 blocks, 2 for 6
        x_step = (x_max - x_min) / n_cols
        y_step = (y_max - y_min) / n_rows
        # Build full grid then drop the cell farthest from robot when we
        # have one fewer block than cells. Far-from-robot = row 0 (small y),
        # far-from-base in x = col 2 (large x). Robot base is at x=-0.31.
        full = [
            (x_min + col * x_step + pad,
             x_min + (col + 1) * x_step - pad,
             y_min + row * y_step + pad,
             y_min + (row + 1) * y_step - pad)
            for row in range(n_rows) for col in range(n_cols)
        ]
        n_cells = n_rows * n_cols
        if N_BLOCKS == n_cells:
            self._spawn_zones = full
        elif n_cells - N_BLOCKS == 1:
            # Drop (row=0, col=n_cols-1) = far corner from robot.
            drop_idx = 0 * n_cols + (n_cols - 1)
            self._spawn_zones = [z for i, z in enumerate(full) if i != drop_idx]
        else:
            # Generic fallback: drop trailing cells.
            self._spawn_zones = full[:N_BLOCKS]

        # Delete legacy waypoints.
        for i in range(200):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        self._n_waypoints = N_WAYPOINTS
        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_ABOVE)
            wp.set_parent(task_base)

        # Per-phase waypoint offsets (base + offset):
        #   +0..+2   open drawer (grip at +1, release at +2)
        #   +3..+6   pick block (grasp at +6)
        #   +7       vertical lift after grasp
        #   +8..+11  place block (release at +10)
        #   +12..+14 close drawer (grip at +13, release at +14)
        # Linear-only paths: +2 (pull), +10 (final drop), +14 (push).
        callbacks = {}
        for phase in range(N_PHASES):
            base = phase * PHASE_WP
            block = self._blocks[phase]
            callbacks[base + 1]  = self._make_grip_phase(phase)
            callbacks[base + 2]  = self._open
            callbacks[base + 6]  = self._make_close_block(block)
            callbacks[base + 10] = self._open
            callbacks[base + 13] = self._make_grip_phase(phase)
            callbacks[base + 14] = self._open
        for wp_idx, cb in callbacks.items():
            self.register_waypoint_ability_end(wp_idx, cb)

        # Ignore collisions for every waypoint during planning.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        # Linear-only paths.
        linear_offsets = (2, 10, 14)
        for phase in range(N_PHASES):
            for off in linear_offsets:
                self.register_waypoint_ability_start(
                    phase * PHASE_WP + off, self._set_linear)

        self.goal_conditions = []

    # -- gripper / drawer callbacks -----------------------------------

    def _dump_drawer_shapes(self, tag):
        if not _BID_DEBUG:
            return
        try:
            parts = []
            for key, shp in self._drawer_shapes.items():
                parts.append(f'{key}=y{shp.get_position()[1]:+.4f}')
            _bid_log(f'{tag} drawer_shapes: ' + ' '.join(parts))
        except Exception as e:
            _bid_log(f'{tag} dump-err: {e}')

    def _make_close_block(self, target: Shape):
        def _fn(_waypoint):
            gripper = self.robot.gripper
            done = False
            while not done:
                done = gripper.actuate(0.0, velocity=0.04)
                self.pyrep.step()
            gripper.grasp(target)
        return _fn

    def _make_grip_phase(self, phase_idx):
        """Returns a callback that grips the drawer for the given phase."""
        def _fn(_waypoint):
            slot_key = self._phase_slot_keys[phase_idx]
            self._close_grip_drawer_by_key(slot_key)
        return _fn

    def _close_grip_drawer_by_key(self, slot_key):
        self._dump_drawer_shapes(f'GRIP-DRAWER {slot_key} pre')
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        gripper.grasp(self._drawer_shapes[slot_key])
        # Pin every non-active drawer's joint at its current position.
        self._active_slot_key = slot_key
        for k, j in self._drawer_joints.items():
            if k == slot_key:
                continue
            pos = j.get_joint_position()
            j.set_joint_interval(False, [pos, 0.0])
        self._dump_drawer_shapes(f'GRIP-DRAWER {slot_key} post-lock')

    def _open(self, _waypoint):
        self._dump_drawer_shapes('RELEASE pre')
        gripper = self.robot.gripper
        gripper.release()
        done = False
        while not done:
            done = gripper.actuate(1.0, velocity=0.04)
            self.pyrep.step()
        # Restore full range on every drawer.
        for j in self._drawer_joints.values():
            j.set_joint_interval(False, [0.0, 0.21])
        self._active_slot_key = None
        self._dump_drawer_shapes('RELEASE post')

    def _skip_collisions(self, waypoint):
        waypoint._ignore_collisions = True
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[blocks_in_two_cabinets] >>> START path to {name}')

    def _set_linear(self, waypoint):
        waypoint._linear_only = True
        waypoint._ignore_collisions = True

    def _make_realign_pick(self, block, transit_idx, approach_idx,
                           pre_grasp_idx, grasp_idx):
        """Start-of-path callback at the pick transit. Re-reads live block
        pose right before grasp and rewrites the four pick waypoints.
        """
        def _fn(waypoint):
            waypoint._ignore_collisions = True
            bx, by, bz = block.get_position()
            transit_wp = Dummy(f'waypoint{transit_idx}')
            approach_wp = Dummy(f'waypoint{approach_idx}')
            pre_grasp_wp = Dummy(f'waypoint{pre_grasp_idx}')
            grasp_wp = Dummy(f'waypoint{grasp_idx}')
            transit_wp.set_position([bx, by, Z_TRANSIT])
            transit_wp.set_orientation(GRIPPER_ABOVE)
            approach_wp.set_position([bx, by, Z_BLOCK_APPROACH])
            approach_wp.set_orientation(GRIPPER_ABOVE)
            pre_grasp_wp.set_position([bx, by, bz])
            pre_grasp_wp.set_orientation(GRIPPER_ABOVE)
            grasp_wp.set_position([bx, by, bz])
            grasp_wp.set_orientation(GRIPPER_ABOVE)
        return _fn

    # -- episode setup ------------------------------------------------

    def init_episode(self, index: int) -> List[str]:
        cab1_order, cab2_order = VARIATIONS[index]
        self._active_slot_key = None

        # Per-phase target slot: cab1 (3 drawers) -> cab2 (2 drawers,
        # bottom skipped — unreachable open-handle pose).
        self._phase_slot_keys = (
            [_slot_key(1, d) for d in cab1_order]
            + [_slot_key(2, d) for d in cab2_order]
        )
        assert len(self._phase_slot_keys) == N_PHASES, (
            f'phase plan mismatch: {self._phase_slot_keys} vs N_PHASES={N_PHASES}')

        # Reset every drawer to closed with full [0, 0.21] range.
        for j in self._drawer_joints.values():
            j.set_joint_interval(False, [0.0, 0.21])
            j.set_joint_position(0.0, disable_dynamics=True)

        # Closed-y baseline (all drawer shapes share the same closed-y).
        self._closed_y = float(
            list(self._drawer_shapes.values())[0].get_position()[1])
        _bid_log(f'INIT closed_y baseline={self._closed_y:+.4f}')

        # Randomize block positions in spawn zones.
        zones = list(self._spawn_zones)
        np.random.shuffle(zones)
        for blk, zone in zip(self._blocks, zones):
            x_lo, x_hi, y_lo, y_hi = zone
            x = np.random.uniform(x_lo, x_hi)
            y = np.random.uniform(y_lo, y_hi)
            blk.set_position([x, y, Z_BLOCK_GRASP])

        # Build all 6 phase specs.
        rel_specs = []
        for phase in range(N_PHASES):
            slot_key = self._phase_slot_keys[phase]
            rel_specs += self._drawer_phase_specs(
                slot_key, self._blocks[phase], base_idx=phase * PHASE_WP)

        print(f'[blocks_in_two_cabinets] Setting {self._n_waypoints} '
              f'waypoints (phases -> {self._phase_slot_keys}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, '
                  f'{pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # Re-read live block pose right before grasp, in case prior phases
        # nudged the block.
        for phase in range(N_PHASES):
            base = phase * PHASE_WP
            self.register_waypoint_ability_start(
                base + 3,
                self._make_realign_pick(self._blocks[phase],
                                        transit_idx=base + 3,
                                        approach_idx=base + 4,
                                        pre_grasp_idx=base + 5,
                                        grasp_idx=base + 6))

        # Success: each block in some drawer, all six in DIFFERENT drawers,
        # all six drawers closed, nothing grasped.
        drawer_closed_conds = [
            DrawerClosedCondition(self._drawer_shapes[k], self._closed_y)
            for k in self._drawer_shapes
        ]
        self.goal_conditions = [
            BlocksInDifferentDrawers(
                self._blocks, list(self._drawer_sensors.values())),
            *drawer_closed_conds,
            NothingGrasped(self.robot.gripper),
        ]
        self.register_success_conditions(
            [ConditionSet(self.goal_conditions, order_matters=False)])

        return [
            'put each of the five blocks in a different drawer and close the drawers',
            'store the five blocks in five separate drawers',
            'place the blocks in different drawers and close each one',
        ]

    def _drawer_phase_specs(self, slot_key, block, base_idx):
        """Build 15-waypoint (open + pick + lift + place + close) spec list
        starting at base_idx for the given drawer slot."""
        anchor = self._drawer_anchors[slot_key]
        sensor = self._drawer_sensors[slot_key]

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

            # Place inside open drawer.
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
        if _BID_DEBUG and os.environ.get('BID_DEBUG_STEP', '0') not in ('0', '', 'false', 'False'):
            try:
                bp = [b.get_position() for b in self._blocks]
                parts = ' '.join(
                    f'b{i+1}=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]'
                    for i, p in enumerate(bp))
                _bid_log(f'STEP blocks: {parts}')
            except Exception:
                pass

    def variation_count(self) -> int:
        return len(VARIATIONS)

    def is_static_workspace(self) -> bool:
        return True
