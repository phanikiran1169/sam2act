# inspect_and_pick.py: MemoryBench task -- inspect each drawer, press a
# inspect_and_pick.py: button, then retrieve a target block into a bin. v0.

import os
from itertools import permutations
from typing import List

import numpy as np

# Debug prints gated on IAP_DEBUG=1. Mirrors blocks_in_drawers_hard.py.
_IAP_DEBUG = os.environ.get('IAP_DEBUG', '0') not in ('0', '', 'false', 'False')


def _iap_log(msg):
    if _IAP_DEBUG:
        print(f'[inspect-and-pick-debug] {msg}', flush=True)


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


# -------------------- success conditions --------------------

class JointTriggerCondition(Condition):
    """Latches once the joint has moved more than `threshold` from its
    initial reading. Copied from put_block_back.py.
    """
    def __init__(self, joint: Joint, threshold: float):
        self._joint = joint
        self._origin = joint.get_joint_position()
        self._threshold = threshold
        self._done = False

    def condition_met(self):
        if not self._done:
            delta = abs(self._joint.get_joint_position() - self._origin)
            if delta > self._threshold:
                self._done = True
        return self._done, False


class DetectedTriggerCondition(Condition):
    """Latching proximity detection. Once the sensor sees the object
    even briefly, the condition stays met for the rest of the episode.
    Copied from put_block_back.py.
    """
    def __init__(self, obj: Object, sensor: ProximitySensor):
        self._obj = obj
        self._sensor = sensor
        self._done = False

    def condition_met(self):
        if not self._done and self._sensor.is_detected(self._obj):
            self._done = True
        return self._done, False


class DrawerClosedCondition(Condition):
    """Shape-y closed-baseline check. See blocks_in_drawers.py for the
    full rationale (shape-y is immune to gripper.grasp() reparenting,
    which desyncs the joint's intrinsic DOF reading).
    """
    def __init__(self, drawer_shape: Shape, closed_y: float,
                 threshold: float = 0.03):
        self._shape = drawer_shape
        self._closed_y = closed_y
        self._threshold = threshold

    def condition_met(self):
        y = self._shape.get_position()[1]
        met = abs(y - self._closed_y) < self._threshold
        if _IAP_DEBUG and not met:
            _iap_log(f'COND DrawerClosed FAIL shape={self._shape.get_name()} '
                     f'y={y:+.4f} baseline={self._closed_y:+.4f} '
                     f'delta={y - self._closed_y:+.4f}')
        return met, False


# -------------------- constants --------------------

DRAWER_NAMES = ['bottom', 'middle', 'top']

# Gripper orientations.
GRIPPER_HANDLE = [-1.5705, 0.0, -3.1412]
GRIPPER_ABOVE = [-3.1416, 0.0, 1.5708]

# Scene geometry (inherited from blocks_in_drawers_hard.ttm). Drawer
# slides along -Y; the .ttm uses the same geometry verbatim.
DRAWER_TRAVEL = -0.21
HANDLE_APPROACH_DY = -0.042

# Joint motor force used to hold inactive drawers against arm-brush
# drift. See blocks_in_drawers.py for tuning rationale.
DRAWER_BRAKE_FORCE = 0.0

# Block geometry. Same dimensions as blocks_in_drawers_hard's red cube.
BLOCK_SIZE = [0.04, 0.04, 0.04]
BLOCK_HALF = BLOCK_SIZE[2] / 2
BLOCK_MASS = 0.05

# v0: block_top is the target. The other two blocks are spawned at
# default-gray for visual symmetry but never picked.
DEFAULT_BLOCK_COLOR = [0.5, 0.5, 0.5]
TARGET_COLORS = {
    'red':   [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.0],
    'blue':  [0.0, 0.0, 1.0],
}

Z_TABLE = 0.752
Z_BLOCK_GRASP = Z_TABLE + BLOCK_HALF
Z_BLOCK_APPROACH = Z_BLOCK_GRASP + 0.15

Z_TRANSIT = 1.28
Z_INTERIOR_CLEARANCE = 0.10

# Bias the block spawn toward the front of the drawer interior (-Y),
# so when the drawer is pulled fully open the block sits closer to the
# front edge -- shorter reach for the gripper, more clearance from the
# back wall.
BLOCK_SPAWN_DY = -0.10

# Wrist lift used during drawer pull/push waypoints. The gripper
# still grasps the handle via gripper.grasp(); raising the wrist by
# this amount tilts the handle grip slightly upward but moves the
# elbow into a less-bent configuration so the linear path planner
# (RRTConnect) doesn't fail at the bottom drawer.
DRAWER_PULL_LIFT_Z = 0.05

# Bin drop geometry. Block is lowered to BIN_DROP_Z above the bin
# sensor before release; sensor sits at the bottom of the bin interior.
BIN_DROP_Z = 0.06        # meters above the bin sensor
BIN_APPROACH_Z = 0.20    # meters above the bin sensor (transit)


# -------------------- helpers --------------------

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


# -------------------- task --------------------

class InspectAndPick(Task):

    def init_task(self) -> None:
        task_base = self.get_base()

        # Slow the arm down. Panda defaults are max_velocity=1.0,
        # max_acceleration=4.0; lowered values reduce arm acceleration
        # so blocks resting inside a drawer don't slide on each pull.
        self.robot.arm.max_velocity = 0.4
        self.robot.arm.max_acceleration = 1.5

        # --- drawers (reused from bid_hard scene) ---------------------
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

        # Soft-braked velocity-locked motor on every drawer joint.
        for j in self._drawer_joints.values():
            j.set_motor_enabled(True)
            j.set_control_loop_enabled(False)
            j.set_joint_target_velocity(0.0)
            j.set_motor_locked_at_zero_velocity(True)
            j.set_joint_force(DRAWER_BRAKE_FORCE)

        # --- button ---------------------------------------------------
        self._button_shape = Shape('push_button_target')
        self._button_joint = Joint('target_button_joint')
        self._button_topplate = Shape('target_button_topPlate')

        # --- bin ------------------------------------------------------
        self._bin_shape = Shape('small_container')
        self._bin_sensor = ProximitySensor('success_container')

        # --- blocks (created at runtime) ------------------------------
        # block_<drawer> spawns inside drawer_<drawer>.
        self._blocks = {}
        for name in DRAWER_NAMES:
            obj_name = f'block_{name}'
            if Object.exists(obj_name):
                Shape(obj_name).remove()
            block = Shape.create(
                type=PrimitiveShape.CUBOID, size=BLOCK_SIZE,
                respondable=True, static=False, mass=BLOCK_MASS)
            block.set_name(obj_name)
            block.set_color(DEFAULT_BLOCK_COLOR)
            block.set_parent(task_base)
            # Initial position is overwritten in init_episode. Anywhere
            # off the table is fine -- avoids spawn-overlap with scene
            # objects on the first physics step.
            block.set_position([0.0, 0.0, Z_TABLE + 1.0])
            self._blocks[name] = block

        self.register_graspable_objects(
            list(self._blocks.values())
            + list(self._drawer_shapes.values()))

        # --- delete legacy waypoints inherited from the .ttm ----------
        for i in range(100):
            if Object.exists(f'waypoint{i}'):
                Dummy(f'waypoint{i}').remove()

        # --- waypoint layout (v0) -------------------------------------
        # Phase 1: inspect top, middle, bottom (7 wp/drawer) -> 21 wp.
        # Phase 2: button press (4 wp)                         -> 4  wp.
        # Phase 3: retrieve target block into bin (21 wp)      -> 21 wp.
        # Total: 46 waypoints.
        self._n_waypoints = 46
        for idx in range(self._n_waypoints):
            wp = Dummy.create()
            wp.set_name(f'waypoint{idx}')
            wp.set_orientation(GRIPPER_ABOVE)
            wp.set_parent(task_base)

        # Inspect-phase callbacks. Per-drawer offsets within each
        # 7-waypoint chunk (see _inspect_phase_specs for the layout):
        #   +1  close gripper around handle (lock other drawers)
        #   +5  open gripper, restore drawer ranges
        #   +6  retract -- no callback, just a transit pose
        callbacks = {}
        for phase_idx, dname in enumerate(['top', 'middle', 'bottom']):
            base = phase_idx * 7
            callbacks[base + 1] = self._make_grip_drawer(dname)
            callbacks[base + 5] = self._open

        # Button-phase callbacks. Offsets within the 4-waypoint chunk:
        #   +0  above button (transit, open jaws)
        #   +1  same pose, close gripper
        #   +2  press
        #   +3  retract
        button_base = 21
        callbacks[button_base + 1] = self._close_empty

        # Retrieve-phase callbacks. Drawer is the one holding the
        # target color (red); resolved per-episode via
        # self._target_drawer, so the callbacks read it live.
        # See _retrieve_phase_specs for the 17-wp layout.
        retrieve_base = 25
        callbacks[retrieve_base + 0] = self._open
        callbacks[retrieve_base + 2] = self._grip_target_drawer
        callbacks[retrieve_base + 4] = self._open
        callbacks[retrieve_base + 10] = self._grip_target_block
        callbacks[retrieve_base + 14] = self._open
        callbacks[retrieve_base + 16] = self._open  # open jaws at standoff
        callbacks[retrieve_base + 18] = self._grip_target_drawer_smooth
        # Release at end of push (+19), NOT at end of retract (+20),
        # so the gripper isn't still grasping the handle when it moves
        # back along -Y -- otherwise the retract pulls the drawer open.
        callbacks[retrieve_base + 19] = self._open

        for wp_idx, cb in callbacks.items():
            self.register_waypoint_ability_end(wp_idx, cb)

        # Ignore collisions for every waypoint during planning. We rely
        # on RRT-Connect non-linear paths almost everywhere (matches
        # stack_and_swap). Linear-only on the pick descent (halfway,
        # approach, grasp) -- pure top-down vertical motion where we
        # need the gripper to land exactly on the block, not near a
        # random IK solution.
        for idx in range(self._n_waypoints):
            self.register_waypoint_ability_start(idx, self._skip_collisions)
        for idx in (retrieve_base + 6,
                    retrieve_base + 8, retrieve_base + 9, retrieve_base + 10,
                    retrieve_base + 15):
            self.register_waypoint_ability_start(idx, self._set_linear)

        # Live-align transit + halfway + approach + grasp waypoints
        # to the actual block pos at the start of the transit (fires
        # before the planner picks the path to the high transit pose).
        self.register_waypoint_ability_start(
            retrieve_base + 7,
            self._make_align_grasp(
                transit_idx=retrieve_base + 7,
                halfway_idx=retrieve_base + 8,
                approach_idx=retrieve_base + 9,
                grasp_idx=retrieve_base + 10))

        self.goal_conditions = []

    # -- gripper / drawer callbacks -----------------------------------

    def _make_grip_block(self, drawer_name):
        def _fn(_waypoint):
            target = self._blocks[drawer_name]
            gripper = self.robot.gripper
            done = False
            while not done:
                done = gripper.actuate(0.0, velocity=0.04)
                self.pyrep.step()
            gripper.grasp(target)
        return _fn

    def _grip_target_block(self, _waypoint):
        """End-of-path callback: close gripper around the block. The
        joint-space pre-positioning happens earlier (start-callback at
        retrieve+7), and the planner drives the short descent from
        there to the grasp pose.
        """
        target = self._blocks[self._target_drawer]
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        ok = gripper.grasp(target)
        try:
            tip = self.robot.arm.get_tip().get_position()
            bp = target.get_position()
            print(f'[iap-grasp-block] target={target.get_name()} '
                  f'grasp_ok={ok} tip={[round(v, 3) for v in tip]} '
                  f'block={[round(v, 3) for v in bp]} '
                  f'grasped={[o.get_name() for o in gripper.get_grasped_objects()]}',
                  flush=True)
        except Exception as e:
            print(f'[iap-grasp-block] log-err: {e}', flush=True)

    def _grip_target_drawer(self, _waypoint):
        """End-of-path callback: grip the handle of the drawer holding
        the target color (resolved live from self._target_drawer).
        """
        self._close_grip_drawer_by_name(self._target_drawer)

    def _grip_target_drawer_smooth(self, _waypoint):
        """End-of-path callback used for the *retrieve close* phase.
        After the bin drop the planner often lands the gripper far
        from the open-position handle; we bypass the planner with a
        smooth interpolated joint drive (same pattern as the pick).
        """
        drawer_name = self._target_drawer
        anchor = self._drawer_anchors[drawer_name]
        joint = self._drawer_joints[drawer_name]
        arm = self.robot.arm

        # Target pose: anchor's current world pose offset by the
        # current joint position along -Y (drawer face slides with
        # joint travel).
        ax, ay, az = anchor.get_position()
        jpos = joint.get_joint_position()
        target_xyz = [ax, ay - jpos, az]
        target_euler = list(anchor.get_orientation())

        # Verify the linear descent first.
        try:
            path = arm.get_linear_path(
                position=target_xyz, euler=target_euler,
                steps=80, ignore_collisions=True)
            done = False
            while not done:
                done = path.step()
                self.pyrep.step()
            tip = arm.get_tip().get_position()
            err = max(abs(tip[i] - target_xyz[i]) for i in range(3))
            print(f'[iap-close-approach] linear tip='
                  f'{[round(v,3) for v in tip]} '
                  f'target={[round(v,3) for v in target_xyz]} err={err:.3f}',
                  flush=True)
        except Exception as e:
            print(f'[iap-close-approach] linear FAIL: {e}', flush=True)
            err = 1.0

        # If linear was short, fall back to interpolated sampling IK.
        if err > 0.02:
            try:
                target_joints = arm.solve_ik_via_sampling(
                    position=target_xyz, euler=target_euler,
                    ignore_collisions=True, max_configs=1,
                    max_time_ms=500, trials=600)[0]
                start_joints = arm.get_joint_positions()
                steps = 120
                for s in range(1, steps + 1):
                    alpha = s / steps
                    interp = [
                        sj + alpha * (tj - sj)
                        for sj, tj in zip(start_joints, target_joints)
                    ]
                    arm.set_joint_target_positions(interp)
                    self.pyrep.step()
                tip = arm.get_tip().get_position()
                err2 = max(abs(tip[i] - target_xyz[i]) for i in range(3))
                print(f'[iap-close-approach] sampling-fallback tip='
                      f'{[round(v,3) for v in tip]} err={err2:.3f}',
                      flush=True)
            except Exception as e:
                print(f'[iap-close-approach] sampling-fallback FAIL: {e}',
                      flush=True)

        # Now grip the handle.
        self._close_grip_drawer_by_name(drawer_name)

    def _make_align_grasp(self, transit_idx, halfway_idx, approach_idx,
                          grasp_idx, approach_offset_z=0.10):
        """Start-of-path callback fired at transit_idx.

        Reads the live block pose, rewrites four descent waypoints
        (transit / halfway / approach / grasp), and *pre-positions the
        arm* via two-stage joint-space motion:

          stage A: current -> safe neutral mid-table top-down pose
          stage B: neutral -> high transit above block (top-down)

        Both stages happen in open space (no drawer interaction) so
        the joint-space interpolation can find a comfortable config
        without colliding. After this callback returns, RLBench's
        planner only has to handle a short descent (halfway, approach,
        grasp) from a known-good starting config.
        """
        def _fn(waypoint):
            waypoint._ignore_collisions = True
            block = self._blocks[self._target_drawer]
            bx, by, bz = block.get_position()
            transit_z = Z_TRANSIT
            halfway_z = (transit_z + bz) / 2.0
            transit_wp = Dummy(f'waypoint{transit_idx}')
            halfway_wp = Dummy(f'waypoint{halfway_idx}')
            approach_wp = Dummy(f'waypoint{approach_idx}')
            grasp_wp = Dummy(f'waypoint{grasp_idx}')
            transit_wp.set_position([bx, by, transit_z])
            transit_wp.set_orientation(GRIPPER_ABOVE)
            halfway_wp.set_position([bx, by, halfway_z])
            halfway_wp.set_orientation(GRIPPER_ABOVE)
            approach_wp.set_position([bx, by, bz + approach_offset_z])
            approach_wp.set_orientation(GRIPPER_ABOVE)
            grasp_wp.set_position([bx, by, bz])
            grasp_wp.set_orientation(GRIPPER_ABOVE)
            try:
                name = waypoint.get_waypoint_object().get_name()
            except Exception:
                name = '?'
            print(f'[inspect_and_pick] >>> START path to {name} '
                  f'(aligned grasp to block@[{bx:.3f},{by:.3f},{bz:.3f}], '
                  f'halfway_z={halfway_z:.3f})', flush=True)

            # Two-stage joint-space pre-positioning.
            arm = self.robot.arm

            def _drive_to_pose(pose_xyz, label, steps=120):
                try:
                    target_joints = arm.solve_ik_via_sampling(
                        position=pose_xyz, euler=GRIPPER_ABOVE,
                        ignore_collisions=True, max_configs=1,
                        max_time_ms=500, trials=600)[0]
                    start_joints = arm.get_joint_positions()
                    for s in range(1, steps + 1):
                        alpha = s / steps
                        interp = [
                            sj + alpha * (tj - sj)
                            for sj, tj in zip(start_joints, target_joints)
                        ]
                        arm.set_joint_target_positions(interp)
                        self.pyrep.step()
                    tip = arm.get_tip().get_position()
                    err = max(abs(tip[i] - pose_xyz[i]) for i in range(3))
                    print(f'[iap-prepos] {label} tip='
                          f'{[round(v,3) for v in tip]} '
                          f'target={[round(v,3) for v in pose_xyz]} '
                          f'err={err:.3f}', flush=True)
                except Exception as e:
                    print(f'[iap-prepos] {label} FAIL: {e}', flush=True)

            # Stage A: safe mid-table neutral, top-down. High z (well
            # above any drawer/cabinet geometry) and centered laterally.
            _drive_to_pose([0.25, 0.0, Z_TRANSIT], 'stage-A-neutral')
            # Stage B: high transit above the live block XY.
            _drive_to_pose([bx, by, Z_TRANSIT], 'stage-B-above-block')
        return _fn

    def _make_grip_drawer(self, drawer_name):
        def _fn(_waypoint):
            self._close_grip_drawer_by_name(drawer_name)
        return _fn

    def _close_grip_drawer_by_name(self, drawer_name):
        gripper = self.robot.gripper
        done = False
        while not done:
            done = gripper.actuate(0.0, velocity=0.04)
            self.pyrep.step()
        ok = gripper.grasp(self._drawer_shapes[drawer_name])
        try:
            tip = self.robot.arm.get_tip().get_position()
            handle = self._drawer_shapes[drawer_name].get_position()
            print(f'[iap-grasp-drawer] drawer={drawer_name} '
                  f'grasp_ok={ok} tip={[round(v, 3) for v in tip]} '
                  f'shape={[round(v, 3) for v in handle]} '
                  f'grasped={[o.get_name() for o in gripper.get_grasped_objects()]}',
                  flush=True)
        except Exception as e:
            print(f'[iap-grasp-drawer] log-err: {e}', flush=True)
        # Pin every non-active drawer's joint by shrinking its range to
        # zero around its current position. The active drawer keeps its
        # full range. See blocks_in_drawers_hard.py for rationale.
        self._active_drawer_name = drawer_name
        for name, j in self._drawer_joints.items():
            if name == drawer_name:
                continue
            pos = j.get_joint_position()
            j.set_joint_interval(False, [pos, 0.0])

    def _close_empty(self, _waypoint):
        """Close gripper without grasping anything. Used for the button
        press waypoint -- the closed fingertips push the top-plate.
        """
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
        # Restore the original [0, 0.21] range on every drawer so the
        # next grip callback can open/close any of them.
        for j in self._drawer_joints.values():
            j.set_joint_interval(False, [0.0, 0.21])
        self._active_drawer_name = None

    def _skip_collisions(self, waypoint):
        waypoint._ignore_collisions = True
        try:
            name = waypoint.get_waypoint_object().get_name()
        except Exception:
            name = '?'
        print(f'[inspect_and_pick] >>> START path to {name}')

    def _set_linear(self, waypoint):
        waypoint._linear_only = True
        waypoint._ignore_collisions = True

    # -- episode setup -------------------------------------------------

    def init_episode(self, index: int) -> List[str]:
        self._active_drawer_name = None

        # Reset every drawer to closed with its full [0, 0.21] range.
        for name in DRAWER_NAMES:
            j = self._drawer_joints[name]
            j.set_joint_interval(False, [0.0, 0.21])
            j.set_joint_position(0.0, disable_dynamics=True)

        self._closed_y = float(
            self._drawer_shapes[DRAWER_NAMES[0]].get_position()[1])
        _iap_log(f'INIT closed_y baseline={self._closed_y:+.4f}')

        # --- color-to-drawer permutation ------------------------------
        # Each drawer holds a distinct R/G/B block; the assignment is
        # one of the 6 permutations of {red,green,blue} ->
        # {top,middle,bottom}, selected deterministically by the
        # variation index. Target color is fixed (red); the drawer
        # holding red is what the model must recall.
        all_perms = list(permutations(['red', 'green', 'blue']))
        colors = list(all_perms[index])
        self._color_to_drawer = dict(zip(colors, DRAWER_NAMES))
        for color_name, dname in self._color_to_drawer.items():
            self._blocks[dname].set_color(TARGET_COLORS[color_name])

        self._target_color = 'red'
        self._target_drawer = self._color_to_drawer[self._target_color]
        _iap_log(f'INIT color->drawer: {self._color_to_drawer}, '
                 f'target={self._target_color} -> drawer={self._target_drawer}')

        # --- block in-drawer placement --------------------------------
        # Each block sits at the geometric center of its drawer's
        # interior (the success_<name> sensor is positioned there).
        # Z is offset upward so the block rests on the drawer floor.
        for name in DRAWER_NAMES:
            spos = self._drawer_sensors[name].get_position()
            self._blocks[name].set_position(
                [spos[0],
                 spos[1] + BLOCK_SPAWN_DY,
                 spos[2] + BLOCK_HALF + 0.005])

        # --- waypoint placement ---------------------------------------
        target_block = self._blocks[self._target_drawer]
        rel_specs = (
            self._inspect_phase_specs('top',    base_idx=0)
            + self._inspect_phase_specs('middle', base_idx=7)
            + self._inspect_phase_specs('bottom', base_idx=14)
            + self._button_phase_specs(base_idx=21)
            + self._retrieve_phase_specs(
                self._target_drawer, target_block, base_idx=25)
        )

        print(f'[inspect_and_pick] Setting {self._n_waypoints} '
              f'waypoints (target={self._target_color} '
              f'in drawer={self._target_drawer}):')
        for idx, rel in rel_specs[:self._n_waypoints]:
            pos = rel.world_position()
            ori = rel.world_orientation()
            self._wp(idx, pos[0], pos[1], pos[2], ori)
            print(f'  wp{idx:2d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, '
                  f'{pos[2]:.3f}] ori={[round(o,3) for o in ori]}')

        # --- success conditions ---------------------------------------
        self.goal_conditions = [
            JointTriggerCondition(self._button_joint, threshold=0.003),
            DetectedTriggerCondition(target_block, self._bin_sensor),
            DrawerClosedCondition(self._drawer_shapes['top'],    self._closed_y),
            DrawerClosedCondition(self._drawer_shapes['middle'], self._closed_y),
            DrawerClosedCondition(self._drawer_shapes['bottom'], self._closed_y),
            NothingGrasped(self.robot.gripper),
        ]
        self.register_success_conditions(
            [ConditionSet(self.goal_conditions, order_matters=False)])

        return [
            'inspect each drawer, press the button, '
            'then pick the red block and put it in the box'
        ]

    # -- phase specs ---------------------------------------------------

    def _inspect_phase_specs(self, drawer_name, base_idx):
        """7-waypoint inspect chunk.

          +0 arrive at handle (open jaws)
          +1 close gripper at handle (callback)
          +2 pull fully open (linear)
          +3 hold open (camera keyframe)
          +4 push fully closed (linear)
          +5 open gripper at handle (callback)
          +6 retract along -Y, away from the just-closed drawer
        """
        anchor = self._drawer_anchors[drawer_name]
        retract_dy = HANDLE_APPROACH_DY * 2
        close_overshoot = 0.005
        i = base_idx
        return [
            (i + 0, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            (i + 1, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            (i + 2, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 3, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 4, RelWaypoint(anchor,
                [0.0, close_overshoot, 0.0], 'anchor')),
            (i + 5, RelWaypoint(anchor,
                [0.0, close_overshoot, 0.0], 'anchor')),
            (i + 6, RelWaypoint(anchor, [0.0, retract_dy, 0.0], 'anchor')),
        ]

    def _button_phase_specs(self, base_idx):
        """4-waypoint button press: above (open jaws), close gripper at
        same pose, press (descend below plate), retract.

        Press z is 1 cm BELOW the top-plate top so the closed gripper
        body actually depresses the button enough to fire the joint
        trigger (3 mm threshold). Pattern matches stack_and_swap, where
        Z_BUTTON_PRESS sits below the plate surface.
        """
        topplate = self._button_topplate
        i = base_idx
        return [
            (i + 0, RelWaypoint(topplate,
                [0.0, 0.0, 0.20], GRIPPER_ABOVE)),
            (i + 1, RelWaypoint(topplate,
                [0.0, 0.0, 0.20], GRIPPER_ABOVE)),
            (i + 2, RelWaypoint(topplate,
                [0.0, 0.0, -0.01], GRIPPER_ABOVE)),
            (i + 3, RelWaypoint(topplate,
                [0.0, 0.0, 0.20], GRIPPER_ABOVE)),
        ]

    def _retrieve_phase_specs(self, drawer_name, block, base_idx):
        """21-waypoint retrieve chunk.

          +0  pre-approach pose, callback opens gripper
          +1  arrive at handle (open jaws)
          +2  close gripper at handle (callback)
          +3  pull fully open
          +4  hold open (gripper releases here)
          +5  retract along -Y at low Z, sideways orientation
          +6  vertical lift, same XY as +5, sideways orientation (linear)
          +7  reorient to top-down at high z above block XY (joint-space pre-pos)
          +8  halfway descent (top-down) at block XY (linear)
          +9  approach block from above (10 cm) (linear)
          +10 at block (callback grasps block) (linear)
          +11 lift to transit Z
          +12 above bin (transit)
          +13 just above bin floor
          +14 release block (callback opens gripper)
          +15 vertical lift above bin (linear, top-down) -- gets gripper
              out of the bin into open space before the joint-space
              reorient to the close-handle approach
          +16 standoff in front of open handle (open jaws, sideways)
          +17 arrive at handle (open jaws)
          +18 close gripper at handle (callback: smooth approach + grip)
          +19 push fully closed (5 mm overshoot, callback releases)
          +20 retract along -Y
        """
        anchor = self._drawer_anchors[drawer_name]
        drawer_sensor = self._drawer_sensors[drawer_name]
        bin_sensor = self._bin_sensor

        sensor_z = drawer_sensor.get_position()[2]
        # Block z when the drawer is open (block sits on drawer floor).
        block_z_open = sensor_z + BLOCK_HALF + 0.005
        # Approach from 10 cm above the block.
        approach_offset_z = 0.10
        z_transit_from_block = Z_TRANSIT - block_z_open
        retract_dy = HANDLE_APPROACH_DY * 2

        i = base_idx
        return [
            # Pre-approach: gripper-open callback fires here.
            (i + 0, RelWaypoint(anchor, [0.0, retract_dy, 0.0], 'anchor')),

            # Open drawer.
            (i + 1, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            (i + 2, RelWaypoint(anchor, [0.0, 0.0, 0.0], 'anchor')),
            (i + 3, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 4, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),

            # Retract along -Y, low z, sideways orientation.
            (i + 5, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL + retract_dy, 0.0], 'anchor')),

            # Vertical lift first -- straight up, same XY as +5,
            # still sideways orient. Resets the joint config to a
            # comfortable mid-reach pose before the big reorient.
            (i + 6, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL + retract_dy, 0.20], 'anchor')),

            # Reorient to top-down at high z over the block XY.
            # (Position rewritten by align-grasp callback at runtime.)
            (i + 7, RelWaypoint(drawer_sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_block], GRIPPER_ABOVE)),

            # Halfway descent (top-down). Rewritten by align-grasp.
            (i + 8, RelWaypoint(drawer_sensor,
                [0.0, DRAWER_TRAVEL,
                 (z_transit_from_block + BLOCK_HALF + 0.005) / 2.0],
                GRIPPER_ABOVE)),

            # Approach block from 10 cm above.
            (i + 9, RelWaypoint(drawer_sensor,
                [0.0, DRAWER_TRAVEL, BLOCK_HALF + 0.005 + approach_offset_z],
                GRIPPER_ABOVE)),
            # At block (grasp pose).
            (i + 10, RelWaypoint(drawer_sensor,
                [0.0, DRAWER_TRAVEL, BLOCK_HALF + 0.005], GRIPPER_ABOVE)),

            # Lift block to transit Z.
            (i + 11, RelWaypoint(drawer_sensor,
                [0.0, DRAWER_TRAVEL, z_transit_from_block], GRIPPER_ABOVE)),

            # Drop in bin.
            (i + 12, RelWaypoint(bin_sensor,
                [0.0, 0.0, BIN_APPROACH_Z], GRIPPER_ABOVE)),
            (i + 13, RelWaypoint(bin_sensor,
                [0.0, 0.0, BIN_DROP_Z], GRIPPER_ABOVE)),
            (i + 14, RelWaypoint(bin_sensor,
                [0.0, 0.0, BIN_DROP_Z], GRIPPER_ABOVE)),

            # Vertical lift above bin -- gets the gripper out of the
            # bin into open space before the joint-space pre-position
            # for the close approach.
            (i + 15, RelWaypoint(bin_sensor,
                [0.0, 0.0, BIN_APPROACH_Z + 0.20], GRIPPER_ABOVE)),

            # Close drawer.
            (i + 16, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL - 0.08, 0.0], 'anchor')),
            (i + 17, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 18, RelWaypoint(anchor,
                [0.0, DRAWER_TRAVEL, 0.0], 'anchor')),
            (i + 19, RelWaypoint(anchor, [0.0, 0.005, 0.0], 'anchor')),
            (i + 20, RelWaypoint(anchor, [0.0, retract_dy, 0.0], 'anchor')),
        ]

    # -- misc ----------------------------------------------------------

    def _wp(self, idx, x, y, z, orientation):
        wp = Dummy(f'waypoint{idx}')
        wp.set_position([x, y, z])
        wp.set_orientation(orientation)

    def step(self) -> None:
        if _IAP_DEBUG and os.environ.get('IAP_DEBUG_STEP', '0') not in (
                '0', '', 'false', 'False'):
            try:
                bp = [self._blocks[n].get_position() for n in DRAWER_NAMES]
                _iap_log(f'STEP blocks: '
                         + ' '.join(f'{n}=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]'
                                     for n, p in zip(DRAWER_NAMES, bp)))
            except Exception:
                pass

    def variation_count(self) -> int:
        return 6

    def is_static_workspace(self) -> bool:
        return True
