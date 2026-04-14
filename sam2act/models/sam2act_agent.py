# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import sam2act.utils.peract_utils as peract_utils
import sam2act.mvt.utils as mvt_utils
import sam2act.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from sam2act.mvt.augmentation import apply_se3_aug_con, apply_se3_aug_con_same, apply_se3_aug_con_sequence, apply_se3_aug_given_matrix, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from sam2act.utils.dataset import _clip_encode_text
from sam2act.utils.lr_sched_utils import GradualWarmupScheduler

# Waypoint index bins for per-waypoint-group diagnostic metrics.
# Each element (lo, hi) is inclusive on both sides.
WP_BINS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 999)]


def _normalize_step_idx(raw, device):
    """Normalize a keypoint index from any source to a [B] int64 tensor.

    The replay buffer stores ``keypoint_idx`` as a flat [B] extra element, but
    at eval time the rollout generator wraps every observation value as
    ``torch.tensor(np.array([v]))`` which can produce [1, 1] or deeper shapes
    depending on what the env returns. This helper flattens any shape to [B]
    and casts to int64 on the target device. Returns None if the input is None
    so callers can cleanly bypass the step-embedding branch.
    """
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        t = raw
    else:
        t = torch.as_tensor(raw)
    return t.to(device).long().reshape(-1)


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        if val is not None:
            filtered_val = [v for v in val if v is not None]
            mean_value = np.mean(filtered_val) if filtered_val else None  # Avoid empty list error
        else:
            mean_value = None
        if mean_value is not None:
            out[key] = mean_value
    pprint.pprint(out)
    return out


def manage_diag_log(agent, tasks, keypoint_idx, per_elem_trans_loss,
                    q_trans, action_trans, rot_q, action_rot_x_oh,
                    action_rot_y_oh, action_rot_z_oh, grip_q, action_grip_oh,
                    use_memory, num_rot_classes, reset_log=False,
                    per_elem_total_loss=None):
    """Per-task and per-waypoint training diagnostics from detached tensors.

    `per_elem_trans_loss` has shape (bs, nc) — the raw translation CE output.
    `per_elem_total_loss` has shape (bs,) — per-sample sum of all active loss
    components (trans + rot + grip + collision in Stage 0/1, or trans alone
    in Stage 2). When None, falls back to the trans loss alone.
    """
    from collections import defaultdict
    if not hasattr(agent, 'diag_log') or reset_log:
        agent.diag_log = {k: defaultdict(list) for k in
                          ["task_trans_loss", "task_total_loss",
                           "task_pixel_match", "task_grip_match",
                           "task_rot_match", "wp_pixel_match"]}
        agent.diag_log["prev_window"] = {}

    elem_trans = per_elem_trans_loss.mean(dim=1)  # (bs, nc) -> (bs,)
    if per_elem_total_loss is None:
        elem_total = elem_trans
    else:
        elem_total = per_elem_total_loss
    elem_pixel = (q_trans.argmax(dim=1) == action_trans.argmax(dim=1)).float().mean(dim=1)

    elem_grip = elem_rot = None
    if not use_memory and grip_q is not None and rot_q is not None:
        elem_grip = (grip_q.argmax(-1) == action_grip_oh.argmax(-1)).float()
        nrc = num_rot_classes
        elem_rot = (
            (rot_q[:, 0*nrc:1*nrc].argmax(-1) == action_rot_x_oh.argmax(-1)).float() +
            (rot_q[:, 1*nrc:2*nrc].argmax(-1) == action_rot_y_oh.argmax(-1)).float() +
            (rot_q[:, 2*nrc:3*nrc].argmax(-1) == action_rot_z_oh.argmax(-1)).float()
        ) / 3.0

    for task in set(tasks):
        mask = torch.tensor([t == task for t in tasks], device=elem_trans.device)
        if mask.sum() == 0:
            continue
        agent.diag_log["task_trans_loss"][task].append(elem_trans[mask].mean().item())
        agent.diag_log["task_total_loss"][task].append(elem_total[mask].mean().item())
        agent.diag_log["task_pixel_match"][task].append(elem_pixel[mask].mean().item())
        if elem_grip is not None:
            agent.diag_log["task_grip_match"][task].append(elem_grip[mask].mean().item())
            agent.diag_log["task_rot_match"][task].append(elem_rot[mask].mean().item())

        if keypoint_idx is not None:
            for lo, hi in WP_BINS:
                wp_mask = mask & (keypoint_idx >= lo) & (keypoint_idx <= hi)
                if wp_mask.any():
                    agent.diag_log["wp_pixel_match"][f"{task}/wp_{lo}-{hi}"].append(
                        elem_pixel[wp_mask].mean().item())


def get_diag_summary(agent, window=200):
    """Rolling window per-task diagnostics with convergence detection.

    Emits `{task}/trans_loss` (translation CE only) and `{task}/total_loss`
    (sum of all active loss components). In Stage 2 these are equal because
    rot/grip/collision heads are frozen.

    The convergence detector tracks `{task}/total_loss` since that is the
    quantity being optimized.
    """
    d = agent.diag_log
    out = {}
    min_samples = 50
    print("--- per-task diagnostics ---")
    for task in sorted(d["task_total_loss"].keys()):
        vals = d["task_total_loss"][task]
        if len(vals) < min_samples:
            continue
        cur_total = np.mean(vals[-window:])
        cur_trans = np.mean(d["task_trans_loss"][task][-window:])
        cur_pm = np.mean(d["task_pixel_match"][task][-window:])
        cur_gm = np.mean(d["task_grip_match"][task][-window:]) if d["task_grip_match"].get(task) else None
        cur_rm = np.mean(d["task_rot_match"][task][-window:]) if d["task_rot_match"].get(task) else None

        prev = d["prev_window"].get(task, {})
        prev_total = prev.get("total_loss", cur_total)
        if len(vals) < window * 2:
            status = "WARMING"
        elif cur_total < prev_total * 0.97:
            status = "CONVERGING"
        elif cur_total > prev_total * 1.03:
            status = "DIVERGING"
        else:
            status = "STALLED"

        d["prev_window"][task] = {"total_loss": cur_total}
        gm_str = f" g_match={cur_gm:.3f}" if cur_gm is not None else ""
        rm_str = f" r_match={cur_rm:.3f}" if cur_rm is not None else ""
        print(
            f"  {task}: total={cur_total:.4f} trans={cur_trans:.4f} "
            f"px_match={cur_pm:.3f}{gm_str}{rm_str} [{status}]"
        )
        out[f"{task}/total_loss"] = cur_total
        out[f"{task}/trans_loss"] = cur_trans
        out[f"{task}/pixel_match"] = cur_pm
        if cur_gm is not None:
            out[f"{task}/grip_match"] = cur_gm
        if cur_rm is not None:
            out[f"{task}/rot_match"] = cur_rm

    for key, vals in d["wp_pixel_match"].items():
        if len(vals) >= min_samples:
            out[key] = np.mean(vals[-window:])
    return out


def horizon_loss_cal(output, data_horizon):

    shape = output.shape
    bs, ah = shape[0], shape[1]
    mask = torch.arange(ah).expand(bs, ah).to(output.device) < data_horizon.view(bs, 1)
    if len(shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(output)

        output_mask = output * mask
        ah_ave = output_mask.sum(dim=(0,2))/(mask.sum(dim=(0,2)).float() + 1e-5)
        
    else:
        output_mask = output * mask
        ah_ave = output_mask.sum(dim=0)/(mask.sum(dim=0).float() + 1e-5)
        
    total_ave = output_mask.sum()/(mask.sum().float())
    
    return total_ave,ah_ave


class SAM2Act_Agent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        add_lang: bool,
        amp: bool,
        bnb: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        action_horizon=None,
        same_trans_aug_per_seq: bool = False,
        use_memory: bool = False,
        num_maskmem: int = 7,
        gradient_accumulation_steps: int = 1,
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

        self.scaler = GradScaler(enabled=self.amp)

        self.action_horizon = action_horizon

        self._same_trans_aug_per_seq = same_trans_aug_per_seq

        self.use_memory = use_memory
        self._num_maskmem = num_maskmem
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Agent-side keypoint counter used by the learnable step embedding.
        # Reset to 0 on every rollout via reset(); incremented in act() when
        # the env does not supply its own keypoint_idx.
        self._current_step = 0

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        sam_mem_params = []
        upsample_params = []
        
        for name, param in self._network.named_parameters():
            if "sam" in name and "image encoder" not in name:
                sam_mem_params.append(param)
            elif "up0" in name:
                upsample_params.append(param)

        param_groups = [
            {"params": sam_mem_params, "lr": self._lr * 2},  # SAM2 (memory)
            {"params": upsample_params, "lr": self._lr},  # upsample layers
        ]

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    self._network.parameters(),
                    # param_groups,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            # q
            if self.stage_two and not self.use_memory:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if not self.use_memory:

            if self.rot_ver == 0:
                # (bs, 218)
                rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
                grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
                # (bs, 2)
                collision_q = out["feat"].view(bs, -1)[
                    :, self.num_all_rot + 2 : self.num_all_rot + 4
                ]
            elif self.rot_ver == 1:
                rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                                dim=-1).view(bs, -1)
                grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
                collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
            else:
                assert False

            y_q = None

            return q_trans, rot_q, grip_q, collision_q, y_q, pts
        
        else:
            return q_trans, None, None, None, None, None

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        # Absolute keypoint index (0..N-1) for the learnable step embedding.
        # Stored as a flat [B] ReplayElement in utils/dataset.py, so skip
        # stack_on_channel (which expects [B, T, C]) and normalize directly.
        step_idx = _normalize_step_idx(
            replay_sample.get("keypoint_idx", None), proprio.device
        )
        return_out = {}

        obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )


            if self._transform_augmentation and backprop:
                if not self._same_trans_aug_per_seq:
                    action_trans_con, action_rot, pc = apply_se3_aug_con(
                        pcd=pc,
                        action_gripper_pose=action_gripper_pose,
                        bounds=torch.tensor(self.scene_bounds),
                        trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                        rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                    )
                    action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                    action_rot = torch.tensor(action_rot).to(pc.device)
                else:
                    bs = pc.shape[0]
                    num_obs = self._num_maskmem + 1
                    num_seq = bs // num_obs

                    action_trans_con_after = []
                    action_rot_after = []
                    pc_after = []
                    for seq_idx in range(num_seq):
                        pc_i = pc[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_gripper_pose_i = action_gripper_pose[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_trans_con_i, action_rot_i, pc_i = apply_se3_aug_con_same(
                            pcd=pc_i,
                            action_gripper_pose=action_gripper_pose_i,
                            bounds=torch.tensor(self.scene_bounds),
                            trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                            rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        )
                        action_trans_con_i = torch.tensor(action_trans_con_i).to(pc.device)
                        action_rot_i = torch.tensor(action_rot_i).to(pc.device)

                        action_trans_con_after.append(action_trans_con_i)
                        action_rot_after.append(action_rot_i)
                        pc_after.append(pc_i)

                    action_trans_con = torch.cat(action_trans_con_after, dim=0)
                    action_rot = torch.cat(action_rot_after, dim=0)
                    pc = torch.cat(pc_after, dim=0)


            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        with autocast(enabled=self.amp):
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            if self.rot_ver == 1:
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                if self.rot_x_y_aug != 0:
                    # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    rot_x_y %= self._num_rotation_classes

            # hm_gt = self.get_gt_hm(
            #     wpt_local, dyn_cam_info, dims=(bs, nc, h, w)
            # )

            out = self._network(
                pc=pc,
                img_feat=img_feat,
                proprio=proprio,
                lang_emb=lang_goal_embs,
                img_aug=img_aug,
                wpt_local=wpt_local if self._network.training else None,
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,
                step_idx=step_idx,  # consumed by MVT_SAM2_Single when use_step_embedding
                # hm_gt=hm_gt,
            )

            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )

            action_trans = self.get_action_trans(
                wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            )

        loss_log = {}
        if not backprop:
            # Compute loss without backprop for validation metrics
            with torch.no_grad():
                trans_loss_per_elem = self._cross_entropy_loss(q_trans, action_trans)
                trans_loss_per_sample = trans_loss_per_elem.mean(dim=1)  # (bs,)
                trans_loss = trans_loss_per_sample.mean()
                # Running per-sample total for the diag log. Starts as trans;
                # other components are added below when they are computed.
                total_per_sample = trans_loss_per_sample
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if not self.use_memory and self.add_rgc_loss:
                    nrc = self._num_rotation_classes
                    rot_x_per_sample = self._cross_entropy_loss(
                        rot_q[:, 0*nrc:1*nrc], action_rot_x_one_hot.argmax(-1))
                    rot_y_per_sample = self._cross_entropy_loss(
                        rot_q[:, 1*nrc:2*nrc], action_rot_y_one_hot.argmax(-1))
                    rot_z_per_sample = self._cross_entropy_loss(
                        rot_q[:, 2*nrc:3*nrc], action_rot_z_one_hot.argmax(-1))
                    grip_per_sample = self._cross_entropy_loss(
                        grip_q, action_grip_one_hot.argmax(-1))
                    collision_per_sample = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1))
                    rot_loss_x = rot_x_per_sample.mean()
                    rot_loss_y = rot_y_per_sample.mean()
                    rot_loss_z = rot_z_per_sample.mean()
                    grip_loss = grip_per_sample.mean()
                    collision_loss = collision_per_sample.mean()
                    total_per_sample = (
                        trans_loss_per_sample
                        + rot_x_per_sample + rot_y_per_sample + rot_z_per_sample
                        + grip_per_sample + collision_per_sample
                    )
                total_loss = (trans_loss + rot_loss_x + rot_loss_y + rot_loss_z
                              + grip_loss + collision_loss)
                loss_log = {
                    "total_loss": total_loss.item(),
                    "trans_loss": trans_loss.item(),
                    "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,
                    "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,
                    "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,
                    "grip_loss": grip_loss.item() if not self.use_memory else None,
                    "collision_loss": collision_loss.item() if not self.use_memory else None,
                    "lr": self._optimizer.param_groups[0]["lr"],
                }
                manage_loss_log(self, loss_log, reset_log=reset_log)
                return_out.update(loss_log)

                if getattr(self, '_enable_diagnostics', True):
                    kp_idx = replay_sample.get("keypoint_idx", None)
                    manage_diag_log(
                        agent=self, tasks=tasks, keypoint_idx=kp_idx,
                        per_elem_trans_loss=trans_loss_per_elem.detach(),
                        q_trans=q_trans.detach(), action_trans=action_trans.detach(),
                        rot_q=rot_q.detach() if not self.use_memory else None,
                        action_rot_x_oh=action_rot_x_one_hot if not self.use_memory else None,
                        action_rot_y_oh=action_rot_y_one_hot if not self.use_memory else None,
                        action_rot_z_oh=action_rot_z_one_hot if not self.use_memory else None,
                        grip_q=grip_q.detach() if not self.use_memory else None,
                        action_grip_oh=action_grip_one_hot if not self.use_memory else None,
                        use_memory=self.use_memory,
                        num_rot_classes=self._num_rotation_classes,
                        reset_log=reset_log,
                        per_elem_total_loss=total_per_sample.detach(),
                    )

        if backprop:
            with autocast(enabled=self.amp):
                # cross-entropy loss
                trans_loss_per_elem = self._cross_entropy_loss(q_trans, action_trans)  # (bs, nc)
                trans_loss_per_sample = trans_loss_per_elem.mean(dim=1)  # (bs,)
                trans_loss = trans_loss_per_sample.mean()
                # Running per-sample total (detached copy for diagnostics).
                total_per_sample = trans_loss_per_sample.detach()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if not self.use_memory:
                    if self.add_rgc_loss:
                        rot_x_per_sample = self._cross_entropy_loss(
                            rot_q[
                                :,
                                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                            ],
                            action_rot_x_one_hot.argmax(-1),
                        )
                        rot_loss_x = rot_x_per_sample.mean()

                        rot_y_per_sample = self._cross_entropy_loss(
                            rot_q[
                                :,
                                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                            ],
                            action_rot_y_one_hot.argmax(-1),
                        )
                        rot_loss_y = rot_y_per_sample.mean()

                        rot_z_per_sample = self._cross_entropy_loss(
                            rot_q[
                                :,
                                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                            ],
                            action_rot_z_one_hot.argmax(-1),
                        )
                        rot_loss_z = rot_z_per_sample.mean()

                        grip_per_sample = self._cross_entropy_loss(
                            grip_q,
                            action_grip_one_hot.argmax(-1),
                        )
                        grip_loss = grip_per_sample.mean()

                        collision_per_sample = self._cross_entropy_loss(
                            collision_q, action_collision_one_hot.argmax(-1)
                        )
                        collision_loss = collision_per_sample.mean()

                        total_per_sample = (
                            trans_loss_per_sample.detach()
                            + rot_x_per_sample.detach()
                            + rot_y_per_sample.detach()
                            + rot_z_per_sample.detach()
                            + grip_per_sample.detach()
                            + collision_per_sample.detach()
                        )

                    total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                    )

                else:

                    total_loss = trans_loss


            # Save unscaled loss for logging before dividing
            total_loss_for_log = total_loss.item()

            # Average gradients across accumulation window
            total_loss = total_loss / self.gradient_accumulation_steps

            # Zero gradients at start of each accumulation window
            if step % self.gradient_accumulation_steps == 0:
                self._optimizer.zero_grad(set_to_none=True)

            # Backward — skip DDP all-reduce on non-final micro-steps
            is_accumulating = (step + 1) % self.gradient_accumulation_steps != 0
            if is_accumulating and isinstance(self._network, DistributedDataParallel):
                with self._network.no_sync():
                    self.scaler.scale(total_loss).backward()
            else:
                self.scaler.scale(total_loss).backward()

            # Step optimizer only at end of accumulation window
            if not is_accumulating:
                self.scaler.step(self._optimizer)
                self.scaler.update()
                self._lr_sched.step()

            loss_log = {
                "total_loss": total_loss_for_log,
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,
                "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,
                "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,
                "grip_loss": grip_loss.item() if not self.use_memory else None,
                "collision_loss": collision_loss.item() if not self.use_memory else None,
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

            if getattr(self, '_enable_diagnostics', True):
                kp_idx = replay_sample.get("keypoint_idx", None)
                manage_diag_log(
                    agent=self, tasks=tasks, keypoint_idx=kp_idx,
                    per_elem_trans_loss=trans_loss_per_elem.detach(),
                    q_trans=q_trans.detach(), action_trans=action_trans.detach(),
                    rot_q=rot_q.detach() if not self.use_memory else None,
                    action_rot_x_oh=action_rot_x_one_hot if not self.use_memory else None,
                    action_rot_y_oh=action_rot_y_one_hot if not self.use_memory else None,
                    action_rot_z_oh=action_rot_z_one_hot if not self.use_memory else None,
                    grip_q=grip_q.detach() if not self.use_memory else None,
                    action_grip_oh=action_grip_one_hot if not self.use_memory else None,
                    use_memory=self.use_memory,
                    num_rot_classes=self._num_rotation_classes,
                    reset_log=reset_log,
                    per_elem_total_loss=total_per_sample,
                )

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        # Note: the `step` argument is bound to rollout_generator's
        # step_signal.value which is never incremented (stays at -1). We ignore
        # it and use an agent-side counter (_current_step) instead.
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        # Resolve step_idx for the optional learnable step embedding. Prefer
        # the env-provided keypoint_idx (if the env exposes it), otherwise
        # fall back to the agent-side counter which resets in reset() and
        # increments once per act() call.
        raw_step = observation.get("keypoint_idx", None)
        env_provided_step = raw_step is not None
        if not env_provided_step:
            raw_step = torch.tensor([self._current_step], dtype=torch.long)
        step_idx = _normalize_step_idx(raw_step, proprio.device)

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
            step_idx=step_idx,
        )
        # Advance the agent-side counter only if the env didn't provide one.
        # Agent counter assumes act() is called exactly once per keypoint in
        # order (true for all current eval loops).
        if not env_provided_step:
            self._current_step += 1
        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        if pred_distri:
            x_distri = rot_grip_q[
                0,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ]
            y_distri = rot_grip_q[
                0,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ]
            z_distri = rot_grip_q[
                0,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ]
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            return ActResult(continuous_action)

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two and not self.use_memory:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans


    def get_gt_hm(
        self,
        wpt_local,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        # if self.stage_two and not self.use_memory:
        #     wpt_img2 = self._net_mod.get_pt_loc_on_img(
        #         wpt_local.unsqueeze(1),
        #         mvt1_or_mvt2=False,
        #         dyn_cam_info=dyn_cam_info,
        #         out=out,
        #     )
        #     assert wpt_img2.shape[1] == 1

        #     # (bs, 1, 2 * num_img, 2)
        #     wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
        #     nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans_hm = action_trans.view(bs, nc, h, w).clone()

        return action_trans_hm

    def reset(self):
        # Reset the agent-side keypoint counter used by the learnable step
        # embedding. Called at the start of every eval episode by
        # rollout_generator.generator(). Training path does not touch this
        # field — the replay buffer supplies keypoint_idx directly.
        self._current_step = 0

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
