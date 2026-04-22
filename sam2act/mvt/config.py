# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

_C.depth = 8
_C.img_size = 220
_C.add_proprio = True
_C.proprio_dim = 4
_C.add_lang = True
_C.lang_dim = 512
_C.lang_len = 77
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.attn_dim = 512
_C.attn_heads = 8
_C.attn_dim_head = 64
_C.activation = "lrelu"
_C.weight_tie_layers = False
_C.attn_dropout = 0.1
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.norm_corr = False
_C.add_pixel_loc = True
_C.add_depth = True
_C.rend_three_views = False
_C.use_point_renderer = False
_C.pe_fix = True
_C.feat_ver = 0
_C.wpt_img_aug = 0.01
_C.inp_pre_pro = True
_C.inp_pre_con = True
_C.cvx_up = False
_C.xops = False
_C.rot_ver = 0
_C.num_rot = 72
_C.stage_two = False
_C.st_sca = 4
_C.st_wpt_loc_aug = 0.05
_C.st_wpt_loc_inp_no_noise = False
_C.img_aug_2 = 0.0

_C.ifSAM2 = True
_C.lora_finetune = True
_C.lora_r = 16
_C.ifsep = False
_C.resize_rgb = True
_C.use_memory = False
_C.num_maskmem = 7

_C.sam2_config = '/configs/sam2.1/sam2.1_hiera_b+'
_C.sam2_ckpt = './mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt'

# Learnable step embedding (Stage 0 only, additive on proprio features).
# max_period=100 gives distinct embeddings across 0..24 keypoints.
_C.use_step_embedding = False
_C.step_embedding_freq_size = 32
_C.step_embedding_max_period = 100
# Under Stage 2 (use_memory=True), mvt1's non-SAM2 params are frozen by default.
# When True, step_embedder MLP stays trainable so the proprio-path step signal
# can be fine-tuned at Stage 2 alongside memory training.
_C.train_step_embedder = False

# Phase-keyed memory retrieval (Stage 2). Injects the step-embedding signal
# into the memory attention's positional-encoding channel (K and Q sides)
# so retrieval scores by phase proximity in addition to visual similarity.
# No-op when use_phase_keyed_memory is False.
_C.use_phase_keyed_memory = False
_C.phase_key_injection = "both"       # one of: "both" | "write" | "read"
_C.phase_key_alpha = 1.0              # float scalar; initial value when learnable
_C.phase_key_alpha_learnable = False  # when True, alpha becomes an nn.Parameter

# Phase-persistent memory anchors (Stage 2). Adds P permanent slots to the
# memory bank that are never evicted. Phase transitions detected via
# gripper-open thresholding and/or step-embedding cosine delta. No-op when
# use_phase_anchors is False.
_C.use_phase_anchors = False
_C.max_phase_anchors = 6                   # P: max persistent anchor slots
_C.anchor_detect_gripper = True            # gripper open/close triggers transition
_C.anchor_detect_step_emb_delta = True     # cosine-delta on step embedding triggers
_C.anchor_step_emb_delta_threshold = 0.15  # cosine-distance threshold for the delta
_C.anchor_settle_frames = 2                # frames the candidate phase must hold

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
