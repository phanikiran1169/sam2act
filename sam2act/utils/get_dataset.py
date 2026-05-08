
import os
import sys
import shutil
import torch
import clip

from sam2act.libs.peract.helpers.utils import extract_obs
from sam2act.utils.rvt_utils import ForkedPdb
from sam2act.utils.dataset import create_replay, fill_replay, create_replay_temporal, fill_replay_temporal
from sam2act.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)
from sam2act.utils.replay_buffer_config import build_per_task_dict
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
    demo_aug_every_n_per_task=None,
):
    # Resolve per-task augmentation from replay_buffer.yaml if not provided
    if demo_aug_every_n_per_task is None:
        demo_aug_every_n_per_task = build_per_task_dict(tasks)
    missing = [t for t in tasks if t not in demo_aug_every_n_per_task]
    if missing:
        raise ValueError(f"demo_aug_every_n_per_task missing tasks: {missing}")

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )
    if not only_train:
        test_replay_buffer = create_replay(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
        )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=demo_aug_every_n_per_task[task],
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
        )

        if not only_train:
            # print("----- Test Buffer -----")
            fill_replay(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=0,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=demo_aug_every_n_per_task[task],
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=data_path_val,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        # Val uses random sampling with task_uniform distribution so every
        # task is represented proportionally. Enumerate mode walks tasks in
        # insertion order and would under-sample any task beyond the first
        # when val_iters * batch_size < first-task transition count.
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="random",
            num_workers=num_workers,
            sample_distribution_mode="task_uniform",
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset


def get_dataset_temporal(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    num_maskmem,
    rank,
    sample_distribution_mode="transition_uniform",
    val_from_train=False,
    val_start_idx=0,
    demo_aug_every_n_per_task=None,
):
    # Resolve per-task augmentation from replay_buffer.yaml if not provided
    if demo_aug_every_n_per_task is None:
        demo_aug_every_n_per_task = build_per_task_dict(tasks)
    missing = [t for t in tasks if t not in demo_aug_every_n_per_task]
    if missing:
        raise ValueError(f"demo_aug_every_n_per_task missing tasks: {missing}")

    # Size replay capacity from on-disk transitions per task to avoid silent
    # circular-overwrite truncation when total transitions exceed the default
    # cap (3e5). Bookkeeping arrays scale linearly with capacity but stay
    # small with disk_saving=True (~17 MB per million slots).
    def _count_replay_files(storage_dir, task_list):
        total = 0
        per_task = {}
        for t in task_list:
            d = os.path.join(storage_dir, t)
            if os.path.isdir(d):
                n = sum(1 for f in os.listdir(d) if f.endswith(".replay"))
            else:
                n = 0
            per_task[t] = n
            total += n
        return total, per_task

    train_total, train_per_task = _count_replay_files(TRAIN_REPLAY_STORAGE_DIR, tasks)
    test_total = 0
    if not only_train:
        test_total, _ = _count_replay_files(TEST_REPLAY_STORAGE_DIR, tasks)

    # 1.5x margin so the buffer never wraps; floor at 3e5 (legacy default) so
    # small runs are unaffected.
    train_replay_size = max(int(3e5), int(max(train_total, 1) * 1.5))
    test_replay_size = max(int(3e5), int(max(test_total, 1) * 1.5))
    if rank == 0:
        print(f"[replay-buffer] train tasks={len(tasks)} "
              f"transitions={train_total} capacity={train_replay_size} "
              f"per_task={train_per_task}", flush=True)
        if not only_train:
            print(f"[replay-buffer] test transitions={test_total} "
                  f"capacity={test_replay_size}", flush=True)

    train_replay_buffer = create_replay_temporal(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
        num_maskmem=num_maskmem,
        replay_size=train_replay_size,
    )
    if not only_train:
        test_replay_buffer = create_replay_temporal(
            batch_size=BATCH_SIZE_TEST,
            timesteps=1,
            disk_saving=True,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            num_maskmem=num_maskmem,
            replay_size=test_replay_size,
        )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")
            if os.path.exists(test_replay_storage_folder) and os.path.isdir(
                test_replay_storage_folder
            ):
                shutil.rmtree(test_replay_storage_folder)
                print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay_temporal(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=demo_aug_every_n_per_task[task],
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
            rank=rank,
        )

        if not only_train:
            val_data_path = data_path_train if val_from_train else data_path_val
            val_sidx = val_start_idx if val_from_train else 0
            fill_replay_temporal(
                replay=test_replay_buffer,
                task=task,
                task_replay_storage_folder=test_replay_storage_folder,
                start_idx=val_sidx,
                num_demos=NUM_VAL,
                demo_augmentation=True,
                demo_augmentation_every_n=demo_aug_every_n_per_task[task],
                cameras=CAMERAS,
                rlbench_scene_bounds=SCENE_BOUNDS,
                voxel_sizes=VOXEL_SIZES,
                rotation_resolution=ROTATION_RESOLUTION,
                crop_augmentation=False,
                data_path=val_data_path,
                episode_folder=EPISODE_FOLDER,
                variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
                clip_model=clip_model,
                device=device,
                rank=rank,
            )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        # Val uses random sampling with task_uniform distribution so every
        # task is represented proportionally. Enumerate mode walks tasks in
        # insertion order and would under-sample any task beyond the first
        # when val_iters * batch_size < first-task transition count.
        test_wrapped_replay = PyTorchReplayBuffer(
            test_replay_buffer,
            sample_mode="random",
            num_workers=num_workers,
            sample_distribution_mode="task_uniform",
        )
        test_dataset = test_wrapped_replay.dataset()
    return train_dataset, test_dataset



if __name__ == "__main__":
    get_dataset_sequence(
        ['close_jar'],
        10,
        None,
        'replay/replay_sequence',
        None,
        'data',
        100,
        None,
        False,
        'cuda:0',
        num_workers=3,
        only_train=True,
        sample_distribution_mode='task_uniform',
        if_chunk=True,
    )