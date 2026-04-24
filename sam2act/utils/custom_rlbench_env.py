# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from sam2act.libs.peract.helpers.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv,
    _resolve_cam_placeholder,
)
from yarr.utils.process_str import change_case


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        # Re-pose the eval recording camera for the active task. Task swaps
        # change which cinematic placeholder applies; without this the camera
        # stays stuck on whichever placeholder was chosen at launch.
        if self.eval and self._record_cam is not None:
            task_name = change_case(self._task._task.__class__.__name__)
            cam_placeholder = _resolve_cam_placeholder(task_name)
            self._record_cam.set_pose(cam_placeholder.get_pose())

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
