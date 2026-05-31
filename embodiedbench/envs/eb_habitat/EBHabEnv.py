"""
Habitat Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the habitat framework. It supports various object interactions and task scenarios.
The code is based on https://github.com/facebookresearch/habitat-lab and https://github.com/apple/ml-llarp 

Dependencies:
- habitat-lab
- gym
- numpy
- PIL
"""
import gym
import os
import time
import json
import imageio
from PIL import Image 
import numpy as np
import habitat
import hydra
from habitat.datasets import make_dataset
from embodiedbench.envs.eb_habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
)
from habitat.gym.gym_definitions import _add_sim_sensor_to_config
from omegaconf import OmegaConf

from habitat_sim.utils import viz_utils as vut
from embodiedbench.envs.eb_habitat.config import default_structured_configs
import embodiedbench.envs.eb_habitat.predicate_task
import embodiedbench.envs.eb_habitat.config
import embodiedbench.envs.eb_habitat.measures
from embodiedbench.envs.eb_habitat.utils import observations_to_image, merge_to_file, draw_text
from embodiedbench.main import logger

HABITAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/task/language_rearrangement.yaml')


ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 
        'spatial_relationship', 'visual_appearance', 'long_horizon'
    ] 

def add_receptacle(string, skill):
    if 'table_0' in skill[1][0]:
        string += 'table ' + skill[1][0].split('table_0')[1]
    elif 'fridge' in skill[1][0]:
        string += 'refrigerator push point'
    elif 'refrigerator' in skill[1][0]:
        string += 'refrigerator' 
    elif 'drawer_right' in skill[1][0]:
        string += 'right drawer of the kitchen counter'
    elif 'drawer_left' in skill[1][0]:
        string += 'left drawer of the kitchen counter'
    elif 'chair_0' in skill[1][0]:
        string += 'chair ' + skill[1][0].split('chair_0')[1]
    elif 'tvstand' in skill[1][0]:
        string += 'TV stand'
    elif 'counter_left' in skill[1][0]:
        string += 'left counter in the kitchen'
    elif 'counter_right' in skill[1][0]:
        string += 'right counter in the kitchen'
    elif 'sink' in skill[1][0]:
        string += 'sink in the kitchen'
    elif 'sofa' in skill[1][0]:
        string += 'sofa' 
    elif 'cab' in skill[1][0]:
        string += 'cabinet ' + skill[1][0].split('_')[-1]
    else:
        raise NotImplementedError
    return string


def transform_action_to_natural_language(skill_set):
    language_skill_set = []
    for skill in skill_set:
        if 'nav' in skill[0]:
            string = 'navigate to the '
            string = add_receptacle(string, skill)
        elif 'pick' in skill[0]:
            string = 'pick up the ' + skill[0].split('_')[1]
        elif 'open' in skill[0]:
            string = 'open the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'close' in skill[0]:
            string = 'close the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'place' in skill[0]:
            string = 'place at the '
            string = add_receptacle(string, skill)
        else:
            raise NotImplementedError
        
        language_skill_set.append(string)
    return language_skill_set



class EBHabEnv(gym.Env):
    def __init__(self, eval_set='long_horizon', exp_name='', down_sample_ratio=1.0, start_epi_index=0, resolution=500, recording=False):
        """
        Initialize the HabitatRearrange environment.
        """
        # load config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = habitat.get_config(HABITAT_CONFIG_PATH)
        _add_sim_sensor_to_config(self.config, ThirdRGBSensorConfig())
        # set the dataset
        assert eval_set in ValidEvalSets
        OmegaConf.set_readonly(self.config, False)
        self.config.habitat.dataset.data_path = os.path.join(os.path.dirname(__file__), 'datasets/{}.pickle'.format(eval_set))
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height = resolution
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width = resolution
        self.resolution = resolution

        # modify config path to ease data loading
        self.dataset = make_dataset(self.config.habitat.dataset.type, config=self.config.habitat.dataset)

        # initilaize env
        self.env = habitat.gym.make_gym_from_config(self.config, self.dataset)
        self.observation_space = self.env.observation_space
        # action of LanguageRearangeEnv is discrete value from 0 to 69
        self.action_space = self.env.action_space

        # Episode tracking
        self.down_sample_ratio = down_sample_ratio
        self.number_of_episodes = self.env.number_of_episodes * down_sample_ratio
        self._reset = False
        self._current_episode_num = 0 
        while start_epi_index >= 1 and self._current_episode_num < start_epi_index:
            self.env.reset(return_info=False)
            self._current_episode_num += 1

        self._current_step = 0
        # Long-horizon tasks require more object searches and pick-place cycles.
        # Give them a larger step / invalid-action budget to avoid early termination.
        if eval_set == 'long_horizon':
            self._max_episode_steps = 50
            self._max_invalid_actions = 15
        else:
            self._max_episode_steps = 30
            self._max_invalid_actions = 10
        self._cur_invalid_actions = 0
        self._episode_start_time = 0
        # is holding an object
        self.is_holding = False
        self.episode_log = []

        # init instruction and skill sets
        self.episode_language_instruction = ''
        self.episode_data = None
        self.skill_set = self.env.env.env._env.task.actions['pddl_hl_action']._action_datas
        self.language_skill_set = transform_action_to_natural_language(self.skill_set)

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 1
        self.log_path = 'running/eb_habitat/{}'.format(exp_name)
        # video recorder
        self.recording = recording
        self.episode_video = []
        self.seed(42)
        
    def current_episode(self, all_info: bool = False):
        return self.env.current_episode(all_info)


    def reset(self, **kwargs):
        """
        Reset the environment for a new episode. The env will iterate over all the task data from the dataset
        Returns: observation
        """
        assert self._current_episode_num <= self.number_of_episodes
        obs, info = self.env.reset(return_info=True, **kwargs)
        logger.info('Episode {}: {}'.format(str(self._current_episode_num), str(self.current_episode())))
        self.episode_language_instruction = info['lang_goal']
        self.episode_data = self.dataset.episodes[self._current_episode_num]
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        self.is_holding = False
        self._reset = True
        self.episode_log = []
        if self.recording:
            self.episode_video = []
            # capture the initial frame (before any action)
            frame = self.env.render("rgb_array")
            frame = self._annotate_frame(frame, action=None, info=None)
            self.episode_video.append(frame)
        self._episode_start_time = time.time()
        return obs

    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        if info['was_prev_action_invalid']:
            env_feedback = 'The action is invalid.'
            if 'pick' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot pick any object when holding something. Please place the object before picking something.'
                else:
                    env_feedback += ' Robot cannot pick any object that is not near the robot. Navigate to other place to find the object first.'
            elif 'place' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot place an object here. Navigate closer to the target receptacle.'
                else:
                    env_feedback += ' Robot cannot place any object when not holding something. Please pick the object before place it.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already open or the robot is not near the receptacle."
            elif 'close' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already closed or the robot is not near the receptacle."
        else:
            env_feedback = 'The action executed successfully'
            if 'pick' in info['action'] and self.feedback_verbosity:
                self.is_holding = True
                env_feedback += ' and you are holding {}.'.format(info['action'].split('(')[0].split('_')[1])
            elif 'place' in info['action'] and self.feedback_verbosity:
                self.is_holding = False
                env_feedback += ' and you are holding nothing.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is open.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is open.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            elif 'close' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is closed.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is closed.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            else:
                env_feedback += '.'
        
        # we don't use this info
        # env_feedback += ' The current task progress is {}.'.format(info['task_progress'])
        return env_feedback

    def step(self, action, reasoning='', **kwargs):
        """
        Execute a single environment step.
        Args:
            action (int): Index of action in action space
        Returns:
            tuple: (observation, reward, done, environment feedback)
        """
        assert self._reset, 'Reset env before stepping'
        self._current_step += 1
        obs, reward, done, info = self.env.step(action, **kwargs)
        if self.recording:
            frame = self.env.render("rgb_array")
            frame = self._annotate_frame(frame, action, info)
            self.episode_video.append(frame)

        if info['was_prev_action_invalid']:
            self._cur_invalid_actions += 1

        # if exceed the max step
        if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
            done = True
        # env feedback
        env_feedback = self.get_env_feedback(info)
        info['env_feedback'] = env_feedback
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time,
        info['action_id'] = action
        info['action_description'] = self.language_skill_set[action]
        info['reasoning'] = reasoning
        info['instruction'] = self.episode_language_instruction
        info['last_action_success'] = 1 - float(info['was_prev_action_invalid'])
        info['task_success'] = info['predicate_task_success']
        if info['task_success']:
            info['task_progress'] = 1.0
        self.episode_log.append(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_image(self, obs, key='head_rgb'):
        """Save current agent observation as a PNG image."""
        folder = self.log_path + '/images/episode_{}'.format(self._current_episode_num)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(observations_to_image(obs, key))
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(self._current_episode_num, self._current_step)) #, time_stamp))
        img.save(image_path)
        return image_path

    def _annotate_frame(self, frame: np.ndarray, action, info) -> np.ndarray:
        """Overlay task instruction and action text on a video frame."""
        from PIL import Image as PILImage, ImageDraw, ImageFont
        img = PILImage.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Try to use a slightly larger default font; fall back gracefully
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        # -- Instruction banner at the top --
        instruction = self.episode_language_instruction or ''
        # wrap long instructions
        max_chars = 60
        words = instruction.split()
        lines, current = [], ''
        for w in words:
            if len(current) + len(w) + 1 <= max_chars:
                current = (current + ' ' + w).strip()
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)

        y = 6
        for line in lines:
            # semi-transparent background rectangle
            bbox = draw.textbbox((8, y), line, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0, 160))
            draw.text((8, y), line, font=font, fill=(255, 255, 100))
            y += bbox[3] - bbox[1] + 4

        # -- Action label at the bottom --
        if action is not None:
            step = self._current_step
            if info is not None:
                action_str = info.get('action_description', '') or (
                    self.language_skill_set[action] if isinstance(action, int) else str(action))
                success = info.get('last_action_success', None)
                success_str = '' if success is None else (' ✓' if success else ' ✗')
            else:
                action_str = self.language_skill_set[action] if isinstance(action, int) else str(action)
                success_str = ''
            bottom_text = f'Step {step}: {action_str}{success_str}'
        else:
            bottom_text = 'Step 0: (initial observation)'

        h = img.height
        bbox = draw.textbbox((8, h - 28), bottom_text, font=font_small)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0, 160))
        draw.text((8, h - 28), bottom_text, font=font_small, fill=(100, 255, 100))

        return np.ascontiguousarray(img)

    def save_episode_log(self, fps: int = 2):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if self.episode_video:
            folder = os.path.join(self.log_path, 'video')
            os.makedirs(folder, exist_ok=True)
            video_path = os.path.join(
                folder,
                'video_episode_{}_steps_{}.mp4'.format(self._current_episode_num, self._current_step),
            )
            video_writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)
            for frame in self.episode_video:
                video_writer.append_data(np.ascontiguousarray(frame))
            video_writer.close()
            self.episode_video = []
            logger.info(f"Episode {self._current_episode_num} video saved to {video_path}")

    def render(self, mode: str = "rgb"):
        return self.env.render(mode)

    def get_scene_objects(self) -> list:
        """
        Return a list of objects known to be in the current scene.

        Habitat does not expose per-object scene metadata the way AI2-THOR does,
        so we return a lightweight representation derived from the action space:
        one entry per unique object type that has a 'pick up' action available.
        This is sufficient for the symbolic critic's range and holding-state checks;
        fine-grained object-availability checks are delegated to the VLM critic.

        Returns:
            list[dict]: list of dicts with at least an 'objectType' key.
        """
        objects = []
        seen = set()
        for action_str in self.language_skill_set:
            if action_str.startswith('pick up the '):
                obj_type = action_str[len('pick up the '):]
                if obj_type not in seen:
                    seen.add(obj_type)
                    objects.append({'objectType': obj_type, 'objectId': obj_type})
        return objects

    def get_inventory_objects(self) -> list:
        """
        Return the list of objects currently held by the robot.

        Habitat tracks holding state via the boolean ``self.is_holding``.
        When the robot is holding something we return a single placeholder
        dict so that the symbolic critic can detect the held-object conflict
        without needing the exact object type.

        Returns:
            list[dict]: one-element list with objectType when holding, else [].
        """
        if self.is_holding:
            return [{"objectType": "held_object", "objectId": "held_object"}]
        return []

    def get_metadata(self) -> dict:
        """Return scene metadata for initial 3D scene-graph construction.

        Habitat does not expose full per-object position data the way AI2-THOR
        does, so we derive a best-effort metadata dict from the action space:

        - ``objects``     : list of ``{"objectType": name}`` dicts for all
                            pick-up-able objects in the episode's action space.
        - ``receptacles`` : list of receptacle name strings inferred from
                            ``navigate to the …`` and ``place at the …`` actions.
        - ``is_holding``  : whether the robot is currently holding an object.
        - ``scene_id``    : task instruction string (episode identifier).

        Returns:
            dict: metadata snapshot.
        """
        objects: list = []
        seen_obj: set = set()
        receptacles: set = set()

        for action_str in self.language_skill_set:
            if action_str.startswith("pick up the "):
                obj_type = action_str[len("pick up the "):]
                if obj_type not in seen_obj:
                    seen_obj.add(obj_type)
                    objects.append({"objectType": obj_type, "objectId": obj_type})
            elif action_str.startswith("navigate to the "):
                receptacles.add(action_str[len("navigate to the "):])
            elif action_str.startswith("place at the "):
                receptacles.add(action_str[len("place at the "):])

        return {
            "objects": objects,
            "receptacles": list(receptacles),
            "scene_id": self.episode_language_instruction,
        }

    def close(self) -> None:
        """Terminate the environment."""
        self.env.close()


if __name__ == '__main__':
    """
    Example usage of the EBHabEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = EBHabEnv(eval_set='long_horizon', start_epi_index=0)
    obs = env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    print('Instruction: {}'.format(env.episode_language_instruction))
    print('log_path: {}'.format(env.log_path))
    env.save_image(obs)
    for _ in range(30):
        action = int(input('action id: ')) #env.action_space.sample()
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break

        obs_new, reward, done, info = env.step(action)
        print(reward, done, info)
        env.save_image(obs_new)
        if done:
            break
    env.close()
