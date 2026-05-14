import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.planner.critic import DualCritic, HabitatSymbolicCritic, VLMCritic
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import habitat_system_prompt
from embodiedbench.main import logger

link_path = os.path.join(os.path.dirname(__file__), '../envs/eb_habitat/data')
try:
    os.symlink(link_path, 'data')
except FileExistsError:
    pass 


example_path = os.path.join(os.path.dirname(__file__), 'config/habitat_examples.json')
examples = json.load(open(example_path, 'r+'))
system_prompt = habitat_system_prompt


class EB_HabitatEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None
        self.system_prompt = system_prompt

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
        
    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets
            
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = EBHabEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], exp_name=exp_name,
                                             start_epi_index=self.config.get('start_epi_index', 0), resolution=self.config.get('resolution', 500),
                                             recording=self.config.get('record_video', False))

            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, self.system_prompt, examples, n_shot=self.config['n_shots'], obs_key='head_rgb',
                                                 chat_history=self.config['chat_history'], language_only=self.config['language_only'], 
                                                 use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), tp=self.config.get('tp', 1))
            self.planner.log_path = self.env.log_path  # enable tree-structured planner debug logs

            # --- Dual-Critic setup ---
            use_critic = self.config.get('use_critic', False)
            self.dual_critic = None
            if use_critic:
                sym_critic = HabitatSymbolicCritic()
                vlm_critic = VLMCritic(
                    model=self.planner.model,
                    model_name=self.model_name,
                    env="habitat",
                    language_only=self.config.get('language_only', False),
                    n_shot=self.config.get('critic_n_shot', self.config.get('n_shots', 0)),
                )
                self.dual_critic = DualCritic(sym_critic, vlm_critic)
                self.dual_critic.log_path = self.env.log_path
                logger.info("[DualCritic] Enabled for this evaluation run.")

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        dual_critic = getattr(self, 'dual_critic', None)
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            if dual_critic is not None:
                dual_critic.reset()
            done = False
            info = {
                'task_success': 0, 'task_progress': 0, 'subgoal_reward': 0,
                'env_step': 0, 'env_feedback': '', 'last_action_success': 0,
                'episode_elapsed_seconds': 0,
            }
            while not done:
                try: 
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")

                    if action == -2: # empty plan stop here
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'env_step': self.env._current_step,
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'subgoal_reward': episode_info.get("subgoal_reward", 0),
                            'env_step': self.env._current_step,
                        }
                        break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'env_step': self.env._current_step,
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'subgoal_reward': episode_info.get("subgoal_reward", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    # multiple actions
                    if type(action) == list:
                        capped_actions = action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]
                        critic_triggered = False
                        for step_i, action_single in enumerate(capped_actions):
                            # --- Dual-Critic evaluation before execution ---
                            if dual_critic is not None:
                                remaining = [(a, self.env.language_skill_set[a]
                                              if isinstance(a, int) else a)
                                             for a in capped_actions[step_i:]]
                                scene_objects = self.env.get_scene_objects()
                                inventory_objects = self.env.get_inventory_objects()
                                critic_result = dual_critic.evaluate(
                                    action_id=action_single if isinstance(action_single, int) else -1,
                                    action_str=(self.env.language_skill_set[action_single]
                                                if isinstance(action_single, int) else str(action_single)),
                                    scene_objects=scene_objects,
                                    num_actions=len(self.env.language_skill_set),
                                    image_path=img_path,
                                    instruction=user_instruction,
                                    remaining_actions=remaining,
                                    is_first_step=(step_i == 0),
                                    inventory_objects=inventory_objects,
                                )
                                dual_critic._record_evaluation(
                                    env_step=self.env._current_step,
                                    planner_step=self.planner.planner_steps,
                                    action_step_in_plan=step_i,
                                    action_id=action_single if isinstance(action_single, int) else -1,
                                    action_str=(self.env.language_skill_set[action_single]
                                                if isinstance(action_single, int) else str(action_single)),
                                    image_path=img_path,
                                    remaining_actions=remaining,
                                    is_first_step=(step_i == 0),
                                    result=critic_result,
                                    vlm_prompt=critic_result.get("vlm_prompt"),
                                    inventory_objects=inventory_objects,
                                )
                                if not critic_result["valid"]:
                                    logger.info(f"[DualCritic] Replanning triggered at step {step_i}: "
                                                f"{critic_result['feedback']}")
                                    self.planner.update_critic_feedback(critic_result["feedback"])
                                    critic_triggered = True
                                    break  # exit action loop → outer while loop triggers replanning

                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            if done or info['last_action_success'] == 0:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                print(f"Info: {info['env_feedback']}")
                                break
                        if critic_triggered:
                            continue  # go back to planner.act() for replanning
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                            
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    print(e)
                    time.sleep(30)

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward']) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['subgoal_reward'] = info.get('subgoal_reward', 0)
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error

            # --- Replan rate ---
            num_replans = max(self.planner.planner_steps - 1, 0)
            episode_info['num_replans'] = num_replans
            episode_info['replan_rate'] = num_replans / info['env_step'] if info['env_step'] > 0 else 0.0

            # --- Invalid action metrics ---
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info['invalid_action_rate'] = (
                episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0.0
            )

            # --- JSON parse error rate ---
            episode_info['planner_json_error_rate'] = (
                self.planner.output_json_error / self.planner.planner_steps
                if self.planner.planner_steps > 0 else 0.0
            )

            # --- Critic metrics (only when critic is enabled) ---
            if dual_critic is not None:
                critic_records = dual_critic._episode_critic_records
                total_evals = len(critic_records)
                sym_rejects = sum(1 for r in critic_records if not r.get('symbolic_critic', {}).get('valid', True))
                vlm_rejects = sum(1 for r in critic_records
                                  if r.get('vlm_critic', {}).get('ran') and not r.get('vlm_critic', {}).get('valid', True))
                total_rejects = sum(1 for r in critic_records if not r.get('final_decision', {}).get('valid', True))
                critic_replans = sum(1 for act_id, _ in self.planner.episode_act_feedback if act_id == -3)
                episode_info['critic_total_evaluations'] = total_evals
                episode_info['critic_total_rejections'] = total_rejects
                episode_info['critic_rejection_rate'] = total_rejects / total_evals if total_evals > 0 else 0.0
                episode_info['critic_symbolic_rejections'] = sym_rejects
                episode_info['critic_vlm_rejections'] = vlm_rejects
                episode_info['critic_triggered_replans'] = critic_replans

            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)
            
            self.env.save_episode_log(fps=self.config.get('video_fps', 2))
            self.save_episode_metric(episode_info)
            episode_idx = self.env._current_episode_num
            self.planner.save_episode_planner_log(
                instruction=user_instruction,
                episode_idx=episode_idx,
            )
            if dual_critic is not None:
                dual_critic.save_episode_critic_log(
                    instruction=user_instruction,
                    episode_idx=episode_idx,
                )
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    import logging
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--start_epi_index', type=int, help='Starting episode index.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='number of tensor parallel splits of the model parameters')
        parser.add_argument('--record_video', type=int, help='Set to 1 to record annotated episode videos.')
        parser.add_argument('--use_critic', type=int, help='Set to 1 to enable the dual-critic module.')
        parser.add_argument('--critic_n_shot', type=int, help='Number of examples for the critic (overrides n_shots if set).')
        parser.add_argument('--video_fps', type=int, help='FPS for saved episode videos.')
        parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging verbosity level.')
        return parser.parse_args()

    config = {
        'model_name': 'gpt-4o-mini',
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote',
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0,
        'start_epi_index': 0,
        'eval_sets': ['base', 'common_sense', 'complex_instruction', 'spatial_relationship', 'visual_appearance', 'long_horizon'],
        'multistep': 0,
        'resolution': 500,
        'env_feedback': 1,
        'tp': 1,
        'record_video': 1,
        'use_critic': 0,       # set to 1 to enable the dual-critic module
        'critic_n_shot': 0,    # 0 = no few-shot examples for critic; set to -1 for all
        'video_fps': 2,
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    # Mirror what main.py does: configure logger level before anything runs
    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    logger.setLevel(log_level)

    evaluator = EB_HabitatEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()

    try:
        os.unlink('data')
        print(f"The symbolic link {link_path} has been successfully removed.")
    except FileNotFoundError:
        print(f"Error: The symbolic link {link_path} does not exist.")
    except OSError as e:
        print(f"Error: {e}")


