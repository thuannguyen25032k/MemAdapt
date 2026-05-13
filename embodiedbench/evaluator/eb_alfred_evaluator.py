import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.planner.critic import DualCritic, SymbolicCritic, VLMCritic
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import alfred_system_prompt
from embodiedbench.main import logger

example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_examples.json')
exploration_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_long_horizon_examples.json')
system_prompt = alfred_system_prompt

class EB_AlfredEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
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
            self.env = EBAlfEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                          exp_name=exp_name, selected_indexes=self.config.get('selected_indexes', []), 
                                          detection_box=self.config.get('detection_box', False),
                                          resolution=self.config.get('resolution', 500), 
                                          )
            examples = json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, system_prompt, examples, n_shot=self.config['n_shots'], 
                                            obs_key='head_rgb', chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                            use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), tp=self.config.get('tp', 1))
            self.planner.log_path = self.env.log_path  # enable prompt/output debug logging

            # --- Dual-Critic setup ---
            use_critic = self.config.get('use_critic', False)
            self.dual_critic = None
            if use_critic:
                sym_critic = SymbolicCritic()
                vlm_critic = VLMCritic(
                    model=self.planner.model,
                    model_name=self.model_name,
                    language_only=self.config.get('language_only', False),
                    examples_path=self.config.get(
                        'critic_examples_path',
                        os.path.join(os.path.dirname(__file__), 'config', 'critic_examples.json')
                    ),
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

            # Set up the planner for the new episode
            self.planner.reset()
            if dual_critic is not None:
                dual_critic.reset()
            # update the action space for alfred due to dynamic objects
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            info = {
                'task_success': 0, 'task_progress': 0, 'env_step': 0,
                'env_feedback': '', 'last_action_success': 0,
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
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    
                    # mutiple actions
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
                            if done or not info['last_action_success']:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                print(f"Info: {info['env_feedback']}")
                                break
                        if critic_triggered:
                            continue  # go back to planner.act() for replanning
                    else: # single action
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
            episode_info['task_progress'] = info['task_progress']
            episode_info['num_steps'] = info['env_step']
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error

            # --- Replan rate: how often the planner had to replan per env step ---
            # planner_steps includes the initial plan, so replans = planner_steps - 1
            num_replans = max(self.planner.planner_steps - 1, 0)
            episode_info['num_replans'] = num_replans
            episode_info['replan_rate'] = num_replans / info['env_step'] if info['env_step'] > 0 else 0.0

            # --- Invalid action metrics ---
            episode_info['invalid_action_rate'] = (
                episode_info['num_invalid_actions'] / info['env_step'] if info['env_step'] > 0 else 0.0
            )

            # --- JSON parse error rate (planner output malformed) ---
            episode_info['planner_json_error_rate'] = self.planner.output_json_error / self.planner.planner_steps if self.planner.planner_steps > 0 else 0.0

            # --- Critic metrics (only when critic is enabled) ---
            if dual_critic is not None:
                critic_records = dual_critic._episode_critic_records
                total_evals = len(critic_records)
                sym_rejects = sum(1 for r in critic_records if not r.get('symbolic_critic', {}).get('valid', True))
                vlm_rejects = sum(1 for r in critic_records
                                  if r.get('vlm_critic', {}).get('ran') and not r.get('vlm_critic', {}).get('valid', True))
                total_rejects = sum(1 for r in critic_records if not r.get('final_decision', {}).get('valid', True))
                # Critic-triggered replans: entries in act_feedback with action_id == -3
                critic_replans = sum(1 for act_id, _ in self.planner.episode_act_feedback if act_id == -3)
                episode_info['critic_total_evaluations'] = total_evals
                episode_info['critic_total_rejections'] = total_rejects
                episode_info['critic_rejection_rate'] = total_rejects / total_evals if total_evals > 0 else 0.0
                episode_info['critic_symbolic_rejections'] = sym_rejects
                episode_info['critic_vlm_rejections'] = vlm_rejects
                episode_info['critic_triggered_replans'] = critic_replans

            episode_info['episode_elapsed_seconds'] = info.get('episode_elapsed_seconds', time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            self.env.save_episode_video(fps=self.config.get('video_fps', 2))
            episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
            self.planner.save_episode_planner_log(instruction=user_instruction, episode_idx=episode_idx)
            if dual_critic is not None:
                dual_critic.save_episode_critic_log(instruction=user_instruction, episode_idx=episode_idx)
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
        parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='number of tensor parallel splits of the model parameters')
        parser.add_argument('--use_critic', type=int, help='Set to 1 to enable the dual-critic module.')
        parser.add_argument('--critic_n_shot', type=int, help='Number of examples for the critic (overrides n_shots if set).')
        parser.add_argument('--critic_examples_path', type=str, help='Path to critic few-shot examples JSON.')
        parser.add_argument('--video_fps', type=int, help='FPS for saved episode videos.')
        parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging verbosity level.')
        return parser.parse_args()


    config = {
        'model_name': 'gpt-4o-mini',
        'n_shots': 1,
        'down_sample_ratio': 1.0,
        'model_type': 'remote',
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0,
        'detection_box': 0,
        'eval_sets': ['base'],
        'selected_indexes': [0, 1, 2, 3, 4],
        'multistep': 0,
        'resolution': 500,
        'env_feedback': 1,
        'tp': 1,
        'use_critic': 0,          # set to 1 to enable the dual-critic module
        'critic_n_shot': 0,       # null → uses n_shots value (10); set 0 to disable examples
        'critic_examples_path': None,
        'video_fps': 2,
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    # Mirror what main.py does: configure logger level before anything runs
    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    logger.setLevel(log_level)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()




