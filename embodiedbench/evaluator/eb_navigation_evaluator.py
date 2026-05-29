import os
import time
import numpy as np
from tqdm import tqdm
import json
import argparse
import logging

from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
from embodiedbench.planner.nav_planner import EBNavigationPlanner
from embodiedbench.planner.nav_critic import NavigationSymbolicCritic, NavigationDualCritic
from embodiedbench.planner.critic import VLMCritic
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import update_config_with_args

from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
from embodiedbench.evaluator.config.eb_navigation_example import examples
from embodiedbench.main import logger
from embodiedbench.memory.integration import (
    create_memory_manager_from_config,
    attach_memory_to_planner,
    attach_memory_to_critic,
    finalize_memory_episode,
    save_memory_if_configured,
    create_memory_adapter_from_config,
    attach_memory_adapter_to_planner,
    attach_memory_adapter_to_critic,
    unload_memory_adapter,
    setup_memory_experiment,
)

system_prompt = eb_navigation_system_prompt

class EB_NavigationEvaluator():
    def __init__(self, config):

        self.model_name = config['model_name']
        self.eval_sets = config.get("eval_sets", ValidEvalSets)
        self.eval_set = None
        self.config = config

        self.env = None
        self.planner = None
        self.memory_manager = None
        self.memory_adapter = None
        self.dual_critic = None

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
        self.eval_sets = list(valid_eval_sets)
        if isinstance(self.eval_sets, list) and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets
            
        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"

            self.env = EBNavigationEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                   exp_name=exp_name, multiview=self.config['multiview'], boundingbox=self.config['detection_box'], 
                                   multistep = self.config['multistep'], resolution = self.config['resolution'],
                                   selected_indexes=self.config.get('selected_indexes', []))

            self.planner = EBNavigationPlanner(model_name=self.model_name, model_type = self.config['model_type'], 
                                           actions = self.env.language_skill_set, system_prompt = system_prompt, 
                                           examples = examples, n_shot=self.config['n_shots'], obs_key='head_rgb', 
                                           chat_history=self.config['chat_history'], language_only=self.config['language_only'], 
                                           multiview=self.config['multiview'], multistep = self.config['multistep'], 
                                           visual_icl = self.config['visual_icl'], truncate=self.config.get('truncate', False))

            # --- Memory + MemoryAdapter setup (experiment-mode aware) ---
            self.memory_manager, self.memory_adapter = setup_memory_experiment(
                self.config, self.planner, None
            )

            # Enable per-episode planner debug logs
            self.planner.log_path = self.env.log_path

            # --- Dual-Critic setup ---
            use_critic = self.config.get('use_critic', False)
            self.dual_critic = None
            if use_critic:
                sym_critic = NavigationSymbolicCritic()
                vlm_critic = VLMCritic(
                    model=self.planner.model,
                    model_name=self.model_name,
                    env="eb_navigation",
                    language_only=self.config.get('language_only', False),
                    n_shot=self.config.get('critic_n_shot', self.config.get('n_shots', 0)),
                )
                self.dual_critic = NavigationDualCritic(sym_critic, vlm_critic)
                self.dual_critic.log_path = self.env.log_path
                # forward memory to critic
                attach_memory_to_critic(self.dual_critic, self.memory_manager)
                attach_memory_adapter_to_critic(self.dual_critic, self.memory_adapter)
                logger.info("[NavigationDualCritic] Enabled for this evaluation run.")

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key = None)
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))
            save_memory_if_configured(self.memory_manager, self.config, on_run_end=True)
            unload_memory_adapter(self.memory_adapter)

    def evaluate(self):
        dual_critic = self.dual_critic
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")
            self.planner.reset()
            self.planner.set_episode_context(env_name="navigation", task_type=str(self.eval_set))
            if dual_critic is not None:
                dual_critic.reset()
            done = False
            # track recent executed action ids for symbolic rotation-loop detection
            recent_action_ids: list = []
            info = {
                'task_success': 0, 'env_step': 0,
                'env_feedback': '', 'last_action_success': 0,
                'episode_elapsed_seconds': 0,
            }
            while not done:
                try:
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")

                    # Handle sentinel returns from planner
                    if action == -2:  # empty plan
                        episode_info['empty_plan'] = 1
                        logger.info("Empty plan returned by planner, stopping episode.")
                        break
                    if action == -1:  # JSON parse / invalid action id
                        logger.info("Invalid action id returned by planner, replanning.")
                        continue

                    reasoning_parsed = json.loads(reasoning) if isinstance(reasoning, str) else reasoning

                    if isinstance(action, list):
                        capped_actions = action[:min(
                            self.env._max_episode_steps - self.env._current_step + 1,
                            len(action)
                        )]
                        critic_triggered = False
                        full_plan = [
                            (a, self.env.language_skill_set[a]
                                if 0 <= a < len(self.env.language_skill_set) else str(a))
                            for a in capped_actions
                        ]
                        for step_i, action_single in enumerate(capped_actions):
                            action_str = (
                                self.env.language_skill_set[action_single]
                                if 0 <= action_single < len(self.env.language_skill_set)
                                else str(action_single)
                            )

                            if dual_critic is not None:
                                critic_result = dual_critic.evaluate(
                                    action_id=action_single,
                                    action_str=action_str,
                                    scene_objects=[],
                                    num_actions=len(self.env.language_skill_set),
                                    image_path=img_path,
                                    instruction=user_instruction,
                                    full_plan=full_plan,
                                    current_index=step_i,
                                    is_first_step=(step_i == 0 and self.planner.planner_steps <= 1),
                                    inventory_objects=[],
                                    info=info,
                                    recent_action_ids=recent_action_ids,
                                )
                                if not critic_result["valid"]:
                                    feedback = critic_result["feedback"]
                                    logger.info(f"[DualCritic] Rejected action '{action_str}': {feedback}")
                                    self.planner.update_critic_feedback(feedback)
                                    critic_triggered = True
                                    break

                            i_flag = 1 if step_i == 0 else 0
                            obs, reward, done, info = self.env.step(action_single, reasoning_parsed, i_flag)
                            recent_action_ids.append(action_single)
                            if len(recent_action_ids) > 8:
                                recent_action_ids.pop(0)
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)

                            if done:
                                break
                            if info['last_action_success'] == 0:
                                print('invalid action, start replanning')
                                break

                        if critic_triggered:
                            continue  # replan without executing
                    else:
                        action_str = (
                            self.env.language_skill_set[action]
                            if 0 <= action < len(self.env.language_skill_set)
                            else str(action)
                        )
                        single_plan = [(action, action_str)]

                        if dual_critic is not None:
                            critic_result = dual_critic.evaluate(
                                action_id=action,
                                action_str=action_str,
                                scene_objects=[],
                                num_actions=len(self.env.language_skill_set),
                                image_path=img_path,
                                instruction=user_instruction,
                                full_plan=single_plan,
                                current_index=0,
                                is_first_step=(self.planner.planner_steps <= 1),
                                inventory_objects=[],
                                info=info,
                                recent_action_ids=recent_action_ids,
                            )
                            if not critic_result["valid"]:
                                feedback = critic_result["feedback"]
                                logger.info(f"[DualCritic] Rejected action '{action_str}': {feedback}")
                                self.planner.update_critic_feedback(feedback)
                                continue  # replan

                        obs, reward, done, info = self.env.step(action, reasoning_parsed, 1)
                        recent_action_ids.append(action)
                        if len(recent_action_ids) > 8:
                            recent_action_ids.pop(0)
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)

                except Exception as e:
                    time.sleep(1)
                    print(e)
                    print("retrying...")


            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward']) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info['subgoal_reward'] = info.get('subgoal_reward', 0)
            num_valid_actions = info["env_step"] - episode_info['num_invalid_actions']
            episode_info['num_valid_actions'] = num_valid_actions
            episode_info['eff_rate'] = (
                num_valid_actions / info["env_step"] if info["env_step"] > 0 else 0.0
            )
            num_replans = max(self.planner.planner_steps - 1, 0)
            episode_info['num_replans'] = num_replans
            episode_info['replan_rate'] = num_replans / info['env_step'] if info['env_step'] > 0 else 0.0
            episode_info['planner_json_error_rate'] = (
                self.planner.output_json_error / self.planner.planner_steps
                if self.planner.planner_steps > 0 else 0.0
            )
            episode_info["episode_elapsed_seconds"] = info.get(
                "episode_elapsed_seconds", time.time() - self.env._episode_start_time
            )

            # --- Critic metrics ---
            if dual_critic is not None:
                critic_records = dual_critic._episode_critic_records
                total_evals   = len(critic_records)
                sym_rejects   = sum(1 for r in critic_records if not r.get('symbolic_critic', {}).get('valid', True))
                vlm_rejects   = sum(1 for r in critic_records
                                    if r.get('vlm_critic', {}).get('ran') and not r.get('vlm_critic', {}).get('valid', True))
                total_rejects = sum(1 for r in critic_records if not r.get('final_decision', {}).get('valid', True))
                critic_replans = sum(1 for act_id, _ in self.planner.episode_act_feedback if act_id == -3)
                episode_info['critic_total_evaluations']  = total_evals
                episode_info['critic_total_rejections']   = total_rejects
                episode_info['critic_rejection_rate']     = total_rejects / total_evals if total_evals > 0 else 0.0
                episode_info['critic_symbolic_rejections'] = sym_rejects
                episode_info['critic_vlm_rejections']     = vlm_rejects
                episode_info['critic_triggered_replans']  = critic_replans

            episode_idx = (
                self.env._current_episode_num
                if not len(self.env.selected_indexes)
                else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
            )
            self.save_episode_metric(episode_info)
            self.env.save_episode_video(fps=self.config.get('video_fps', 2))
            self.planner.save_episode_planner_log(
                instruction=user_instruction,
                episode_idx=episode_idx,
            )
            if dual_critic is not None:
                dual_critic.save_episode_critic_log(
                    instruction=user_instruction,
                    episode_idx=episode_idx,
                )

            # --- Memory: finalize episode and save ---
            finalize_memory_episode(
                self.memory_manager, self.planner,
                task_instruction=user_instruction,
                info=info,
                env_name="navigation",
                task_type=str(self.eval_set),
                episode_idx=getattr(self.env, '_current_episode_num', None),
                extra_metadata={"model_name": self.model_name, "eval_set": str(self.eval_set)},
            )
            save_memory_if_configured(self.memory_manager, self.config, on_episode_end=True)

            progress_bar.update()

    def check_config_valid(self):
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multiview, multistep, visual_icl, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0


if __name__ == '__main__':

    def parse_arguments():
        parser = argparse.ArgumentParser(description='Run EB-Navigation evaluation.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of few-shot examples.')
        parser.add_argument('--down_sample_ratio', type=float, help='Down-sample ratio for the dataset.')
        parser.add_argument('--model_type', type=str, help='Type of the model (remote/local).')
        parser.add_argument('--language_only', type=int, help='Set to 1 for language-only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to 1 to enable chat history.')
        parser.add_argument('--detection_box', type=int, help='Set to 1 to enable bounding-box detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Set to 1 to enable multi-step mode.')
        parser.add_argument('--multiview', type=int, help='Set to 1 to enable multi-view mode.')
        parser.add_argument('--resolution', type=int, help='Image resolution.')
        parser.add_argument('--visual_icl', type=int, help='Set to 1 to enable visual in-context learning.')
        parser.add_argument('--truncate', type=int, help='Set to 1 to truncate chat history.')
        parser.add_argument('--selected_indexes', type=lambda s: list(map(int, s.split(','))),
                            help='Comma-separated episode indexes to evaluate.')
        parser.add_argument('--use_critic', type=int, help='Set to 1 to enable the dual-critic module.')
        parser.add_argument('--critic_n_shot', type=int, help='Number of few-shot examples for the critic.')
        parser.add_argument('--video_fps', type=int, help='FPS for saved episode videos.')
        parser.add_argument('--log_level', type=str, default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                            help='Logging verbosity level.')
        return parser.parse_args()

    config = {
        'model_name': 'gpt-4o-mini',
        'down_sample_ratio': 1.0,
        'model_type': 'remote',
        'language_only': False,
        'eval_sets': ['base'],
        'chat_history': True,
        'action_num_per_plan': 5,
        'fov': 100,
        'n_shots': 1,
        'sleep_time': 0,
        'multiview': 0,
        'detection_box': 0,
        'multistep': 0,
        'resolution': 500,
        'exp_name': 'test',
        'visual_icl': 0,
        'truncate': False,
        'selected_indexes': [],
        'use_critic': 0,
        'critic_n_shot': 0,
        'video_fps': 2,
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    logger.setLevel(log_level)

    evaluator = EB_NavigationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()



