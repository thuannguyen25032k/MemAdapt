# import torch
import re
import os
import numpy as np
import cv2
import json
# import lmdeploy
# from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from openai import OpenAI
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, truncate_message_prompts, fix_json
# from embodiedbench.planner.eb_navigation.RemoteModel_claude import RemoteModel
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.evaluator.config.visual_icl_examples.eb_navigation.ebnav_visual_icl import create_example_json_list
from embodiedbench.planner.planner_utils import template, template_lang
from embodiedbench.main import logger

template = template
template_lang = template_lang

MESSAGE_WINDOW_LEN = 5

class EBNavigationPlanner():
    def __init__(self, model_name = '', model_type = 'remote', actions = [], system_prompt = '', examples = '', n_shot=1, obs_key='head_rgb', chat_history=False, language_only=False, multiview = False, multistep = False, visual_icl = False, tp=1, truncate=False, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.truncate = truncate # whether to truncate message history when chat_history is True
        self.set_actions(actions)
        self.planner_steps = 0
        self.output_json_error = 0

        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.log_path = None  # set to env.log_path externally to enable debug logging
        self._episode_planner_records = []  # accumulate per-step records

        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl

        if not self.visual_icl:
            self.examples = examples[:n_shot]
            self.language_only = language_only
        else:
            self.examples = []
            self.language_only = False
            if language_only:
                self.icl_text_only = True
            else:
                self.icl_text_only = False


        self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[]):

        user_instruction = user_instruction.rstrip('.')

        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += self.first_prompt
     
        elif self.chat_history:

            # This is to support the sliding window feature
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                act_id = action_feedback[0]
                act_name = self.actions[act_id] if 0 <= act_id < len(self.actions) else '[unknown]'
                feedback_text = action_feedback[1] if len(action_feedback) > 1 else ''
                if act_id == -3:
                    prompt += '\n Step {}, [CRITIC FEEDBACK]: {}'.format(i, feedback_text)
                elif act_id == -2:
                    prompt += '\n Step {}, [PLANNER EMPTY PLAN]: {}'.format(i, feedback_text)
                elif act_id == -1:
                    prompt += '\n Step {}, [PLANNER INVALID ACTION]: {}'.format(i, feedback_text)
                else:
                    prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, act_id, act_name, feedback_text)

            prompt += f"\n\n{self.following_prompt}"

        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                act_id = action_feedback[0]
                act_name = self.actions[act_id] if 0 <= act_id < len(self.actions) else '[unknown]'
                feedback_text = action_feedback[1] if len(action_feedback) > 1 else ''
                if act_id == -3:
                    prompt += '\n Step {}, [CRITIC FEEDBACK]: {}'.format(i, feedback_text)
                elif act_id == -2:
                    prompt += '\n Step {}, [PLANNER EMPTY PLAN]: {}'.format(i, feedback_text)
                elif act_id == -1:
                    prompt += '\n Step {}, [PLANNER INVALID ACTION]: {}'.format(i, feedback_text)
                else:
                    prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, act_id, act_name, feedback_text)
            
            prompt += f"\n\n{self.following_prompt}"

        return prompt
    

    def get_message(self, image, prompt, messages=[]):

        if self.language_only:
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}],
            }
        elif self.multiview:
            data_url1 = local_image_to_data_url(image_path=image[0])
            data_url2 = local_image_to_data_url(image_path=image[1])
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url1,
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url2,
                        }
                    },
                    {"type": "text", "text": prompt}],
            }
        elif self.multistep:
            content = []
            for img_path in image:
                data_url = local_image_to_data_url(image_path=img_path)
                content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }) 
            content.append({"type": "text", "text": prompt})
            current_message = {
                "role":"user",
                "content":content
            }
        elif self.visual_icl:
            content = []
            content.append({"type": "text", "text": prompt})
            visual_example = create_example_json_list((not self.icl_text_only))
            content.extend(visual_example)
            content.append({"type": "text", "text": "Below is your current step observation, please starting planning to navigate to the target object by learning from the above-mentioned strategy and in-context learning examples. ### Output nothing else but a JSON string following the above mentioned format ###"})
            data_url = local_image_to_data_url(image_path=image)
            content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }) 
            current_message = {
                "role":"user",
                "content":content
            }
        else:
            data_url = local_image_to_data_url(image_path=image)
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }, 
                    {"type": "text", "text": prompt}],
            }

        messages = messages + [current_message]

        return messages[-MESSAGE_WINDOW_LEN:]


    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0
        self._episode_planner_records = []

    def _save_planner_log(self, prompt, obs, output):
        """Accumulate a full input/output record for the current planner step."""
        if self.log_path is None:
            return

        # Parse output into a structured form when possible
        try:
            output_parsed = json.loads(output)
        except Exception:
            output_parsed = output

        # Build typed action-history list
        action_history = []
        for i, fb in enumerate(self.episode_act_feedback):
            act_id = fb[0]
            if act_id == -3:
                entry = {
                    'history_step': i,
                    'entry_type':   'critic_feedback',
                    'action_id':    -3,
                    'action_name':  '[critic feedback]',
                    'feedback':     fb[1],
                }
            elif act_id == -2:
                entry = {
                    'history_step': i,
                    'entry_type':   'empty_plan',
                    'action_id':    -2,
                    'action_name':  '[empty plan]',
                    'feedback':     fb[1],
                }
            elif act_id == -1:
                entry = {
                    'history_step': i,
                    'entry_type':   'invalid_action',
                    'action_id':    -1,
                    'action_name':  '[invalid action]',
                    'feedback':     fb[1],
                }
            else:
                entry = {
                    'history_step': i,
                    'entry_type':   'env_step',
                    'action_id':    act_id,
                    'action_name':  (self.actions[act_id]
                                     if 0 <= act_id < len(self.actions)
                                     else f'[unknown id {act_id}]'),
                    'env_feedback': fb[1],
                }
            action_history.append(entry)

        self._episode_planner_records.append({
            'planner_step': self.planner_steps,
            'input': {
                'image':          obs if isinstance(obs, str) else '<numpy array>',
                'action_history': action_history,
                'prompt':         prompt,
            },
            'output': {
                'raw':    output,
                'parsed': output_parsed,
            },
        })

    def save_episode_planner_log(self, instruction='', episode_idx=None):
        """Write a complete tree-structured JSON debug log for the current episode."""
        if self.log_path is None or not self._episode_planner_records:
            return

        log_dir = os.path.join(self.log_path, 'planner_logs')
        os.makedirs(log_dir, exist_ok=True)
        suffix   = (f'episode_{episode_idx}' if episode_idx is not None
                    else f'episode_{len(self._episode_planner_records)}steps')
        log_file = os.path.join(log_dir, f'{suffix}.json')

        document = {
            'model_name':          self.model_name,
            'model_type':          self.model_type,
            'language_only':       self.language_only,
            'instruction':         instruction,
            'total_planner_steps': self.planner_steps,
            'total_json_errors':   self.output_json_error,
            'steps':               self._episode_planner_records,
        }
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        logger.info(f"Planner log saved to {log_file}")

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action

    def _action_name_to_id(self, action_name: str):
        """
        Look up an action id by its name string.
        Returns the matching id, or None if not found.
        Comparison is case-insensitive and strips extra whitespace.
        """
        name_lower = action_name.strip().lower()
        for idx, a in enumerate(self.actions):
            if a.strip().lower() == name_lower:
                return idx
        return None

    def _resolve_action_id(self, act_id: int, act_name: str) -> int:
        """
        Cross-check act_id against act_name.
        If they are inconsistent (the name at act_id does not match act_name),
        try to resolve via name lookup and return the correct id.
        Falls back to act_id if name lookup also fails.
        """
        if 0 <= act_id < len(self.actions):
            expected_name = self.actions[act_id].strip().lower()
            if act_name and act_name.strip().lower() != expected_name:
                resolved = self._action_name_to_id(act_name)
                if resolved is not None:
                    logger.warning(
                        f"action_id/action_name mismatch: id={act_id} "
                        f"('{self.actions[act_id]}') vs name='{act_name}'. "
                        f"Resolved to id={resolved} via name lookup."
                    )
                    return resolved
                else:
                    logger.warning(
                        f"action_id/action_name mismatch: id={act_id} "
                        f"('{self.actions[act_id]}') vs name='{act_name}'. "
                        f"Name not found in action list; keeping id={act_id}."
                    )
        return act_id

    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            json_object = json.loads(output_text)
            raw_items = json_object[json_key]
            action = []
            for item in raw_items:
                act_id   = item.get(self.action_key, -1)
                act_name = item.get('action_name', '')
                act_id   = self._resolve_action_id(act_id, act_name)
                action.append(act_id)
            if not len(action):
                print('empty plan, stop here')
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        logger.warning(
                            f"Invalid action id {act} at position {i} "
                            f"(valid range: 0~{len(self.actions)-1}). "
                            f"{'Rejecting full plan.' if i == 0 else f'Truncating plan to first {i} action(s).'}"
                        )
                        action = -1 if i == 0 else action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
        return action

        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        self._save_planner_log(prompt, obs, out)
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out


    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        if self.model_type == 'custom':
            return self.act_custom(prompt, obs)

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        # Apply truncation if chat_history and truncate are both True
        messages_to_send = self.episode_messages
        if self.chat_history and self.truncate:
            messages_to_send = truncate_message_prompts(self.episode_messages)
        
        for entry in messages_to_send:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        try:
            out = self.model.respond(messages_to_send)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
            
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action(out)
        self._save_planner_log(prompt, obs, out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])


        

