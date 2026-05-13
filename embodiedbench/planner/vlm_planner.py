import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
        self.log_path = None  # set externally (e.g. env.log_path) to enable debug logging
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        """
        Build a compact grouped action table instead of listing every action on its own line.

        Groups actions by verb (find, pick up, put down, drop, open, close, turn on, turn off, slice)
        and formats each group as:
            FIND   : Cart(0), Potato(1), ...
            PICK UP: KeyChain(80), Potato(81), ...
            ...
        This reduces prompt length by ~60-70% versus the one-per-line format while
        preserving all action-id information.
        """
        import re as _re

        # Categorise each action by its verb prefix
        groups: dict[str, list[tuple[int, str]]] = {}
        verb_order = []

        verb_patterns = [
            ("FIND",     _re.compile(r'^find a (.+)$',        _re.I)),
            ("PICK UP",  _re.compile(r'^pick up the (.+)$',   _re.I)),
            ("PUT DOWN", _re.compile(r'^put down (.+)$',      _re.I)),
            ("DROP",     _re.compile(r'^drop (.+)$',          _re.I)),
            ("OPEN",     _re.compile(r'^open the (.+)$',      _re.I)),
            ("CLOSE",    _re.compile(r'^close the (.+)$',     _re.I)),
            ("TURN ON",  _re.compile(r'^turn on the (.+)$',   _re.I)),
            ("TURN OFF", _re.compile(r'^turn off the (.+)$',  _re.I)),
            ("SLICE",    _re.compile(r'^slice the (.+)$',     _re.I)),
        ]

        for idx, action in enumerate(available_actions):
            matched = False
            for verb, pat in verb_patterns:
                m = pat.match(action.strip())
                if m:
                    obj = m.group(1)
                    if verb not in groups:
                        groups[verb] = []
                        verb_order.append(verb)
                    groups[verb].append((idx, obj))
                    matched = True
                    break
            if not matched:
                # Keep any unrecognised action verbatim under "OTHER"
                if "OTHER" not in groups:
                    groups["OTHER"] = []
                    verb_order.append("OTHER")
                groups["OTHER"].append((idx, action))

        lines = []
        max_verb_len = max(len(v) for v in verb_order)
        for verb in verb_order:
            items = groups[verb]
            if verb in ("PUT DOWN", "DROP", "OTHER"):
                # These usually have only one entry; keep them inline
                entries = ", ".join(f"{obj}({idx})" for idx, obj in items)
            else:
                entries = ", ".join(f"{obj}({idx})" for idx, obj in items)
            lines.append(f"  {verb.ljust(max_verb_len)}: {entries}")

        return "\n" + "\n".join(lines)

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            if self.language_only:
                prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
        
        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                act_id = action_feedback[0]
                act_name = self.actions[act_id] if 0 <= act_id < len(self.actions) else "[critic feedback]"
                feedback_text = action_feedback[1] if len(action_feedback) > 1 else ""

                if act_id == -3:
                    prompt += '\nStep {}, [CRITIC FEEDBACK]: {}'.format(i, feedback_text)
                elif act_id == -2:
                    prompt += '\nStep {}, [PLANNER EMPTY PLAN]: {}'.format(i, feedback_text)
                elif act_id == -1:
                    prompt += '\nStep {}, [PLANNER INVALID ACTION]: {}'.format(i, feedback_text)
                elif self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, act_id, act_name, feedback_text)
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, act_id, act_name)

            has_critic = any(fb[0] == -3 for fb in prev_act_feedback)
            last_is_critic = bool(prev_act_feedback) and prev_act_feedback[-1][0] == -3
            failure_clause = (""
                              if last_is_critic else
                              "and reason why the last action or plan failed and did not finish the task")
            critic_clause = (" In addition, you MUST consider the MOST RECENT [CRITIC FEEDBACK] to reason why the proposed action is invalid and replan accordingly." if has_critic else "")
            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}{failure_clause}.{critic_clause} Output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}{failure_clause}.{critic_clause} Output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                act_id = action_feedback[0]
                act_name = self.actions[act_id] if 0 <= act_id < len(self.actions) else "[critic feedback]"
                feedback_text = action_feedback[1] if len(action_feedback) > 1 else ""

                if act_id == -3:
                    prompt += '\nStep {}, [CRITIC FEEDBACK]: {}'.format(i, feedback_text)
                elif act_id == -2:
                    prompt += '\nStep {}, [PLANNER EMPTY PLAN]: {}'.format(i, feedback_text)
                elif act_id == -1:
                    prompt += '\nStep {}, [PLANNER INVALID ACTION]: {}'.format(i, feedback_text)
                elif self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, act_id, act_name, feedback_text)
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, act_id, act_name)

            has_critic = any(fb[0] == -3 for fb in prev_act_feedback)
            last_is_critic = bool(prev_act_feedback) and prev_act_feedback[-1][0] == -3
            failure_clause = (""
                              if last_is_critic else
                              "and reason why the last action or plan failed and did not finish the task")
            critic_clause = (" In addition, you MUST consider the MOST RECENT [CRITIC FEEDBACK] to reason why the proposed action is invalid and replan accordingly." if has_critic else "")
            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and {failure_clause}.{critic_clause} Output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}{failure_clause}.{critic_clause} Output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        return prompt
    

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image 
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            if self.multistep: # handle multiple images
                ind = int(image_path.split('step_')[-1].strip('.png'))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind +1):
                    temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0
        self._episode_log_path = None  # reset per-episode log file path
        self._episode_planner_records = []  # accumulate step records for tree-structured log

    def update_critic_feedback(self, feedback: str):
        """
        Inject critic rejection as a special entry in episode_act_feedback so
        that the planner's next prompt includes the critic's reasoning.
        The action_id -3 is the sentinel for 'critic rejection'.
        """
        msg = str(feedback).strip()
        # Remove critic-source wrappers — the prompt already labels the entry [CRITIC FEEDBACK].
        msg = re.sub(r'\[(?:Symbolic|VLM)\s+Critic\]\s*', '', msg, flags=re.IGNORECASE).strip()
        msg = re.sub(r'^\[Critic\]\s*', '', msg, flags=re.IGNORECASE).strip()
        if not msg:
            msg = "The proposed next action is not appropriate. Please replan accordingly."
        # Store the clean message without any prefix; process_prompt labels it [CRITIC FEEDBACK].
        self.episode_act_feedback.append((-3, msg))
        logger.info(f"[VLMPlanner] Critic feedback injected into planner history: {msg}")

    def _save_planner_log(self, prompt, obs, output):
        """Accumulate a full input/output record for the current planner step."""
        if self.log_path is None:
            return

        # --- Parse output ---
        try:
            output_parsed = json.loads(output)
        except Exception:
            output_parsed = output

        # --- Build action-history list with explicit entry types ---
        action_history = []
        for i, fb in enumerate(self.episode_act_feedback):
            act_id = fb[0]
            if act_id == -3:
                entry = {
                    'history_step':  i,
                    'entry_type':    'critic_feedback',
                    'action_id':     -3,
                    'action_name':   '[critic feedback]',
                    'feedback':      fb[1],
                }
            elif act_id == -2:
                entry = {
                    'history_step':  i,
                    'entry_type':    'empty_plan',
                    'action_id':     -2,
                    'action_name':   '[empty plan]',
                    'feedback':      fb[1],
                }
            elif act_id == -1:
                entry = {
                    'history_step':  i,
                    'entry_type':    'invalid_action',
                    'action_id':     -1,
                    'action_name':   '[invalid action]',
                    'feedback':      fb[1],
                }
            else:
                entry = {
                    'history_step':  i,
                    'entry_type':    'env_step',
                    'action_id':     act_id,
                    'action_name':   (self.actions[act_id]
                                      if 0 <= act_id < len(self.actions)
                                      else f'[unknown id {act_id}]'),
                    'env_feedback':  fb[1],
                }
            action_history.append(entry)

        self._episode_planner_records.append({
            'planner_step': self.planner_steps,
            'input': {
                'image':          obs if isinstance(obs, str) else '<numpy array>',
                'action_history': action_history,
                'prompt':         prompt,          # full string — easy to read / diff
                # 'prompt_lines':   prompt.splitlines(),  # line-by-line for tree viewers
            },
            'output': {
                'raw':    output,
                'parsed': output_parsed,
            },
        })

    def save_episode_planner_log(self, instruction='', episode_idx=None):
        """Write a complete tree-structured JSON log for the current episode."""
        if self.log_path is None or not self._episode_planner_records:
            return
        log_dir = os.path.join(self.log_path, 'planner_logs')
        os.makedirs(log_dir, exist_ok=True)
        suffix   = (f'episode_{episode_idx}' if episode_idx is not None
                    else f'episode_{len(self._episode_planner_records)}steps')
        log_file = os.path.join(log_dir, f'{suffix}.json')

        # --- Episode-level summary counters ---
        critic_injections = sum(
            1 for r in self._episode_planner_records
            for h in r['input']['action_history']
            if h['entry_type'] == 'critic_feedback'
        )
        invalid_actions = sum(
            1 for r in self._episode_planner_records
            for h in r['input']['action_history']
            if h['entry_type'] == 'invalid_action'
        )

        document = {
            'model_name':           self.model_name,
            'model_type':           self.model_type,
            'language_only':        self.language_only,
            'instruction':          instruction,
            'total_planner_steps':  self.planner_steps,
            'total_json_errors':    self.output_json_error,
            'total_critic_injections': critic_injections,
            'total_invalid_actions':   invalid_actions,
            'steps': self._episode_planner_records,
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
                # Mismatch — try to find the correct id from act_name
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
                        logger.warning(f"Invalid action id {act} at position {i} (valid range: 0~{len(self.actions)-1}). "
                                       f"{'Rejecting full plan.' if i == 0 else f'Truncating plan to first {i} action(s).'}")
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
        return action

    
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
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
        # some models do not support json scheme, add style into prompt
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name or 'Qwen3-VL' in self.model_name or self.model_type == 'custom':
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        logger.debug(f"Model Output:\n{out}\n")
        self._save_planner_log(prompt, obs, out)

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])
