import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger

# ---------------------------------------------------------------------------
# Optional memory imports — the planner works normally when memory is absent.
# ---------------------------------------------------------------------------
try:
    from embodiedbench.memory.manager import MemoryManager
    from embodiedbench.memory.prompt_formatter import MemoryPromptFormatter
    from embodiedbench.memory.base import MemoryQuery
    _MEMORY_AVAILABLE = True
except ImportError as _mem_import_err:
    _MEMORY_AVAILABLE = False
    logger.warning(f"[VLMPlanner] Memory package not available: {_mem_import_err}")

# ---------------------------------------------------------------------------
# Optional memory adapter imports — disabled gracefully if package not present.
# ---------------------------------------------------------------------------
try:
    from embodiedbench.memory_adapter.schemas import MemoryAdapterInput
    from embodiedbench.memory_adapter.adapter import build_planner_context as _build_planner_context
    from embodiedbench.memory_adapter.utils import is_unsafe_adapter_output
    _ADAPTER_AVAILABLE = True
except ImportError:
    _ADAPTER_AVAILABLE = False
    def is_unsafe_adapter_output(prompt, **_): return False  # noqa: E731

# Models that require a post-call sleep to respect rate limits.
_GEMINI_RATE_LIMITED_MODELS = frozenset({"gemini-1.5-pro", "gemini-2.0-flash"})

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

        # -----------------------------------------------------------------
        # Memory integration — disabled by default; activate via
        # set_memory_manager() or by assigning self.memory_manager directly.
        # -----------------------------------------------------------------
        self.memory_manager = None
        self.memory_formatter = MemoryPromptFormatter() if _MEMORY_AVAILABLE else None
        self.last_memory_context = None
        self.last_memory_prompt = ""
        self.current_instruction = ""
        self.current_info: dict = {}       # last info dict from update_info

        # -----------------------------------------------------------------
        # Memory Adapter — optional; overrides raw formatter when attached.
        # -----------------------------------------------------------------
        self.memory_adapter = None
        self.last_adapted_memory_output = None
        self.last_adapted_memory_prompt = ""

        # -----------------------------------------------------------------
        # Metrics — injected externally via set_metrics(); safe no-op when None.
        # -----------------------------------------------------------------
        self.metrics = None
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    # ------------------------------------------------------------------
    # Memory integration helpers
    # ------------------------------------------------------------------

    def set_memory_manager(self, memory_manager) -> None:
        """Attach (or detach, if None) a MemoryManager instance."""
        self.memory_manager = memory_manager

    def _memory_enabled(self) -> bool:
        return (
            _MEMORY_AVAILABLE
            and self.memory_manager is not None
            and self.memory_manager.is_enabled()
        )

    def set_memory_adapter(self, memory_adapter) -> None:
        """Attach (or detach, if None) a MemoryAdapter instance."""
        self.memory_adapter = memory_adapter
        # Forward log_path so the adapter writes debug logs to the same episode folder.
        if memory_adapter is not None and self.log_path:
            memory_adapter.log_path = self.log_path

    def set_metrics(self, metrics) -> None:
        """Attach a MemoryExperimentMetrics instance for counter tracking."""
        self.metrics = metrics

    def _adapter_enabled(self) -> bool:
        """True when a MemoryAdapter is attached and its config.enabled is True."""
        return (
            _ADAPTER_AVAILABLE
            and self.memory_adapter is not None
            and getattr(getattr(self.memory_adapter, "config", None), "enabled", True)
        )

    def _extract_known_objects(self, info: dict) -> list:
        """
        Safely pull a flat list of object name strings from an info dict.
        Handles scene_objects, inventory_objects, visible_objects, objects
        that may be either lists-of-dicts or lists-of-strings.
        """
        if not isinstance(info, dict):
            return []
        raw: list = []
        for key in ("scene_objects", "inventory_objects", "visible_objects", "objects"):
            val = info.get(key)
            if not val:
                continue
            if isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, dict):
                        name = item.get("objectType") or item.get("name") or item.get("object_type", "")
                        if name:
                            raw.append(str(name))
                    elif isinstance(item, str):
                        raw.append(item)
            elif isinstance(val, str):
                raw.append(val)
        return raw

    def _build_memory_query(
        self,
        instruction: str,
        info: dict = None,
    ) -> "MemoryQuery":
        """Build a MemoryQuery from current planner state."""
        info = info or {}

        # Scene / env identifiers
        env_name = str(info.get("env_name", "") or "")
        task_type = str(info.get("task_type", "") or info.get("scene_id", "") or info.get("scene", "") or "")

        # scene_name from spatial memory (AI2-THOR sceneName); "" for Habitat
        scene_name: Optional[str] = None
        if self._memory_enabled():
            spatial = getattr(self.memory_manager, "spatial", None)
            if spatial is not None:
                names = {n.scene for n in spatial.nodes.values() if n.scene}
                scene_name = next(iter(names), None)

        # Recent action texts (up to last 5 env steps, excluding sentinel ids)
        recent_actions = [
            self.actions[fb[0]]
            for fb in self.episode_act_feedback[-5:]
            if isinstance(fb[0], int) and 0 <= fb[0] < len(self.actions)
        ]

        return MemoryQuery(
            task_instruction=instruction,
            recent_actions=recent_actions,
            env_name=env_name,
            task_type=task_type,
            scene_name=scene_name,
        )

    def _get_planner_memory_prompt(
        self,
        instruction: str,
        info: dict = None,
    ) -> str:
        """
        Retrieve memory and return a formatted planner-safe string.

        Tries the MemoryAdapter first; falls back to the raw MemoryPromptFormatter.
        Returns ``""`` on any failure. Never crashes the planner.
        """
        if not self._memory_enabled():
            return ""

        # --- adapter path: short-circuit on cache hit (steps 1, 2, …) ---
        # Skip memory retrieval entirely — the adapter output is fixed for the episode.
        if self._adapter_enabled():
            cached_out = getattr(self.memory_adapter, "last_output", None)
            if cached_out is not None:
                adapted_prompt = _build_planner_context(cached_out)
                if adapted_prompt.strip():
                    logger.debug("[Memory] Reusing cached adapter output for planner.")
                    self.last_adapted_memory_output = cached_out
                    self.last_adapted_memory_prompt = adapted_prompt
                    self.last_memory_prompt = adapted_prompt
                    if self.metrics is not None:
                        self.metrics.planner_memory_injections += 1
                        self.metrics.planner_memory_prompt_chars += len(adapted_prompt)
                    return adapted_prompt
                # cached output exists but is empty → fall through to retrieval + raw formatter

        # --- memory retrieval (step 0, or when adapter is disabled / cache empty) ---
        query = self._build_memory_query(instruction, info=info)
        ctx = self.memory_manager.retrieve(query)  # 5 is the new top_k; can be tuned or made dynamic if needed
        self.last_memory_context = ctx

        if self.metrics is not None:
            self.metrics.memory_retrieval_calls += 1

        # --- adapter first call (step 0 only) ---
        if self._adapter_enabled():
            if self.metrics is not None:
                self.metrics.adapter_planner_calls += 1
                self.metrics.adapter_calls += 1
            try:
                adapter_input = MemoryAdapterInput(
                    task_instruction=instruction,
                    memory_context=ctx,
                )
                adapted_out = self.memory_adapter.adapt(adapter_input)
                self.last_adapted_memory_output = adapted_out
                adapted_prompt = _build_planner_context(adapted_out) if adapted_out else ""

                # Safety: reject code fences / bad adapter output
                if is_unsafe_adapter_output(adapted_prompt):
                    logger.warning(
                        "[Memory] Adapter output contained code fences; "
                        "falling back to raw formatter."
                    )
                    adapted_prompt = ""
                    if self.metrics is not None:
                        self.metrics.adapter_fallbacks += 1

                if adapted_prompt.strip():
                    self.last_adapted_memory_prompt = adapted_prompt
                    self.last_memory_prompt = adapted_prompt
                    if self.metrics is not None:
                        self.metrics.adapted_planner_prompt_chars += len(adapted_prompt)
                        self.metrics.planner_memory_injections += 1
                        self.metrics.planner_memory_prompt_chars += len(adapted_prompt)
                    return adapted_prompt
                else:
                    logger.debug(
                        "[Memory] Adapter returned empty planner context; "
                        "falling back to raw formatter."
                    )
                    if self.metrics is not None:
                        self.metrics.adapter_fallbacks += 1
            except Exception as adapter_err:
                logger.warning(
                    f"[Memory] MemoryAdapter.adapt() failed: {adapter_err}; "
                    "falling back to raw formatter."
                )
                if self.metrics is not None:
                    self.metrics.adapter_fallbacks += 1

        # --- raw formatter path (default / fallback) ---
        mem_prompt = self.memory_formatter.format_for_planner(ctx)
        self.last_memory_prompt = mem_prompt
        if mem_prompt and self.metrics is not None:
            self.metrics.planner_memory_injections += 1
            self.metrics.planner_memory_prompt_chars += len(mem_prompt)
        return mem_prompt

    # ------------------------------------------------------------------

    def get_availabel_action_prompt(self, available_actions):
        """
        Build a compact grouped action table from the action list.

        Groups actions by verb prefix (FIND, PICK UP, PUT DOWN, …) and formats
        each group as ``VERB: Object1(id1), Object2(id2), …``, reducing prompt
        length by ~60-70% versus the one-per-line format.
        """
        # Categorise each action by its verb prefix
        groups: dict[str, list[tuple[int, str]]] = {}
        verb_order = []

        verb_patterns = [
            ("FIND",     re.compile(r'^find a (.+)$',        re.I)),
            ("PICK UP",  re.compile(r'^pick up the (.+)$',   re.I)),
            ("PUT DOWN", re.compile(r'^put down (.+)$',      re.I)),
            ("DROP",     re.compile(r'^drop (.+)$',          re.I)),
            ("OPEN",     re.compile(r'^open the (.+)$',      re.I)),
            ("CLOSE",    re.compile(r'^close the (.+)$',     re.I)),
            ("TURN ON",  re.compile(r'^turn on the (.+)$',   re.I)),
            ("TURN OFF", re.compile(r'^turn off the (.+)$',  re.I)),
            ("SLICE",    re.compile(r'^slice the (.+)$',     re.I)),
            ("NAVIGATE", re.compile(r'^navigate to (.+)$',    re.I)),
            ("PLACE",    re.compile(r'^place at the (.+)$',     re.I)),
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
            entries = ", ".join(f"{obj}({idx})" for idx, obj in groups[verb])
            lines.append(f"  {verb.ljust(max_verb_len)}: {entries}")

        return "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _format_action_history(self, prev_act_feedback: list) -> str:
        """Render the action/feedback history as a multi-line string."""
        lines = []
        for i, action_feedback in enumerate(prev_act_feedback):
            act_id = action_feedback[0]
            act_name = self.actions[act_id] if 0 <= act_id < len(self.actions) else "[critic feedback]"
            feedback_text = action_feedback[1] if len(action_feedback) > 1 else ""
            if act_id == -3:
                lines.append('\nStep {}, [CRITIC FEEDBACK]: {}'.format(i, feedback_text))
            elif act_id == -2:
                lines.append('\nStep {}, [PLANNER EMPTY PLAN]: {}'.format(i, feedback_text))
            elif act_id == -1:
                lines.append('\nStep {}, [PLANNER INVALID ACTION]: {}'.format(i, feedback_text))
            elif self.use_feedback:
                lines.append('\nStep {}, action id {}, {}, env feedback: {}'.format(i, act_id, act_name, feedback_text))
            else:
                lines.append('\nStep {}, action id {}, {}'.format(i, act_id, act_name))
        return "".join(lines)

    def _replan_tail(self, user_instruction: str, prev_act_feedback: list) -> str:
        """Build the replanning instruction tail appended after the action history."""
        has_critic = any(fb[0] == -3 for fb in prev_act_feedback)
        last_is_critic = bool(prev_act_feedback) and prev_act_feedback[-1][0] == -3
        failure_clause = ("" if last_is_critic
                          else "and reason why the last action or plan failed and did not finish the task")
        critic_clause = (
            " You MUST address the MOST RECENT [CRITIC FEEDBACK] and explain why the proposed action was invalid before replanning."
            if has_critic else ""
        )
        env_fb = "and environment feedback " if self.use_feedback else ""
        action_range = f"0 ~ {len(self.actions)-1}"
        if self.language_only:
            return (
                f"\n\nConsidering the above interaction history and the current image, to achieve the human instruction: '{user_instruction}'. "
                f"You need to summarize interaction history {env_fb}{failure_clause}.{critic_clause} "
                f"Then, output a NEW plan to achieve the goal from the current state. "
                f"At the end, output the executable plan with action ids({action_range}) from the available actions."
            )
        return (
            f"\n\nConsidering the above interaction history and the current image, to achieve the human instruction: '{user_instruction}'. "
            f"You need to describe the current scene, summarize interaction history {env_fb}{failure_clause}.{critic_clause} "
            f"Then, output a NEW plan to achieve the goal from the current state. "
            f"At the end, output the executable plan with action ids({action_range}) from the available actions."
        )

    def process_prompt(self, user_instruction, prev_act_feedback=[], memory_prompt=''):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n## Now the human instruction is: {user_instruction}.'
            if memory_prompt:
                prompt += "\n\n" + memory_prompt
            if self.language_only:
                prompt += f"\n You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f"\n You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
        
        elif self.chat_history:
            prompt = memory_prompt + "\n\n" if memory_prompt else ""
            prompt += f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            prompt += self._format_action_history(prev_act_feedback)
            prompt += self._replan_tail(user_instruction, prev_act_feedback)
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n## Now the human instruction is: {user_instruction}.'
            if memory_prompt:
                prompt += "\n\n" + memory_prompt
            prompt += '\n\n The action history:'
            prompt += self._format_action_history(prev_act_feedback)
            prompt += self._replan_tail(user_instruction, prev_act_feedback)
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
            if isinstance(image, str):
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
        self.current_info = {}         # clear episode context; repopulated by set_episode_context()
        # Memory: clear per-episode transient state; long-term memories are preserved.
        self.last_memory_context = None
        self.last_memory_prompt = ""
        self.last_adapted_memory_output = None
        self.last_adapted_memory_prompt = ""
        # Clear the adapter cache so the critic cannot use a stale output from a prior episode.
        if self._adapter_enabled():
            try:
                self.memory_adapter.reset_last_output()
            except Exception:
                pass
        if self._memory_enabled():
            try:
                self.memory_manager.reset_episode()
            except Exception as e:
                logger.warning(f"[Memory] reset_episode failed: {e}")

    def set_episode_context(self, env_name: str = "", task_type: str = "") -> None:
        """Seed per-episode env/task identifiers so memory queries are fully populated.

        Call this once after ``reset()`` at the start of each episode.
        ``env_name`` : simulator name ("alfred", "habitat", "navigation", "manipulation").
        ``task_type`` : eval-set / task-type label (e.g. ``str(self.eval_set)``).
        """
        self.current_info["env_name"] = env_name
        self.current_info["task_type"] = task_type

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
                'prompt':         prompt,
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

    def _resolve_action_id(self, act_id, act_name: str) -> int:
        """
        Cross-check *act_id* against *act_name*; resolve via name lookup on
        mismatch.  Falls back to *act_id* if name lookup also fails.
        """
        # LLMs sometimes return action_id as a string (e.g. "11" instead of 11).
        # Coerce to int early so all downstream comparisons are type-safe.
        try:
            act_id = int(act_id)
        except (TypeError, ValueError):
            act_id = -1

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

        # Store instruction for use in update_info / memory update
        self.current_instruction = user_instruction
        
        memory_prompt = self._get_planner_memory_prompt(user_instruction, info=self.current_info)

        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback, memory_prompt=memory_prompt)

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

        if any(m in self.model_name for m in _GEMINI_RATE_LIMITED_MODELS):
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

    def update_info(self, info, metadata=None):
        """Update episode feedback history and memory."""
        # Merge env info into current_info, preserving env_name/task_type set by
        # set_episode_context() which the env's info dict does not contain.
        self.current_info.update(info or {})
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])

        # Memory update — no-op when disabled.
        if self._memory_enabled():
            action_id = info.get('action_id', -1)
            action_text = (
                self.actions[action_id]
                if isinstance(action_id, int) and 0 <= action_id < len(self.actions)
                else ""
            )
            self.memory_manager.update(
                task_instruction=self.current_instruction,
                info=info,
                metadata=metadata,
                action=action_id,
                action_text=action_text,
                env_feedback=str(info.get('env_feedback', '') or ''),
                success=info.get('last_action_success'),
                step_id=None,
            )
