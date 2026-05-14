"""
Dual-Critic Module for EmbodiedBench.

Consists of:
  - SymbolicCritic : verifies action-ID bounds and object availability in scene metadata.
  - VLMCritic      : uses the same VLM as the planner (different prompt) to assess whether
                     the single NEXT action is appropriate given the current visual state.
                     The remaining follow-up steps are provided as context only (not judged).
  - DualCritic     : orchestrates both critics with the key rule that the FIRST action of
                     every plan is checked only by the SymbolicCritic (preventing an infinite
                     replanning loop where the VLM critic always demands replanning before
                     any action is ever executed).

Prompts and few-shot examples are loaded from:
  - embodiedbench/evaluator/config/system_prompts.py  →  alfred_critic_system_prompt
  - embodiedbench/evaluator/config/critic_examples.json
"""

import re
import json
import os
from embodiedbench.main import logger
from embodiedbench.planner.planner_utils import local_image_to_data_url, fix_json

# ---------------------------------------------------------------------------
# Paths to external prompt / example files
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.join(os.path.dirname(__file__),
                           '..', 'evaluator', 'config')

def _load_critic_system_prompt(env: str) -> str:
    """Import and return env_critic_system_prompt from system_prompts.py."""
    if env.lower() == 'habitat':
        from embodiedbench.evaluator.config.system_prompts import habitat_critic_system_prompt
        return habitat_critic_system_prompt
    elif env.lower() == 'alfred':
        from embodiedbench.evaluator.config.system_prompts import alfred_critic_system_prompt
        return alfred_critic_system_prompt

def _load_critic_examples(env: str) -> list:
    """Load few-shot critic examples from JSON file."""
    path = os.path.join(_CONFIG_DIR, f'{env}_critic_examples.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"VLMCritic: failed to load critic examples from '{path}': {e}. "
                       "Running without few-shot examples.")
        return []

def _format_examples(examples: list) -> str:
    """Render the few-shot examples list into a human-readable string block."""
    if not examples:
        return "(no examples provided)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i}")
        lines.append(f"Instruction: {ex.get('instruction', '')}")
        lines.append(f"Next action: {ex.get('next_action', '')}")
        followup = ex.get('followup_steps', '')
        if followup:
            lines.append(f"Follow-up steps (context):\n{followup}")
        if ex.get('observation_description'):
            lines.append(f"Observation: {ex['observation_description']}")
        out = ex.get('output', {})
        lines.append(f"Critic output: {json.dumps(out)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbolic Critic
# ---------------------------------------------------------------------------
class AlfredSymbolicCritic:
    """
    Checks two things without calling any model:
      1. The action ID is within the valid action-space range.
      2. The object referenced by the action string exists in the current scene metadata.
    """

    # Regex patterns to extract the object name from each action type
    _ACTION_PATTERNS = [
        r'^find a (.+)$',
        r'^pick up the (.+)$',
        r'^open the (.+)$',
        r'^close the (.+)$',
        r'^turn on the (.+)$',
        r'^turn off the (.+)$',
        r'^slice the (.+)$',
    ]
    # Actions that require no specific target object
    _NO_OBJECT_ACTIONS = {"put down the object in hand", "drop the object in hand"}

    def evaluate(self, action_id: int, action_str: str,
                 scene_objects: list, num_actions: int,
                 inventory_objects: list = None) -> dict:
        """
        Returns:
            dict with keys:
              - valid  (bool)
              - reason (str)
        """
        if inventory_objects is None:
            inventory_objects = []

        # 1. Range check
        if action_id < 0 or action_id >= num_actions:
            return {
                "valid": False,
                "reason": (f"Action id {action_id} is out of the valid range "
                           f"(0 ~ {num_actions - 1}).")
            }

        # 2a. Holding conflict check — robot cannot pick up while already holding an object
        if re.match(r'^pick up the .+$', action_str.strip(), re.IGNORECASE):
            if inventory_objects:
                held_type = inventory_objects[0].get('objectType', 'unknown object')
                return {
                    "valid": False,
                    "reason": (f"Robot is currently holding '{held_type}' and cannot pick up "
                               f"another object. Put down or drop the held object first."),
                }

        # 2b. Empty-hand check — robot cannot put down or drop while holding nothing
        if re.match(r'^(put down|drop) .+$', action_str.strip(), re.IGNORECASE):
            if not inventory_objects:
                return {
                    "valid": False,
                    "reason": ("Robot is not holding any object, so 'put down' / 'drop' is "
                               "invalid. Pick up an object first."),
                }

        # 3. Object availability check
        obj_name = self._parse_object(action_str)
        if obj_name is None:
            return {"valid": True, "reason": "Action requires no specific target object."}

        # Build lookup sets from scene metadata
        obj_types_lower = {obj['objectType'].lower() for obj in scene_objects}
        obj_ids         = {obj['objectId']   for obj in scene_objects}

        # Handle numbered duplicates like "Fridge_2" → base type "fridge"
        base_name = obj_name.lower().split('_')[0]

        if base_name not in obj_types_lower and obj_name not in obj_ids:
            return {
                "valid": False,
                "reason": (f"Object '{obj_name}' (base type: '{base_name}') was not found "
                           f"in the current scene metadata.")
            }

        return {"valid": True,
                "reason": f"Object '{obj_name}' is present in the current scene."}

    def _parse_object(self, action_str: str):
        """Extract the target object name from an action string, or None if no object."""
        action_str = action_str.strip()
        if action_str in self._NO_OBJECT_ACTIONS:
            return None
        for pattern in self._ACTION_PATTERNS:
            m = re.match(pattern, action_str, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# Habitat Symbolic Critic
# ---------------------------------------------------------------------------
class HabitatSymbolicCritic:
    """
    Checks three things without calling any model for the Habitat environment:
      1. The action ID is within the valid action-space range.
      2. Pick conflict: robot cannot pick up an object while already holding one.
      3. Place conflict: robot cannot place an object when not holding anything.

    Unlike Alfred, Habitat does not expose per-object scene metadata, so there
    is no object-availability check — that is delegated to the VLM critic.
    """

    def evaluate(self, action_id: int, action_str: str,
                 scene_objects: list, num_actions: int,
                 inventory_objects: list = None) -> dict:
        """
        Returns:
            dict with keys:
              - valid  (bool)
              - reason (str)
        """
        if inventory_objects is None:
            inventory_objects = []

        # 1. Range check
        if action_id < 0 or action_id >= num_actions:
            return {
                "valid": False,
                "reason": (f"Action id {action_id} is out of the valid range "
                           f"(0 ~ {num_actions - 1}).")
            }

        action_lower = action_str.strip().lower()

        # 2. Pick conflict — robot cannot pick while already holding an object
        if action_lower.startswith('pick up the '):
            if inventory_objects:
                held = inventory_objects[0].get('objectType', 'an object')
                return {
                    "valid": False,
                    "reason": (f"Robot is currently holding '{held}' and cannot pick up "
                               "another object. Place the held object first."),
                }

        # 3. Place conflict — robot cannot place while holding nothing
        if action_lower.startswith('place at the '):
            if not inventory_objects:
                return {
                    "valid": False,
                    "reason": ("Robot is not holding any object, so 'place' is invalid. "
                               "Pick up an object first."),
                }

        return {"valid": True, "reason": "Action preconditions are satisfied."}


# ---------------------------------------------------------------------------
# VLM Critic
# ---------------------------------------------------------------------------
class VLMCritic:
    """
    Uses the same RemoteModel instance as the planner, but with a different
    prompt and output schema (critic_schema), to assess whether the remaining
    plan is still feasible given the current visual state.

    The system prompt is loaded from:
        embodiedbench/evaluator/config/system_prompts.py  (alfred_critic_system_prompt)
    Few-shot examples are loaded from:
        embodiedbench/evaluator/config/critic_examples.json
    """

    def __init__(self, model, model_name: str, env: str, 
                 language_only: bool = False, n_shot: int = 0):
        """
        Args:
            model          : the RemoteModel instance shared with VLMPlanner.
            model_name     : model identifier string (for routing decisions).
            language_only  : if True, no image is attached to the critic message.
            examples_path  : path to the critic few-shot examples JSON file.
            n_shot         : number of few-shot examples to include in the prompt
                             (0 = no examples; None or -1 = use all available).
        """
        self.model         = model
        self.model_name    = model_name
        self.language_only = language_only
        self.n_shot        = n_shot

        self._system_prompt_template = _load_critic_system_prompt(env)
        self._examples               = _load_critic_examples(env)

    def _select_examples(self) -> list:
        """
        Return the subset of loaded examples to inject into the prompt.

        Rules:
          - n_shot == 0          → no examples (empty list)
          - n_shot is None or -1 → all available examples
          - n_shot > 0           → first min(n_shot, len(examples)) examples
        """
        if self.n_shot == 0:
            return []
        if self.n_shot is None or self.n_shot < 0:
            return self._examples
        return self._examples[:self.n_shot]

    def evaluate(self, image_path: str, instruction: str,
                 remaining_actions: list) -> dict:
        """
        Args:
            image_path        : path to the current observation image.
            instruction       : task instruction string.
            remaining_actions : list of (action_id, action_name) tuples for steps
                                that have NOT yet been executed.
                                remaining_actions[0] is the NEXT action to judge;
                                remaining_actions[1:] are follow-up steps shown as context.
        Returns:
            dict with keys: valid (bool), reason (str), suggestions (str)
        """
        if not remaining_actions:
            return {"valid": True, "reason": "No remaining actions to evaluate.",
                    "suggestions": "", "_prompt": ""}

        # Split: first item is the judgment target; rest are context only
        next_aid, next_aname = remaining_actions[0]
        next_action_str = f"action id {next_aid}, {next_aname}"

        if len(remaining_actions) > 1:
            followup_str = '\n'.join(
                f"  Step {i + 1}: action id {aid}, {aname}"
                for i, (aid, aname) in enumerate(remaining_actions[1:])
            )
        else:
            followup_str = "(none — this is the last planned step)"

        examples_str = _format_examples(self._select_examples())
        prompt = self._system_prompt_template.format(
            instruction=instruction,
            next_action=next_action_str,
            followup_steps=followup_str,
            examples=examples_str,
        )

        # Build message — optionally include the current image
        if self.language_only:
            messages = [{"role": "user",
                         "content": [{"type": "text", "text": prompt}]}]
        else:
            try:
                data_url = local_image_to_data_url(image_path=image_path)
                messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": prompt},
                ]}]
            except Exception as e:
                logger.warning(f"VLM critic: failed to load image '{image_path}': {e}. "
                               "Falling back to text-only evaluation.")
                messages = [{"role": "user",
                             "content": [{"type": "text", "text": prompt}]}]
                
        try:
            out = self.model.respond(messages)
            result = json.loads(out)
            return {
                "valid":       bool(result.get("valid", True)),
                "reason":      str(result.get("reason", "")),
                "suggestions": str(result.get("suggestions", "")),
                "_prompt":     prompt,
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"VLM critic JSON parse failed ({e}); trying regex fallback.")
            fallback = self._regex_fallback(out)
            if fallback is not None:
                fallback["_prompt"] = prompt
                return fallback
            logger.warning("VLM critic regex fallback also failed. Defaulting to valid=True.")
            return {"valid": True, "reason": "Critic evaluation failed; defaulting to valid.",
                    "suggestions": "", "_prompt": prompt}

    # ------------------------------------------------------------------
    # Regex fallback parser for malformed critic output
    # ------------------------------------------------------------------
    @staticmethod
    def _regex_fallback(text: str):
        """
        Try to extract valid/reason/suggestions from raw model output when
        json.loads() fails.  Returns a dict or None if extraction fails.
        """
        if not text:
            return None
        try:
            # --- valid ---
            # Matches:  "valid": true / "valid": false  (with or without quotes around value)
            m_valid = re.search(r'"valid"\s*:\s*(true|false)', text, re.I)
            if m_valid is None:
                # Heuristic: look for explicit reject/invalid keywords
                text_lower = text.lower()
                if any(kw in text_lower for kw in ("invalid", "reject", "not valid",
                                                    "should not", "cannot", "can not")):
                    valid = False
                else:
                    valid = True
            else:
                valid = m_valid.group(1).lower() == "true"

            # --- reason ---
            m_reason = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            reason = m_reason.group(1).strip() if m_reason else text.strip()[:300]

            # --- suggestions ---
            m_sugg = re.search(r'"suggestions"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            suggestions = m_sugg.group(1).strip() if m_sugg else ""

            return {"valid": valid, "reason": reason, "suggestions": suggestions}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Dual Critic
# ---------------------------------------------------------------------------
class DualCritic:
    """
    Orchestrates SymbolicCritic + VLMCritic with the following rule:

      - is_first_step=True  → run ONLY SymbolicCritic.
        (Prevents an infinite loop where VLM critic always rejects the first
         action before anything is ever executed.)
      - is_first_step=False → run SymbolicCritic first; if it passes, run VLMCritic.
        VLMCritic evaluates ONLY the single next action (remaining_actions[0]),
        using the rest of the plan as context to understand intent.

    Returns a unified result dict with a human-readable `feedback` field that
    the planner can directly append to its next prompt.

    Also accumulates per-call records during an episode and writes them as a
    tree-structured JSON log via save_episode_critic_log().
    """

    def __init__(self, symbolic_critic: AlfredSymbolicCritic, vlm_critic: VLMCritic,
                 log_path: str = None):
        self.symbolic  = symbolic_critic
        self.vlm       = vlm_critic
        self.log_path  = log_path   # set externally (e.g. env.log_path) to enable logging
        self._episode_critic_records: list = []   # filled by _record_evaluation()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        """Clear accumulated records at the start of a new episode."""
        self._episode_critic_records = []

    # ------------------------------------------------------------------
    # Internal: record one evaluation call
    # ------------------------------------------------------------------
    def _record_evaluation(self,
                           env_step: int,
                           planner_step: int,
                           action_step_in_plan: int,
                           action_id: int,
                           action_str: str,
                           image_path: str,
                           remaining_actions: list,
                           is_first_step: bool,
                           result: dict,
                           vlm_prompt: str = None,
                           inventory_objects: list = None):
        """
        Append a structured record of one DualCritic.evaluate() call.

        Args:
            env_step           : env._current_step at evaluation time.
            planner_step       : planner.planner_steps at evaluation time.
            action_step_in_plan: index of this action within the current plan batch (0-based).
            action_id          : action ID being evaluated.
            action_str         : human-readable action string.
            image_path         : path to the observation image used.
            remaining_actions  : list of (id, name) tuples for un-executed steps.
                                 remaining_actions[0] is the next action (judgment target);
                                 remaining_actions[1:] are follow-up steps (context only).
            is_first_step      : whether VLM critic was skipped.
            result             : full dict returned by evaluate().
            vlm_prompt         : the prompt string sent to the VLM (optional; may be None
                                 if VLM was skipped).
        """
        # Split remaining_actions into judgment target and follow-up context
        next_action = (
            {"action_id": remaining_actions[0][0], "action_name": remaining_actions[0][1]}
            if remaining_actions else None
        )
        followup_steps = [
            {"step": i + 1, "action_id": aid, "action_name": aname}
            for i, (aid, aname) in enumerate(remaining_actions[1:])
        ]

        record = {
            "env_step":            env_step,
            "planner_step":        planner_step,
            "action_step_in_plan": action_step_in_plan,
            "input": {
                "image":         image_path,
                "action_id":     action_id,
                "action_str":    action_str,
                "is_first_step": is_first_step,
                # next_action is always remaining_actions[0] — what VLMCritic judges
                "next_action":   next_action,
                # followup_steps are remaining_actions[1:] — shown to VLM as context only
                "followup_steps": followup_steps,
                # objects currently held by the robot (from inventoryObjects metadata)
                "inventory_objects": [obj.get('objectType', obj.get('objectId', ''))
                                      for obj in (inventory_objects or [])],
            },
            "symbolic_critic": {
                "ran":    True,
                "valid":  result["symbolic_result"]["valid"],
                "reason": result["symbolic_result"]["reason"],
            },
            "vlm_critic": (
                {
                    "ran": False,
                    "skipped_reason": (
                        "first step — VLM critic skipped to prevent infinite replanning loop"
                        if is_first_step else
                        "symbolic critic already rejected"
                    ),
                }
                if result["vlm_result"] is None else
                {
                    "ran":         True,
                    "prompt":      vlm_prompt,
                    "valid":       result["vlm_result"]["valid"],
                    "reason":      result["vlm_result"]["reason"],
                    "suggestions": result["vlm_result"]["suggestions"],
                }
            ),
            "final_decision": {
                "valid":    result["valid"],
                "feedback": result["feedback"],
            },
        }
        self._episode_critic_records.append(record)

    # ------------------------------------------------------------------
    # Save log for the current episode
    # ------------------------------------------------------------------
    def save_episode_critic_log(self, instruction: str = '',
                                episode_idx: int = None):
        """
        Write a complete tree-structured JSON log for the current episode to:
            {log_path}/critic_logs/episode_{N}.json

        Top-level document structure:
          {
            "model_name"          : str,
            "instruction"         : str,
            "total_evaluations"   : int,
            "total_rejections"    : int,
            "symbolic_rejections" : int,
            "vlm_rejections"      : int,
            "evaluations": [
              {
                "env_step", "planner_step", "action_step_in_plan",
                "input": {
                  "image", "action_id", "action_str", "is_first_step",
                  "next_action":    {"action_id", "action_name"},   ← VLM judgment target
                  "followup_steps": [{"step", "action_id", "action_name"}, ...]  ← context only
                },
                "symbolic_critic": {"ran", "valid", "reason"},
                "vlm_critic":      {"ran", "prompt", "valid", "reason", "suggestions"}
                                 | {"ran": false, "skipped_reason"},
                "final_decision":  {"valid", "feedback"}
              }, ...
            ]
          }
        """
        if self.log_path is None or not self._episode_critic_records:
            return
        log_dir = os.path.join(self.log_path, 'critic_logs')
        os.makedirs(log_dir, exist_ok=True)
        suffix   = (f'episode_{episode_idx}' if episode_idx is not None
                    else f'episode_{len(self._episode_critic_records)}evals')
        log_file = os.path.join(log_dir, f'{suffix}.json')

        total       = len(self._episode_critic_records)
        rejections  = sum(1 for r in self._episode_critic_records
                          if not r["final_decision"]["valid"])
        sym_rejects = sum(1 for r in self._episode_critic_records
                          if not r["symbolic_critic"]["valid"])
        vlm_rejects = sum(1 for r in self._episode_critic_records
                          if r["vlm_critic"].get("ran") and not r["vlm_critic"]["valid"])

        document = {
            "model_name":          self.vlm.model_name,
            "instruction":         instruction,
            "total_evaluations":   total,
            "total_rejections":    rejections,
            "symbolic_rejections": sym_rejects,
            "vlm_rejections":      vlm_rejects,
            "evaluations":         self._episode_critic_records,
        }
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        logger.info(f"Critic log saved to {log_file} "
                    f"({total} evaluations, {rejections} rejections)")


    def evaluate(self,
                 action_id: int,
                 action_str: str,
                 scene_objects: list,
                 num_actions: int,
                 image_path: str,
                 instruction: str,
                 remaining_actions: list,
                 is_first_step: bool = False,
                 inventory_objects: list = None) -> dict:
        """
        Args:
            action_id        : ID of the action about to be executed.
            action_str       : human-readable action string.
            scene_objects    : env.last_event.metadata['objects'] list.
            num_actions      : total size of the current action space.
            image_path       : path to the latest saved observation image.
            instruction      : task instruction string.
            remaining_actions: list of (action_id, action_name) for all steps
                               not yet executed (including the current one).
            is_first_step    : if True, VLM critic is skipped.
            inventory_objects: env.last_event.metadata['inventoryObjects'] list
                               (objects currently held by the robot).

        Returns:
            dict:
              - valid           (bool)
              - symbolic_result (dict)
              - vlm_result      (dict | None)  — None when skipped
              - feedback        (str)           — empty string when valid=True
        """
        # --- Symbolic check (always runs) ---
        # Log out all the name of objects in the scene for debugging
        obj_names = [obj['objectId'] for obj in scene_objects]
        logger.debug(f"[DualCritic] Evaluating action id {action_id} ('{action_str}') with scene objects: {obj_names}")
        sym_result = self.symbolic.evaluate(
            action_id, action_str, scene_objects, num_actions,
            inventory_objects=inventory_objects,
        )
        if not sym_result["valid"]:
            logger.info(f"[SymbolicCritic] INVALID — {sym_result['reason']}")
            return {
                "valid":           False,
                "symbolic_result": sym_result,
                "vlm_result":      None,
                "vlm_prompt":      None,
                "feedback": (f"[Symbolic Critic] The next action '{action_str}' "
                             f"is not executable: {sym_result['reason']}"),
            }

        # --- VLM check (skip for first step) ---
        if is_first_step:
            logger.debug("[DualCritic] First step — VLM critic skipped to prevent "
                         "infinite replanning loop.")
            return {
                "valid":           True,
                "symbolic_result": sym_result,
                "vlm_result":      None,
                "vlm_prompt":      None,
                "feedback":        "",
            }

        vlm_result = self.vlm.evaluate(image_path, instruction, remaining_actions)
        vlm_prompt = vlm_result.pop("_prompt", None)   # extract before storing result
        if not vlm_result["valid"]:
            next_action_str = (f"action id {remaining_actions[0][0]}, "
                               f"{remaining_actions[0][1]}"
                               if remaining_actions else "unknown")
            feedback = (f"[VLM Critic] The next action '{next_action_str}' "
                        f"is not appropriate: {vlm_result['reason']}")
            if vlm_result["suggestions"]:
                feedback += f" Suggestions: {vlm_result['suggestions']}"
            logger.info(f"[VLMCritic] INVALID — {vlm_result['reason']}")
            return {
                "valid":           False,
                "symbolic_result": sym_result,
                "vlm_result":      vlm_result,
                "vlm_prompt":      vlm_prompt,
                "feedback":        feedback,
            }

        logger.debug(f"[DualCritic] VALID — symbolic: {sym_result['reason']} | "
                     f"vlm: {vlm_result['reason']}")
        return {
            "valid":           True,
            "symbolic_result": sym_result,
            "vlm_result":      vlm_result,
            "vlm_prompt":      vlm_prompt,
            "feedback":        "",
        }
