"""
Navigation Critic Module for EmbodiedBench (eb_navigation).

Consists of:
  - NavigationSymbolicCritic : lightweight, model-free check that verifies the
                               action ID is within bounds and rejects trivially
                               counter-productive moves (e.g. rotating when the
                               target is already in view is handled by the VLM
                               critic, so the symbolic critic only does the
                               cheap, deterministic checks).
  - NavigationDualCritic     : subclass of DualCritic that wires the above
                               symbolic critic together with the shared VLMCritic
                               using the eb_navigation prompt / examples, and
                               forwards `recent_action_ids` to the symbolic critic
                               for rotation-loop detection.
                               Inherits task-level VLM disable and consecutive-
                               rejection tracking from DualCritic unchanged.

Usage
-----
    from embodiedbench.planner.nav_critic import NavigationSymbolicCritic, NavigationDualCritic
    from embodiedbench.planner.critic import VLMCritic

    sym  = NavigationSymbolicCritic()
    vlm  = VLMCritic(model=planner.model, model_name=model_name, env="eb_navigation",
                     language_only=False, n_shot=critic_n_shot)
    dual = NavigationDualCritic(sym, vlm, log_path=env.log_path)
"""

from embodiedbench.main import logger
from embodiedbench.planner.critic import VLMCritic, DualCritic


# ---------------------------------------------------------------------------
# Navigation action constants
# ---------------------------------------------------------------------------

# Action IDs that are pure rotation (never bring the robot closer)
_ROTATION_IDS = {4, 5}

# Human-readable names for all 8 nav actions
NAV_ACTION_NAMES = {
    0: "Move forward by 0.25",
    1: "Move backward by 0.25",
    2: "Move rightward by 0.25",
    3: "Move leftward by 0.25",
    4: "Rotate to the right by 90 degrees",
    5: "Rotate to the left by 90 degrees",
    6: "Tilt the camera upward by 30 degrees",
    7: "Tilt the camera downward by 30 degrees",
}


# ---------------------------------------------------------------------------
# NavigationSymbolicCritic
# ---------------------------------------------------------------------------

class NavigationSymbolicCritic:
    """
    Model-free precondition checker for eb_navigation.

    Checks:
      1. Action ID is within the valid range [0, num_actions).
      2. Consecutive identical rotation: rejects if the same rotation has been
         applied more than _MAX_SAME_ROTATION times in a row (likely stuck in
         a loop).

    Everything else (direction relevance, obstacle avoidance, rotation overuse
    when target is visible) is delegated to the VLMCritic which has access to
    the observation image.

    scene_objects and inventory_objects are accepted for API compatibility but
    are not used (navigation has no object manipulation).
    """

    _MAX_SAME_ROTATION = 4  # reject if the same rotation appears > this many times in a row

    def evaluate(
        self,
        action_id: int,
        action_str: str,
        scene_objects: list,
        num_actions: int,
        inventory_objects: list = None,
        recent_action_ids: list = None,
    ) -> dict:
        """
        Args:
            action_id         : proposed action ID.
            action_str        : human-readable action string.
            scene_objects     : ignored (no object metadata in navigation).
            num_actions       : size of the action space.
            inventory_objects : ignored (robot carries nothing in navigation).
            recent_action_ids : optional list of the last N executed action IDs
                                (oldest first), used for rotation-loop detection.

        Returns:
            dict with keys:
              - valid  (bool)
              - reason (str)
        """
        # 1. Range check
        if action_id < 0 or action_id >= num_actions:
            return {
                "valid": False,
                "reason": (
                    f"Action id {action_id} is out of the valid range "
                    f"(0 ~ {num_actions - 1})."
                ),
            }

        # 2. Rotation-loop detection
        if action_id in _ROTATION_IDS and recent_action_ids:
            tail = recent_action_ids[-self._MAX_SAME_ROTATION:]
            if len(tail) == self._MAX_SAME_ROTATION and all(a == action_id for a in tail):
                return {
                    "valid": False,
                    "reason": (
                        f"The same rotation (action id {action_id}: "
                        f"'{NAV_ACTION_NAMES.get(action_id, action_str)}') "
                        f"has been executed {self._MAX_SAME_ROTATION} times in a row. "
                        "The robot appears stuck in a rotation loop."
                    ),
                }

        return {"valid": True, "reason": "Action is within bounds and passes basic preconditions."}


# ---------------------------------------------------------------------------
# NavigationDualCritic
# ---------------------------------------------------------------------------

class NavigationDualCritic(DualCritic):
    """
    DualCritic specialised for eb_navigation.

    The only difference from the base DualCritic is that `evaluate()` accepts
    an additional `recent_action_ids` kwarg and forwards it to the symbolic
    critic for rotation-loop detection.  All other logic — task-level VLM
    disable, consecutive-rejection tracking, episode logging, and memory
    forwarding — is inherited unchanged from DualCritic.
    """

    def __init__(self, symbolic_critic: NavigationSymbolicCritic,
                 vlm_critic: VLMCritic, log_path: str = None):
        super().__init__(symbolic_critic, vlm_critic, log_path=log_path)

    def evaluate(
        self,
        action_id: int,
        action_str: str,
        scene_objects: list,
        num_actions: int,
        image_path: str,
        instruction: str,
        full_plan: list,
        current_index: int,
        is_first_step: bool = False,
        inventory_objects: list = None,
        info: dict = None,
        recent_action_ids: list = None,
    ) -> dict:
        """
        Evaluate one navigation action.

        Args:
            action_id         : ID of the action about to be executed.
            action_str        : human-readable action string.
            scene_objects     : unused (pass [] for navigation).
            num_actions       : total size of the action space.
            image_path        : path to the current observation image.
            instruction       : task instruction string.
            full_plan         : complete list of (action_id, action_name) for ALL steps in
                                the current plan batch (the VLM planner's full output).
            current_index     : index of the action being evaluated within full_plan.
                                full_plan[current_index] is the judgment target;
                                full_plan[:current_index] are already-executed steps;
                                full_plan[current_index+1:] are future steps (context).
            is_first_step     : if True, VLM critic is skipped.
            inventory_objects : unused (pass [] for navigation).
            info              : optional env info dict forwarded to VLMCritic.
            recent_action_ids : list of recently executed action ids (oldest first),
                                used by the symbolic critic for rotation-loop detection.

        Returns:
            dict with keys: valid, symbolic_result, vlm_result, vlm_skipped_reason,
                            vlm_prompt, feedback
        """
        # --- Symbolic check (with navigation-specific recent_action_ids) ---
        sym_result = self.symbolic.evaluate(
            action_id=action_id,
            action_str=action_str,
            scene_objects=scene_objects,
            num_actions=num_actions,
            inventory_objects=inventory_objects or [],
            recent_action_ids=recent_action_ids or [],
        )
        if not sym_result["valid"]:
            logger.info(f"[NavigationSymbolicCritic] INVALID — {sym_result['reason']}")
            result = {
                "valid":              False,
                "symbolic_result":    sym_result,
                "vlm_result":         None,
                "vlm_skipped_reason": "symbolic critic rejected",
                "vlm_prompt":         None,
                "feedback": (
                    f"[Symbolic Critic] The navigation action '{action_str}' "
                    f"is not executable: {sym_result['reason']}"
                ),
            }
            self._record_evaluation(
                env_step=info.get("env_step", 0) if info else 0,
                planner_step=0,
                action_step_in_plan=current_index,
                action_id=action_id,
                action_str=action_str,
                image_path=image_path,
                full_plan=full_plan,
                current_index=current_index,
                is_first_step=is_first_step,
                result=result,
                vlm_prompt=None,
                inventory_objects=[],
            )
            return result

        # --- VLM check (skip for first step) ---
        if is_first_step:
            logger.debug(
                "[NavigationDualCritic] First step — VLM critic skipped to prevent "
                "infinite replanning loop."
            )
            result = {
                "valid":              True,
                "symbolic_result":    sym_result,
                "vlm_result":         None,
                "vlm_skipped_reason": "first step — VLM critic skipped to prevent infinite replanning loop",
                "vlm_prompt":         None,
                "feedback":           "",
            }
            self._record_evaluation(
                env_step=info.get("env_step", 0) if info else 0,
                planner_step=0,
                action_step_in_plan=current_index,
                action_id=action_id,
                action_str=action_str,
                image_path=image_path,
                full_plan=full_plan,
                current_index=current_index,
                is_first_step=is_first_step,
                result=result,
                vlm_prompt=None,
                inventory_objects=[],
            )
            return result

        # --- Permanently skip VLM if task-level disable was triggered ---
        if self._vlm_task_disabled:
            logger.debug(
                "[NavigationDualCritic] VLM critic is permanently disabled for this task "
                "— auto-approving."
            )
            result = {
                "valid":              True,
                "symbolic_result":    sym_result,
                "vlm_result":         None,
                "vlm_skipped_reason": "VLM critic permanently disabled for this task",
                "vlm_prompt":         None,
                "feedback":           "",
            }
            self._record_evaluation(
                env_step=info.get("env_step", 0) if info else 0,
                planner_step=0,
                action_step_in_plan=current_index,
                action_id=action_id,
                action_str=action_str,
                image_path=image_path,
                full_plan=full_plan,
                current_index=current_index,
                is_first_step=is_first_step,
                result=result,
                vlm_prompt=None,
                inventory_objects=[],
            )
            return result

        # --- VLM check ---
        vlm_result = self.vlm.evaluate(
            image_path, instruction, full_plan, current_index, info=info
        )
        vlm_prompt = vlm_result.pop("_prompt", None)

        if not vlm_result["valid"]:
            reject_count = self._vlm_consecutive_rejections.get(action_str, 0) + 1
            self._vlm_consecutive_rejections[action_str] = reject_count

            if reject_count >= self.VLM_REJECTION_THRESHOLD:
                logger.warning(
                    f"[NavigationVLMCritic] Action '{action_str}' has been rejected "
                    f"{reject_count} times consecutively — permanently disabling "
                    f"VLM critic for the rest of this task."
                )
                self._vlm_task_disabled = True
                self._vlm_consecutive_rejections.clear()
                result = {
                    "valid":              True,
                    "symbolic_result":    sym_result,
                    "vlm_result":         None,
                    "vlm_skipped_reason": f"VLM critic disabled after {reject_count} consecutive rejections",
                    "vlm_prompt":         vlm_prompt,
                    "feedback":           "",
                }
            else:
                next_action_str = (
                    f"action id {full_plan[current_index][0]}, {full_plan[current_index][1]}"
                    if full_plan and current_index < len(full_plan) else "unknown"
                )
                feedback = (
                    f"[VLM Critic] The navigation action '{next_action_str}' "
                    f"is not appropriate: {vlm_result['reason']}"
                )
                if vlm_result.get("suggestions"):
                    feedback += f" Suggestions: {vlm_result['suggestions']}"
                logger.info(f"[NavigationVLMCritic] INVALID (consecutive rejection #{reject_count}) — {vlm_result['reason']}")
                result = {
                    "valid":              False,
                    "symbolic_result":    sym_result,
                    "vlm_result":         vlm_result,
                    "vlm_skipped_reason": None,
                    "vlm_prompt":         vlm_prompt,
                    "feedback":           feedback,
                }
        else:
            # VLM approved — reset consecutive rejection counter for this action
            self._vlm_consecutive_rejections.pop(action_str, None)
            logger.debug(
                f"[NavigationDualCritic] VALID — symbolic: {sym_result['reason']} | "
                f"vlm: {vlm_result['reason']}"
            )
            result = {
                "valid":              True,
                "symbolic_result":    sym_result,
                "vlm_result":         vlm_result,
                "vlm_skipped_reason": None,
                "vlm_prompt":         vlm_prompt,
                "feedback":           "",
            }

        self._record_evaluation(
            env_step=info.get("env_step", 0) if info else 0,
            planner_step=0,
            action_step_in_plan=current_index,
            action_id=action_id,
            action_str=action_str,
            image_path=image_path,
            full_plan=full_plan,
            current_index=current_index,
            is_first_step=is_first_step,
            result=result,
            vlm_prompt=vlm_prompt,
            inventory_objects=[],
        )
        return result
