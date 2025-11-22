#!/usr/bin/env python3
"""
Global Step Scheduler with Subscription Pattern

This module provides a centralized step counter and periodic action scheduler.
Components can subscribe to periodic callbacks, eliminating scattered
'if step % N == 0' patterns throughout the codebase.

Example usage:
    scheduler = StepScheduler()
    scheduler.subscribe(interval=500, callback=run_validation, name='validation')
    scheduler.subscribe(interval=250, callback=log_metrics, name='logging')

    # In training loop:
    for _ in range(num_steps):
        # ... training code ...
        scheduler.step()  # Triggers all scheduled callbacks
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScheduledAction:
    """A single scheduled periodic action."""
    name: str
    interval: int
    callback: Callable
    offset: int = 0  # Start offset (useful for staggered actions)
    enabled: bool = True
    warmup_steps: int = 0  # Don't execute until this many steps have passed
    last_executed: int = -1  # Track last execution for debugging

    def should_execute(self, current_step: int) -> bool:
        """Check if action should execute at current step."""
        if not self.enabled:
            return False
        if current_step < self.warmup_steps:
            return False
        return (current_step - self.offset) % self.interval == 0 and current_step >= self.offset


class StepScheduler:
    """
    Centralized step counter and periodic action scheduler.

    This class serves as the single source of truth for global training steps
    and manages all periodic actions (validation, logging, checkpointing, etc.).
    """

    def __init__(self):
        self._current_step: int = 0
        self._actions: Dict[str, ScheduledAction] = {}
        self._step_listeners: List[Callable[[int], None]] = []

        # Statistics
        self._total_callbacks_executed: int = 0
        self._callbacks_by_name: Dict[str, int] = defaultdict(int)

    @property
    def current_step(self) -> int:
        """Get the current global step count."""
        return self._current_step

    def subscribe(
        self,
        callback: Callable,
        interval: int,
        name: Optional[str] = None,
        offset: int = 0,
        warmup_steps: int = 0,
        enabled: bool = True
    ) -> str:
        """
        Subscribe a callback to be executed periodically.

        Args:
            callback: Function to call (should accept no arguments or current_step)
            interval: Execute every N steps
            name: Unique name for this action (auto-generated if not provided)
            offset: Offset from step 0 (useful for staggered execution)
            warmup_steps: Don't execute until this many steps have passed
            enabled: Whether action is enabled (can be toggled later)

        Returns:
            Name of the registered action (for later reference)

        Example:
            # Execute validation every 500 steps starting from step 0
            scheduler.subscribe(run_validation, interval=500, name='validation')

            # Execute logging every 250 steps, but skip first 100 steps
            scheduler.subscribe(log_metrics, interval=250, warmup_steps=100)

            # Execute checkpoint every 2500 steps, offset by 100 (2600, 5100, etc.)
            scheduler.subscribe(save_checkpoint, interval=2500, offset=100)
        """
        if name is None:
            name = f"action_{len(self._actions)}"

        if name in self._actions:
            logger.warning(f"Action '{name}' already registered, replacing...")

        action = ScheduledAction(
            name=name,
            interval=interval,
            callback=callback,
            offset=offset,
            enabled=enabled,
            warmup_steps=warmup_steps
        )

        self._actions[name] = action
        logger.debug(f"Registered action '{name}': interval={interval}, offset={offset}, warmup={warmup_steps}")

        return name

    def unsubscribe(self, name: str) -> bool:
        """
        Remove a scheduled action.

        Args:
            name: Name of action to remove

        Returns:
            True if action was removed, False if not found
        """
        if name in self._actions:
            del self._actions[name]
            logger.debug(f"Unsubscribed action '{name}'")
            return True
        return False

    def enable_action(self, name: str):
        """Enable a previously disabled action."""
        if name in self._actions:
            self._actions[name].enabled = True
            logger.debug(f"Enabled action '{name}'")

    def disable_action(self, name: str):
        """Disable an action without removing it."""
        if name in self._actions:
            self._actions[name].enabled = False
            logger.debug(f"Disabled action '{name}'")

    def add_step_listener(self, listener: Callable[[int], None]):
        """
        Add a listener that gets called on every step (not periodic).

        Useful for continuous monitoring or actions that need to run every step.
        The listener receives the current step as an argument.

        Example:
            scheduler.add_step_listener(lambda step: print(f"Step {step}"))
        """
        self._step_listeners.append(listener)

    def step(self, **kwargs) -> Dict[str, Any]:
        """
        Advance the step counter by 1 and execute all scheduled actions.

        Args:
            **kwargs: Additional arguments to pass to callbacks (if they accept them)

        Returns:
            Dictionary mapping action names to their return values (if any)

        Example:
            # In training loop:
            for _ in range(num_steps):
                loss = train_step()
                results = scheduler.step(loss=loss)
        """
        self._current_step += 1
        results = {}

        # Execute step listeners (continuous monitoring)
        for listener in self._step_listeners:
            try:
                # Try to call with step argument
                listener(self._current_step)
            except TypeError:
                # Fallback: call without arguments
                listener()

        # Execute scheduled periodic actions
        for name, action in self._actions.items():
            if action.should_execute(self._current_step):
                try:
                    # Try to call with kwargs first (for callbacks that need context)
                    result = self._execute_callback(action.callback, **kwargs)
                    results[name] = result

                    # Update statistics
                    action.last_executed = self._current_step
                    self._total_callbacks_executed += 1
                    self._callbacks_by_name[name] += 1

                    logger.debug(f"Executed '{name}' at step {self._current_step}")

                except Exception as e:
                    logger.error(f"Error executing callback '{name}' at step {self._current_step}: {e}")
                    results[name] = None

        return results

    def _execute_callback(self, callback: Callable, **kwargs) -> Any:
        """
        Execute a callback, trying different argument patterns.

        First tries to call with kwargs, then with current_step, then with no args.
        """
        try:
            # Try with kwargs
            return callback(**kwargs)
        except TypeError:
            try:
                # Try with just current_step
                return callback(self._current_step)
            except TypeError:
                # Try with no arguments
                return callback()

    def reset(self):
        """Reset the scheduler (useful for testing or restarting training)."""
        self._current_step = 0
        for action in self._actions.values():
            action.last_executed = -1
        logger.debug("Scheduler reset to step 0")

    def get_next_execution(self, name: str) -> Optional[int]:
        """
        Get the next step when an action will execute.

        Args:
            name: Name of the action

        Returns:
            Next step number, or None if action not found
        """
        if name not in self._actions:
            return None

        action = self._actions[name]
        if not action.enabled:
            return None

        # Find next execution step
        current = self._current_step
        next_step = current + 1

        while True:
            if action.should_execute(next_step):
                return next_step
            next_step += 1
            # Prevent infinite loop
            if next_step > current + action.interval * 2:
                return None

    def get_schedule_summary(self) -> str:
        """
        Get a human-readable summary of all scheduled actions.

        Returns:
            Formatted string showing all actions and their schedules
        """
        if not self._actions:
            return "No scheduled actions"

        lines = [f"Step Scheduler (current step: {self._current_step})"]
        lines.append("=" * 60)

        for name, action in sorted(self._actions.items()):
            status = "✓" if action.enabled else "✗"
            next_exec = self.get_next_execution(name)
            next_str = f"next: {next_exec}" if next_exec is not None else "disabled"

            lines.append(
                f"{status} {name:20s} | interval: {action.interval:5d} | "
                f"offset: {action.offset:3d} | warmup: {action.warmup_steps:4d} | {next_str}"
            )

        lines.append("=" * 60)
        lines.append(f"Total callbacks executed: {self._total_callbacks_executed}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for all actions."""
        return {
            'current_step': self._current_step,
            'total_callbacks_executed': self._total_callbacks_executed,
            'callbacks_by_name': dict(self._callbacks_by_name),
            'registered_actions': len(self._actions),
            'enabled_actions': sum(1 for a in self._actions.values() if a.enabled)
        }


# Convenience function for simple cases
def create_simple_scheduler(
    validation_interval: int = 500,
    logging_interval: int = 250,
    checkpoint_interval: int = 2500,
    validation_callback: Optional[Callable] = None,
    logging_callback: Optional[Callable] = None,
    checkpoint_callback: Optional[Callable] = None
) -> StepScheduler:
    """
    Create a scheduler with common training callbacks pre-configured.

    This is a convenience function for the typical training setup.
    You can still add more actions after creation.

    Example:
        scheduler = create_simple_scheduler(
            validation_interval=500,
            validation_callback=run_validation
        )
        scheduler.subscribe(my_custom_action, interval=1000)
    """
    scheduler = StepScheduler()

    if validation_callback is not None:
        scheduler.subscribe(
            validation_callback,
            interval=validation_interval,
            name='validation'
        )

    if logging_callback is not None:
        scheduler.subscribe(
            logging_callback,
            interval=logging_interval,
            name='logging'
        )

    if checkpoint_callback is not None:
        scheduler.subscribe(
            checkpoint_callback,
            interval=checkpoint_interval,
            name='checkpoint'
        )

    return scheduler
