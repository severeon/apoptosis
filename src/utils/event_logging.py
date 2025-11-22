import json
import time

class ExperimentLogger:
    """
    A structured logger used by train.py to support:
    - metrics per step
    - apoptosis / senescence events
    - final summary
    - stdout JSON emission for sweep_sqlite.py
    """

    def __init__(self):
        self.start_time = time.time()
        self.metrics = []       # list of {step, loss, mean_fitness, ...}
        self.events = []        # list of {type, layer, neuron, step}
        self.run_params = {}    # saved once at start

    # ------------------------------------------------------------------ #
    # Run metadata
    # ------------------------------------------------------------------ #
    def set_params(self, params_dict):
        """Store hyperparams for final JSON."""
        self.run_params = params_dict

    # ------------------------------------------------------------------ #
    # Per-step metric logging
    # ------------------------------------------------------------------ #
    def log_step(self, step, loss, layer_stats):
        [self.metrics.append({
            "step": int(step),
            "loss": float(loss),
            "layer": str(layer_name),
            **stats
        }) for layer_name, stats in layer_stats.items()]

    # ------------------------------------------------------------------ #
    # Event logging (apoptosis / senescence)
    # ------------------------------------------------------------------ #
    def log_event(self, event_type, layer, neuron_index, step):
        self.events.append({
            "event_type": event_type,
            "layer": layer,
            "neuron_index": int(neuron_index),
            "step": int(step)
        })

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    def finalize(self, final_loss):
        duration = time.time() - self.start_time

        summary = {
            "params": self.run_params,
            "duration": duration,
            "final_loss": float(final_loss),
            "total_apoptosis_events": sum(1 for e in self.events if e["event_type"] == "apoptosis"),
            "total_senescence_events": sum(1 for e in self.events if e["event_type"] == "senescence"),
            "metrics": self.metrics,
            "events": self.events,
        }

        # Print EXACTLY ONE JSON OBJECT to stdout
        print(json.dumps(summary))
        return summary
