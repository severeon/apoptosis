from typing import Dict, List

import numpy as np
import torch
from src.utils.event_logging import ExperimentLogger
from src.utils.histogram import quantize_histogram
from src.neuron_metadata import NeuronMetadata
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class NeuronApoptosisManager:
    """
    Manages neuron-level apoptosis and neurogenesis with a senescence daemon.
    Vectorized and device-safe. Includes stabilization heuristics (Option B).
    """

    def __init__(self,
                 model,
                 target_layers: List[str],
                 prune_rate: float = 0.1,
                 apoptosis_interval: int = 500,
                 fitness_metric: str = 'grad_activation',
                 regrowth_strategy: str = 'mutation',
                 mutation_strength: float = 0.3,
                 writer: SummaryWriter = None, # type: ignore
                 fitness_alpha: float = 1.0,
                 fitness_beta: float = 1.0,
                 fitness_gamma: float = 2.0,
                 activation_ema_decay: float = 0.9,
                 logger: ExperimentLogger = None): # type: ignore
        self.model = model
        self.target_layers = target_layers
        self.prune_rate = prune_rate
        self.apoptosis_interval = apoptosis_interval
        self.fitness_metric = fitness_metric
        self.regrowth_strategy = regrowth_strategy
        self.mutation_strength = mutation_strength
        self.writer = writer
        self.logger = logger

        # Fitness function parameters for 3-term model
        self.fitness_alpha = fitness_alpha        # plasticity weight (grad_norm)
        self.fitness_beta = fitness_beta          # usefulness weight (activation_variance)
        self.fitness_gamma = fitness_gamma        # stagnation penalty weight (cosine similarity)
        self.activation_ema_decay = activation_ema_decay  # EMA decay for activation history

        self.step_count = 0
        self.apoptosis_events = []

        # Track neuron metadata and state
        self.neuron_metadata: Dict[str, List[NeuronMetadata]] = {}
        self.neuron_state: Dict[str, Dict] = {}  # vectorized state on CPU

        # Activation hook storage
        # self.activations keeps the CURRENT forward activation on the SAME DEVICE AS MODEL
        self.activations = {}
        # self.recent_outputs keeps history on CPU for apoptosis/regrowth
        self.recent_outputs = {}

        # args holder (set externally by train())
        self.args = None

        # Initialize metadata and hooks
        self._initialize_metadata()
        self._register_hooks()

    # ----------------------------
    # Initialization & hooks
    # ----------------------------
    def _initialize_metadata(self):
        """Initialize metadata and neuron state for all neurons in target layers."""
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            if isinstance(layer, nn.Linear):
                num_neurons = layer.out_features
            else:
                raise ValueError(f"Layer {layer_name} is not a Linear layer")

            self.neuron_metadata[layer_name] = [
                NeuronMetadata(age=0, birth_step=0)
                for _ in range(num_neurons)
            ]

            # vectorized per-neuron bookkeeping (CPU numpy arrays)
            # also add EMA history buffer for slope computation
            self.neuron_state[layer_name] = {
                'phase': np.zeros(num_neurons, dtype=np.int8),         # 0 normal,1 at-risk,2 pre-apoptosis,3 dead,4 young/regrown
                'phase_step': np.zeros(num_neurons, dtype=np.int32),
                'age': np.zeros(num_neurons, dtype=np.int32),
                'low_streak': np.zeros(num_neurons, dtype=np.int32),
                'fitness_ema': np.zeros(num_neurons, dtype=np.float32),
                'ema_history': [],  # list of numpy arrays (keep limited)
                'scale': np.ones(num_neurons, dtype=np.float32),       # multiplicative per-neuron scale (CPU copy)
                'activation_ema': None  # EMA of activations for stagnation detection [H]
            }

    def _register_hooks(self):
        """Register forward hooks to capture activations (GPU) and recent outputs (CPU)."""
        def hook_fn(name):
            def hook(module, input, output):
                # Keep GPU copy for fitness (same device as model)
                self.activations[name] = output.detach()

                # Keep CPU copy for apoptosis/regrowth history (avoid MPS fallback during heavy ops)
                cpu_copy = output.detach().cpu()
                if name not in self.recent_outputs:
                    self.recent_outputs[name] = []
                self.recent_outputs[name].append(cpu_copy)

                # Trim history to configured buffer size when available
                buf = getattr(self, 'output_buffer_size', 10)
                if len(self.recent_outputs[name]) > buf:
                    self.recent_outputs[name].pop(0)
            return hook

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            layer.register_forward_hook(hook_fn(layer_name))

    def _get_layer(self, layer_name: str):
        """Get layer by name (supports nested attributes)."""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer

    # ----------------------------
    # TensorBoard logging helper
    # ----------------------------
    def log_tensorboard(self, writer, global_step):
        # Log per-layer fitness stats and senescence stats
        for layer_name in self.target_layers:
            # activations (GPU local)
            if layer_name in self.activations:
                act = self.activations[layer_name]
                # per-neuron mean activation (device)
                act_norm = torch.abs(act).mean(dim=(0, 1)).cpu()
                writer.add_histogram(f'fitness/{layer_name}_activation', act_norm, global_step)

            # fitness summary (compute on-device -> move CPU)
            try:
                fitness = self.compute_fitness(layer_name).cpu()
                                
                # if global_step % 50 == 0: print("FITNESS:", fitness[:50])
                
                writer.add_histogram(f'fitness/{layer_name}', fitness, global_step)
                writer.add_scalar(f'fitness/{layer_name}_mean', float(fitness.mean().item()), global_step)
                writer.add_scalar(f'fitness/{layer_name}_std', float(fitness.std().item()), global_step)
                writer.add_histogram(f'fitness/{layer_name}', fitness.norm().detach().cpu(), global_step)
            except Exception:
                pass

            # Age & phase stats (from CPU state)
            state = self.neuron_state[layer_name]
            ages = state['age']
            writer.add_scalar(f'age/{layer_name}_mean', float(np.mean(ages)), global_step)
            writer.add_histogram(f'age/{layer_name}_hist', ages, global_step)

            phases = state['phase']
            writer.add_histogram(f'senescence/{layer_name}_phase', phases, global_step)
            writer.add_scalar(f'senescence/{layer_name}_low_streak_mean', float(np.mean(state['low_streak'])), global_step)

        # Total apoptosis events
        writer.add_scalar('apoptosis/total_events', len(self.apoptosis_events), global_step)

    # ----------------------------
    # Fitness computation
    # ----------------------------
    def compute_fitness(self, layer_name):
        """
        3-term neuron fitness metric:
        fitness = α * grad_norm + β * activation_variance - γ * cosine_similarity(activation_vector, EMA)

        Where:
        - grad_norm          → plasticity (learning intensity)
        - activation variance → usefulness (richness of representation)
        - cosine similarity   → stagnation (penalize neurons whose representation stops evolving)
        """

        layer = self._get_layer(layer_name)
        device = layer.weight.device

        # If no gradient yet → neutral fitness
        if layer.weight.grad is None:
            return torch.zeros(layer.weight.size(0), device=device)

        # ------------------------------------------------------
        # 1. Plasticity: gradient magnitude per neuron
        # ------------------------------------------------------
        grad = layer.weight.grad        # [H, D_in]
        grad_norm = grad.norm(dim=1)    # [H]

        # ------------------------------------------------------
        # 2. Usefulness: activation variance per neuron
        # ------------------------------------------------------
        acts = self.activations.get(layer_name, None)
        if acts is None:
            return torch.ones(layer.weight.size(0), device=device) * 1e-3

        # acts: [B, T, H]
        act_var = acts.var(dim=(0, 1))      # [H]

        # For cosine similarity and EMA we use the layer-wide mean activation vector
        # Shape: [H]
        act_mean_vec = acts.mean(dim=(0, 1))

        # ------------------------------------------------------
        # 3. Stagnation penalty: cosine similarity(EMA, current)
        # ------------------------------------------------------
        state = self.neuron_state[layer_name]

        ema = state['activation_ema']
        if ema is None:
            # First-time initialization (keep on device!)
            ema = act_mean_vec.detach().clone()
            state['activation_ema'] = ema
            cos_sim = torch.zeros_like(grad_norm)
        else:
            # True cosine similarity between vectors (layer-wide)
            # Each neuron gets same cosine penalty because stagnation is
            # a layer-level property of representational drift
            curr = act_mean_vec
            prev = ema

            cos_sim_layer = torch.nn.functional.cosine_similarity(
                curr.unsqueeze(0), prev.unsqueeze(0), dim=1
            )  # shape [1]
            cos_sim_layer = cos_sim_layer.item()

            cos_sim = torch.full_like(grad_norm, cos_sim_layer)

            # Update EMA in-place
            ema.mul_(self.activation_ema_decay).add_(
                curr * (1.0 - self.activation_ema_decay)
            )
            state['activation_ema'] = ema

        # ------------------------------------------------------
        # 4. Normalize each term for scale balance
        # ------------------------------------------------------
        def z(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        grad_z = z(grad_norm)
        var_z = z(torch.log1p(act_var))   # log compress
        cos_z = z(cos_sim)

        # ------------------------------------------------------
        # 5. Weighted combination
        # ------------------------------------------------------
        fitness = (
            self.fitness_alpha * grad_z +
            self.fitness_beta  * var_z  -
            self.fitness_gamma * cos_z
        )
        
        # Cross-layer normalization
        fitness_norm = torch.nan_to_num(z(fitness), nan=0.0, posinf=5.0, neginf=-5.0)
        return fitness_norm

    # ----------------------------
    # Step and triggering
    # ----------------------------
    def step(self, loss) -> bool:
        """
        Update ages and check for apoptosis.
        If senescence enabled (args.enable_senescence), run continuous lifecycle daemon.
        Otherwise fallback to periodic apoptosis by interval.
        """
        self.step_count += 1

        # Update ages (metadata)
        for layer_name in self.target_layers:
            for meta in self.neuron_metadata[layer_name]:
                meta.age += 1
            # note: vectorized age increment handled in daemon (only when low)

        self._senescence_daemon(loss)
        return False  # micro-apoptosis handled inside daemon

    # ----------------------------
    # Regrowth: fast vectorized path
    # ----------------------------
    def _regrow_fast(self, layer, dying_indices, healthy_indices, fitness_cpu, layer_name):
        """
        Fully vectorized fast regrowth using cosine similarity.
        Inputs:
          - layer: nn.Linear on device
          - dying_indices: list or np array of ints (CPU)
          - healthy_indices: list or np array of ints (CPU)
          - fitness_cpu: CPU tensor or array used only if needed (not required)
          - layer_name: string key
        """

        # If no recent history, fallback to simple mutation/cloning
        if layer_name not in self.recent_outputs or len(self.recent_outputs[layer_name]) == 0:
            # fallback: simple mutation init for dying neurons
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
                    if layer.bias is not None:
                        layer.bias[neuron_idx] = 0
            return

        # recent outputs: list of [B, T, H] CPU tensors (possibly different B). We will flatten and trim to equal length.
        recent_list = self.recent_outputs[layer_name]

        # Flatten each snapshot to shape [BT, H] and take min length to trim
        flat_list = [r.reshape(-1, r.size(-1)) for r in recent_list]
        min_len = min(f.size(0) for f in flat_list)
        if min_len <= 0:
            # fallback
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
            return

        trimmed = [f[:min_len] for f in flat_list]  # each is [min_len, H]

        # stacked: (S, P, H)
        stacked = torch.stack(trimmed)
        S, P, H = stacked.shape
        big = stacked.reshape(S * P, H).T.contiguous()  # (H, S*P)

        # Select rows for dying and healthy neurons
        dying_idx = np.array(dying_indices, dtype=np.int64)
        healthy_idx = np.array(healthy_indices, dtype=np.int64)

        # build matrices: (D, F) and (H', F)
        dying_mat = big[dying_idx].detach().clone()
        healthy_mat = big[healthy_idx].detach().clone()

        # Normalize rows
        dying_norm = dying_mat / (dying_mat.norm(dim=1, keepdim=True) + 1e-8)
        healthy_norm = healthy_mat / (healthy_mat.norm(dim=1, keepdim=True) + 1e-8)

        # Similarity matrix (D, H') on CPU
        sim = dying_norm @ healthy_norm.T  # CPU

        # Best match per dying neuron (indices into healthy_idx)
        best_ids = torch.argmax(sim, dim=1).cpu().numpy().astype(np.int64)  # length D
        best_parents = healthy_idx[best_ids]  # actual parent neuron indices

        # Convert to tensors on model device for weight ops
        dev = layer.weight.device
        parent_indices = torch.tensor(best_parents, dtype=torch.long, device=dev)
        dying_indices_t = torch.tensor(dying_idx, dtype=torch.long, device=dev)

        with torch.no_grad():
            # Batch copy weights
            layer.weight[dying_indices_t, :] = layer.weight[parent_indices, :].clone()

            # Gentle noise
            noise = torch.randn_like(layer.weight[dying_indices_t]) * (self.mutation_strength * 0.5)
            layer.weight[dying_indices_t] += noise

            # Bias handling (1D)
            if layer.bias is not None:
                parent_bias = layer.bias[parent_indices].clone()
                noise_b = torch.randn(len(dying_idx), device=dev) * (self.mutation_strength * 0.5)
                layer.bias[dying_indices_t] = parent_bias + noise_b

        # Log average similarity
        if self.writer:
            avg_sim = float(sim.mean().item())
            self.writer.add_scalar(f'functional_similarity/{layer_name}', avg_sim, self.step_count)

    # Generic regrow dispatcher
    def _regrow_neurons(self, layer, dying_indices, healthy_indices, fitness):
        """Regrow neurons using specified strategy (fast vectorized path enabled)."""
        layer_name = None
        for name in self.target_layers:
            if self._get_layer(name) is layer:
                layer_name = name
                break

        if layer_name is None:
            # fallback to simple mutation
            with torch.no_grad():
                for neuron_idx in dying_indices:
                    nn.init.xavier_uniform_(layer.weight[neuron_idx:neuron_idx+1, :])
            return

        # We always use fast path here (vectorized)
        try:
            # ensure fitness on CPU optional
            fitness_cpu = fitness.detach().cpu() if isinstance(fitness, torch.Tensor) else fitness
        except Exception:
            fitness_cpu = None

        self._regrow_fast(layer, dying_indices, healthy_indices, fitness_cpu, layer_name)

    # ----------------------------
    # Senescence daemon & helpers (Option B)
    # ----------------------------
    def _apply_neuron_scales(self, layer, layer_name, new_scales_np):
        """
        Apply multiplicative scales to layer weights & bias for each neuron.
        new_scales_np is a numpy array of shape (out_features,) on CPU.
        This function computes the ratio to previous scale and performs an in-place multiply.
        """
        with torch.no_grad():
            dev = layer.weight.device
            prev_scales = torch.tensor(self.neuron_state[layer_name]['scale'], dtype=torch.float32, device=dev)
            new_scales = torch.tensor(new_scales_np, dtype=torch.float32, device=dev)

            # Avoid division by zero
            ratio = new_scales / (prev_scales + 1e-12)  # shape (out_features,)
            ratio_w = ratio.unsqueeze(1)

            layer.weight.mul_(ratio_w)  # scale rows
            if layer.bias is not None:
                layer.bias.mul_(ratio)

            # Save new scales (CPU copy)
            self.neuron_state[layer_name]['scale'] = new_scales_np.copy()

    def _get_initial_neuron_state(self, num_neurons):
        return {
            'phase': np.zeros(num_neurons, dtype=np.int8),         # 0 normal,1 at-risk,2 pre-apoptosis,3 dead,4 young/regrown
            'phase_step': np.zeros(num_neurons, dtype=np.int32),
            'age': np.zeros(num_neurons, dtype=np.int32),
            'low_streak': np.zeros(num_neurons, dtype=np.int32),
            'fitness_ema': np.zeros(num_neurons, dtype=np.float32),
            'ema_history': [],  # list of numpy arrays (keep limited)
            'scale': np.ones(num_neurons, dtype=np.float32),       # multiplicative per-neuron scale (CPU copy)
            'activation_ema': None  # EMA of activations for stagnation detection [H]
        }

    def _set_dead_neurons(self, layer, layer_name, dead_indices_np, num_neurons):
        """Zero weights & bias rows for dead neurons (vectorized)."""
        with torch.no_grad():
            if len(dead_indices_np) == 0:
                return
            dead_idx = torch.tensor(dead_indices_np, dtype=torch.long, device=layer.weight.device)
            layer.weight[dead_idx, :].zero_()
            if layer.bias is not None:
                layer.bias[dead_idx].zero_()
            # update scale bookkeeping
            for idx in dead_indices_np:
                self.neuron_state[layer_name]['scale'][idx] = 0.0
                self.neuron_state[layer_name]['activation_ema'] = 0.0


    def _senescence_daemon(self, loss):
        """
        Monitor fitness and move neurons through lifecycle phases using Option B stabilizers.

        Phase definitions (milder):
          0 - normal (scale 1.0)
          1 - at-risk (scale 0.95)
          2 - pre-apoptosis (scale 0.90)
          3 - dead (scale 0.0)
          4 - regrown / young (scale 1.05)
        """

        # config from args with safe defaults
        low_pct = getattr(self.args, 'senescence_low_pct', 0.10)
        patience = getattr(self.args, 'senescence_patience', 20)
        age_factor = getattr(self.args, 'senescence_age_factor', 0.01)
        phase_durations_str = getattr(self.args, 'phase_durations', '10,5,1,5')
        phase_durations = [int(x) for x in phase_durations_str.split(',')]
        alpha = 0.9  # EMA smoothing
        slope_window = getattr(self.args, 'slope_window', 5)
        slope_threshold = getattr(self.args, 'slope_threshold', -0.1)  # require meaningful negative slope
        warmup = getattr(self.args, 'lifecycle_warmup_steps', 200)
        max_escalations = getattr(self.args, 'max_escalations_per_step', 5)
        max_kills = getattr(self.args, 'max_kills_per_layer', 3)
        min_to_escalate = getattr(self.args, 'min_to_escalate', 1)
        
        block_fitnesses = dict()
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            try:
                block_fitnesses[layer_name] = self.compute_fitness(layer_name)
            except Exception:
                block_fitnesses[layer_name] = None
                continue
            
        self.logger.log_step(
            step=self.step_count,
            loss=loss.item(),
            layer_stats={
                layer_name: {
                    "mean": float(fitness.mean()),
                    "p05": float(torch.quantile(fitness, 0.05)),
                    "p50": float(torch.quantile(fitness, 0.5)),
                    "p95": float(torch.quantile(fitness, 0.95)),
                    "var": float(torch.var(fitness)),
                    "hist": quantize_histogram(fitness, bins=16),
                }
                for layer_name, fitness in block_fitnesses.items() if fitness is not None
            }
        )
    
        # Warmup safety: don't do anything too early
        if self.step_count < warmup:
            return

        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            fitness = block_fitnesses[layer_name]
            
            # compute fitness on-device -> move to CPU for daemon
            try:
                fitness_cpu = fitness.detach().cpu().numpy()  # raw scale (e.g. ~1e-3)
            except Exception:
                continue

            state = self.neuron_state[layer_name]
            num_neurons = len(fitness_cpu)

            # --- Normalize fitness: z-score (robust to tiny absolute scales) ---
            f_mean = float(np.mean(fitness_cpu))
            f_std = float(np.std(fitness_cpu)) + 1e-12
            fitness_z = (fitness_cpu - f_mean) / f_std

            # update EMA (on raw or z? keep EMA on z for slope stability)
            prev_ema = state['fitness_ema']
            # store EMA of z-scores so slope is in std units
            new_ema = alpha * prev_ema + (1.0 - alpha) * fitness_z
            # keep short history for slope calculation (store numpy arrays)
            hist = state.get('ema_history', [])
            hist.append(new_ema.copy())
            if len(hist) > slope_window:
                hist.pop(0)
            state['ema_history'] = hist
            # store fitness_ema as z-ema (CPU)
            state['fitness_ema'] = new_ema

            # --- thresholding on z-scores (now meaningful) ---
            # compute percentile on z-scores (e.g. bottom 10%)
            thresh = float(np.percentile(new_ema, 100.0 * low_pct))

            # update low_streak vectorized (z-score below threshold)
            low_mask = new_ema < thresh
            state['low_streak'][low_mask] += 1
            state['low_streak'][~low_mask] = np.maximum(state['low_streak'][~low_mask] - 1, 0)

            # Age increases only for low neurons (so age measures chronic underperformance)
            state['age'][low_mask] += 1

            # compute slope from EMA history (in std units)
            slope = np.zeros_like(new_ema)
            if len(hist) >= 2:
                earlier = hist[0]
                slope = new_ema - earlier  # negative -> decreasing fitness

            # risk combines streak and age factor (age is already small integers)
            risk = state['low_streak'] + (state['age'] * age_factor)

            # Ensure caps are at least 1 (avoid accidental zeros)
            max_escalations = max(1, max_escalations)
            max_kills = max(1, max_kills)

            # Debugging metrics: log these scalars for quick inspection
            if self.writer:
                self.writer.add_scalar(f'senescence/{layer_name}_fitness_mean', float(f_mean), self.step_count)
                self.writer.add_scalar(f'senescence/{layer_name}_fitness_std', float(f_std), self.step_count)
                self.writer.add_scalar(f'senescence/{layer_name}_fitness_z_thresh', float(thresh), self.step_count)
                self.writer.add_scalar(f'senescence/{layer_name}_risk_mean', float(np.mean(risk)), self.step_count)
                self.writer.add_scalar(f'senescence/{layer_name}_risk_max', float(np.max(risk)), self.step_count)

            # Candidate escalation: require (low z-score) AND (negative slope beyond threshold) AND (risk > patience)
            # slope_threshold is interpreted in z-score units (e.g., -0.1 = drop of 0.1 stds over window)
            candidates = np.where(
                (new_ema < thresh) &
                (slope <= -abs(slope_threshold)) &
                (risk > patience) &
                (state['phase'] == 0)
            )[0]

            # Cap escalations to avoid mass churn
            if candidates.size > 0:
                # stable ordering by risk (highest risk first)
                order = np.argsort(-risk[candidates])
                candidates_sorted = candidates[order]
                k = min(len(candidates_sorted), max_escalations)
                to_at_risk = candidates_sorted[:k]
            else:
                to_at_risk = np.array([], dtype=np.int32)

            # Informative logging
            if self.writer:
                self.writer.add_scalar(f'senescence/{layer_name}_num_candidates', int(candidates.size), self.step_count)
                self.writer.add_scalar(f'senescence/{layer_name}_num_escalated', int(to_at_risk.size), self.step_count)
                
            [self.logger.log_event(
                event_type="senescence",
                layer=layer_name,
                neuron_index=int(i),
                step=self.step_count
            ) for i in to_at_risk]

            # Ensure at least minimal escalation if configured (rare)
            if to_at_risk.size < min_to_escalate and candidates.size >= min_to_escalate:
                to_at_risk = candidates[:min_to_escalate]

            if to_at_risk.size > 0:
                state['phase'][to_at_risk] = 1
                state['phase_step'][to_at_risk] = 0
                
                self.logger
                if self.writer:
                    for idx in to_at_risk[:10]:
                        self.writer.add_scalar(f'senescence/{layer_name}_escalated', int(idx), self.step_count)

            # Advance phase steps for non-normal neurons
            non_zero = np.where(state['phase'] != 0)[0]
            if non_zero.size > 0:
                state['phase_step'][non_zero] += 1

            # 1 -> 2 when phase_step >= phase_durations[0]
            advance_1_to_2 = np.where((state['phase'] == 1) & (state['phase_step'] >= phase_durations[0]))[0]
            if advance_1_to_2.size > 0:
                state['phase'][advance_1_to_2] = 2
                state['phase_step'][advance_1_to_2] = 0

            # 2 -> 3 when phase_step >= phase_durations[1] (we kill)
            advance_2_to_3 = np.where((state['phase'] == 2) & (state['phase_step'] >= phase_durations[1]))[0]
            if advance_2_to_3.size > 0:
                # limit kills to a small number per layer to keep stability
                kill_k = min(len(advance_2_to_3), max_kills)
                to_kill = advance_2_to_3[:kill_k]
                self._set_dead_neurons(layer, layer_name, to_kill.tolist(), num_neurons)
                state['phase'][to_kill] = 3
                state['phase_step'][to_kill] = 0
                self.apoptosis_events.append((self.step_count, layer_name, int(len(to_kill))))

                if self.writer:
                    self.writer.add_scalar(f'senescence/{layer_name}_killed_count', int(len(to_kill)), self.step_count)
                
                [self.logger.log_event(
                    event_type="apoptosis",
                    layer=layer_name,
                    neuron_index=int(i),
                    step=self.step_count
                ) for i in to_kill]

            # 3 -> 4 (regrowth) when phase_step >= phase_durations[2]
            advance_3_to_4 = np.where((state['phase'] == 3) & (state['phase_step'] >= phase_durations[2]))[0]
            if advance_3_to_4.size > 0:
                # pick healthy indices from phase 0 (normal)
                healthy_idxs = np.where(state['phase'] == 0)[0]
                if healthy_idxs.size == 0:
                    healthy_idxs = np.arange(num_neurons, dtype=np.int32)

                # call regrow using vectorized fast method, limit batch size for safety
                try:
                    self._regrow_neurons(layer, advance_3_to_4.tolist(), healthy_idxs.tolist(), torch.tensor(new_ema))
                except Exception:
                    self._regrow_neurons(layer, advance_3_to_4.tolist(), healthy_idxs.tolist(), torch.tensor(new_ema))

                state['phase'][advance_3_to_4] = 4
                state['phase_step'][advance_3_to_4] = 0

                if self.writer:
                    self.writer.add_scalar(f'senescence/{layer_name}_regrown_count', int(len(advance_3_to_4)), self.step_count)

            # 4 -> 0 when phase_step >= phase_durations[3] (youth finishes)
            advance_4_to_0 = np.where((state['phase'] == 4) & (state['phase_step'] >= phase_durations[3]))[0]
            if advance_4_to_0.size > 0:
                state['phase'][advance_4_to_0] = 0
                state['phase_step'][advance_4_to_0] = 0
                # restore normal scale for these neurons
                new_scales = np.array(state['scale'])
                new_scales[advance_4_to_0] = 1.0
                self._apply_neuron_scales(layer, layer_name, new_scales)

            # Apply per-phase scales to all neurons (vectorized) - milder scales for Option B
            scale_map = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.0, 4: 1.05}
            curr_scales = np.array([scale_map[int(p)] for p in state['phase']], dtype=np.float32)

            if not np.allclose(curr_scales, state['scale']):
                self._apply_neuron_scales(layer, layer_name, curr_scales)
