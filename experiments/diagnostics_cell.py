"""
DIAGNOSTICS CELL - Paste this output into chat for analysis!

Run this cell after your experiment completes to get a comprehensive summary.
"""

def generate_diagnostics_report(baseline_trainer, apoptotic_trainer, apoptosis_manager):
    """Generate comprehensive diagnostics report."""

    report = []
    report.append("\n" + "="*70)
    report.append("APOPTOSIS EXPERIMENT DIAGNOSTICS")
    report.append("="*70)

    # === SECTION 1: Training Overview ===
    report.append("\n📊 TRAINING OVERVIEW")
    report.append("-" * 70)
    baseline_steps = len(baseline_trainer.metrics_history)
    apoptotic_steps = len(apoptotic_trainer.metrics_history)
    report.append(f"Baseline steps:   {baseline_steps:,}")
    report.append(f"Apoptotic steps:  {apoptotic_steps:,}")
    report.append(f"Device used:      {device}")

    # === SECTION 2: Final Performance ===
    report.append("\n🎯 FINAL PERFORMANCE")
    report.append("-" * 70)

    baseline_final = baseline_trainer.metrics_history[-1]
    apoptotic_final = apoptotic_trainer.metrics_history[-1]

    loss_diff = apoptotic_final.loss - baseline_final.loss
    ppl_diff = apoptotic_final.perplexity - baseline_final.perplexity

    report.append(f"Baseline Loss:      {baseline_final.loss:.4f}")
    report.append(f"Apoptotic Loss:     {apoptotic_final.loss:.4f}")
    report.append(f"Difference:         {loss_diff:+.4f}  {'✓ BETTER' if loss_diff < 0 else '✗ WORSE' if abs(loss_diff) > 0.1 else '≈ SAME'}")
    report.append("")
    report.append(f"Baseline Perplexity:   {baseline_final.perplexity:.2f}")
    report.append(f"Apoptotic Perplexity:  {apoptotic_final.perplexity:.2f}")
    report.append(f"Difference:            {ppl_diff:+.2f}")

    # === SECTION 3: Apoptosis Events ===
    report.append("\n💀 APOPTOSIS EVENTS")
    report.append("-" * 70)

    events = apoptosis_manager.apoptosis_events
    report.append(f"Total events: {len(events)}")

    if len(events) == 0:
        report.append("⚠️  WARNING: NO APOPTOSIS EVENTS! Lifespan too long?")
    elif len(events) < 5:
        report.append("⚠️  WARNING: Very few events. Consider shorter lifespan.")
    else:
        # Count deaths per layer
        from collections import Counter
        layer_deaths = Counter(layer for _, layer in events)

        report.append("\nDeaths per layer:")
        for layer in sorted(layer_deaths.keys()):
            count = layer_deaths[layer]
            bar = "█" * count
            report.append(f"  Layer {layer}: {count:2d} {bar}")

        # Event timing
        if len(events) > 1:
            steps_between = [events[i+1][0] - events[i][0] for i in range(len(events)-1)]
            avg_interval = sum(steps_between) / len(steps_between)
            report.append(f"\nAverage interval: {avg_interval:.0f} steps")

        # Show first 5 and last 5 events
        report.append("\nFirst 5 events:")
        for step, layer in events[:5]:
            report.append(f"  Step {step:5d}: Layer {layer}")

        if len(events) > 10:
            report.append("  ...")
            report.append("Last 5 events:")
            for step, layer in events[-5:]:
                report.append(f"  Step {step:5d}: Layer {layer}")

    # === SECTION 4: Vitality Analysis ===
    report.append("\n🧬 VITALITY PATTERNS")
    report.append("-" * 70)

    # Sample vitality at different points
    sample_indices = [0, len(apoptotic_trainer.metrics_history)//4,
                     len(apoptotic_trainer.metrics_history)//2,
                     3*len(apoptotic_trainer.metrics_history)//4,
                     len(apoptotic_trainer.metrics_history)-1]

    report.append("Vitality snapshots (by layer):")
    report.append("Layer | Start  | 25%    | 50%    | 75%    | End")
    report.append("------|--------|--------|--------|--------|--------")

    for layer_idx in range(6):
        row = f"  {layer_idx}   |"
        for idx in sample_indices:
            vitality = apoptotic_trainer.metrics_history[idx].layer_vitalities[layer_idx]
            row += f" {vitality:.3f}  |"

        # Add zone indicator
        zone = apoptotic_trainer.model.senescence[layer_idx].layer_zone
        zone_symbol = {"birth": "🌱", "death": "💀", "stable": "🔒"}[zone]
        row += f"  {zone_symbol}"
        report.append(row)

    # Check if vitality is actually changing
    report.append("\nVitality range per layer:")
    for layer_idx in range(6):
        vitalities = [m.layer_vitalities[layer_idx] for m in apoptotic_trainer.metrics_history]
        min_v, max_v = min(vitalities), max(vitalities)
        range_v = max_v - min_v

        status = "✓" if range_v > 0.3 else "⚠️" if range_v > 0.1 else "✗"
        report.append(f"  Layer {layer_idx}: [{min_v:.2f}, {max_v:.2f}]  range={range_v:.2f}  {status}")

    # === SECTION 5: Gradient Health ===
    report.append("\n⚡ GRADIENT HEALTH")
    report.append("-" * 70)

    # Check final gradient norms
    final_grads = apoptotic_final.gradient_norms
    report.append("Final gradient norms by layer:")
    for layer_idx, grad_norm in enumerate(final_grads):
        status = "✓" if grad_norm > 0.1 else "⚠️" if grad_norm > 0.01 else "✗ DEAD"
        report.append(f"  Layer {layer_idx}: {grad_norm:.4f}  {status}")

    # Check for gradient collapse
    dead_layers = sum(1 for g in final_grads if g < 0.01)
    if dead_layers > 0:
        report.append(f"\n⚠️  WARNING: {dead_layers} layers with near-zero gradients!")

    # === SECTION 6: Loss Stability ===
    report.append("\n📈 LOSS STABILITY")
    report.append("-" * 70)

    # Find loss spikes at apoptosis events
    max_spike = 0.0
    spike_locations = []

    for step, layer in events:
        # Find loss before and after
        step_indices = [i for i, m in enumerate(apoptotic_trainer.metrics_history)
                       if abs(m.step - step) < 20]

        if len(step_indices) > 10:
            before_idx = step_indices[0]
            after_idx = step_indices[-1]

            loss_before = apoptotic_trainer.metrics_history[before_idx].loss
            loss_after = apoptotic_trainer.metrics_history[after_idx].loss
            spike = loss_after - loss_before

            if abs(spike) > max_spike:
                max_spike = abs(spike)
                spike_locations.append((step, layer, spike))

    report.append(f"Max loss spike: {max_spike:.4f}")

    if max_spike > 1.0:
        report.append("✗ CATASTROPHIC: Loss spikes > 1.0 at apoptosis")
    elif max_spike > 0.5:
        report.append("⚠️  WARNING: Significant loss spikes at apoptosis")
    elif max_spike > 0.2:
        report.append("⚠️  Noticeable loss bumps at apoptosis")
    else:
        report.append("✓ Graceful degradation: minimal loss spikes")

    if spike_locations:
        report.append("\nTop 3 loss spikes:")
        for step, layer, spike in sorted(spike_locations, key=lambda x: abs(x[2]), reverse=True)[:3]:
            report.append(f"  Step {step}: Layer {layer} → {spike:+.4f}")

    # === SECTION 7: Convergence Analysis ===
    report.append("\n📉 CONVERGENCE")
    report.append("-" * 70)

    # Compare last 100 steps
    baseline_last100 = [m.loss for m in baseline_trainer.metrics_history[-100:]]
    apoptotic_last100 = [m.loss for m in apoptotic_trainer.metrics_history[-100:]]

    baseline_variance = sum((l - baseline_final.loss)**2 for l in baseline_last100) / len(baseline_last100)
    apoptotic_variance = sum((l - apoptotic_final.loss)**2 for l in apoptotic_last100) / len(apoptotic_last100)

    report.append(f"Loss variance (last 100 steps):")
    report.append(f"  Baseline:   {baseline_variance:.6f}")
    report.append(f"  Apoptotic:  {apoptotic_variance:.6f}")

    if apoptotic_variance > baseline_variance * 2:
        report.append("⚠️  Apoptotic model less stable (higher variance)")
    else:
        report.append("✓ Comparable stability")

    # === SECTION 8: Parameter Efficiency ===
    report.append("\n⚙️  PARAMETER EFFICIENCY")
    report.append("-" * 70)

    total_params = baseline_final.effective_params
    effective_params = apoptotic_final.effective_params
    efficiency = (baseline_final.loss / apoptotic_final.loss) * (effective_params / total_params)

    report.append(f"Total parameters:     {total_params:,}")
    report.append(f"Effective parameters: {effective_params:,}")
    report.append(f"Efficiency ratio:     {efficiency:.3f}")

    if efficiency > 1.1:
        report.append("✓ MORE EFFICIENT than baseline!")
    elif efficiency > 0.95:
        report.append("≈ Similar efficiency to baseline")
    else:
        report.append("✗ Less efficient than baseline")

    # === SECTION 9: Overall Assessment ===
    report.append("\n🎓 OVERALL ASSESSMENT")
    report.append("="*70)

    success_score = 0
    max_score = 7

    # Criteria
    criteria = []

    # 1. No NaN
    if not (np.isnan(apoptotic_final.loss) or np.isinf(apoptotic_final.loss)):
        success_score += 1
        criteria.append("✓ No NaN/Inf losses")
    else:
        criteria.append("✗ NaN/Inf detected")

    # 2. Apoptosis occurred
    if len(events) >= 5:
        success_score += 1
        criteria.append(f"✓ Apoptosis working ({len(events)} events)")
    else:
        criteria.append(f"✗ Too few apoptosis events ({len(events)})")

    # 3. Graceful degradation
    if max_spike < 0.5:
        success_score += 1
        criteria.append("✓ Graceful degradation")
    else:
        criteria.append("✗ Large loss spikes")

    # 4. No gradient collapse
    if dead_layers == 0:
        success_score += 1
        criteria.append("✓ Healthy gradients")
    else:
        criteria.append(f"✗ {dead_layers} layers with dead gradients")

    # 5. Vitality dynamics
    dynamic_layers = sum(1 for layer_idx in range(6)
                        if max([m.layer_vitalities[layer_idx] for m in apoptotic_trainer.metrics_history]) -
                           min([m.layer_vitalities[layer_idx] for m in apoptotic_trainer.metrics_history]) > 0.3)
    if dynamic_layers >= 2:
        success_score += 1
        criteria.append(f"✓ Vitality dynamics ({dynamic_layers} layers)")
    else:
        criteria.append("✗ Flat vitality (senescence not working)")

    # 6. Competitive performance
    if abs(loss_diff) < 0.2:
        success_score += 1
        criteria.append("✓ Competitive with baseline")
    else:
        criteria.append(f"✗ {abs(loss_diff):.3f} worse than baseline")

    # 7. Convergence
    if apoptotic_variance < baseline_variance * 1.5:
        success_score += 1
        criteria.append("✓ Stable convergence")
    else:
        criteria.append("✗ Unstable convergence")

    # Print criteria
    for c in criteria:
        report.append(c)

    report.append("")
    report.append(f"SUCCESS SCORE: {success_score}/{max_score}")
    report.append("")

    if success_score >= 6:
        report.append("🎉 EXCELLENT: Apoptosis working well!")
        report.append("   → Try domain shift experiments to see adaptation benefits")
    elif success_score >= 4:
        report.append("🤔 PROMISING: Core mechanics working, needs tuning")
        report.append("   → Try aggressive hyperparameters (shorter lifespan, higher plasticity)")
    elif success_score >= 2:
        report.append("⚠️  NEEDS WORK: Some issues detected")
        report.append("   → Check vitality dynamics and gradient health")
        report.append("   → Consider alternative experiments (flip zones, gradual death)")
    else:
        report.append("✗ FAILED: Major issues")
        report.append("   → Debug: Check if senescence is updating")
        report.append("   → Verify apoptosis triggers are firing")
        report.append("   → Consider starting with simpler baseline")

    # === SECTION 10: Hyperparameters Used ===
    report.append("\n⚙️  HYPERPARAMETERS USED")
    report.append("="*70)
    report.append(f"max_lifespan:        {apoptosis_manager.max_lifespan}")
    report.append(f"maturation_period:   {apoptosis_manager.maturation_period}")
    report.append(f"apoptosis_interval:  {apoptosis_manager.apoptosis_interval}")
    report.append(f"vitality_threshold:  {apoptosis_manager.vitality_threshold}")
    report.append(f"plasticity_ceiling:  {apoptosis_manager.plasticity_ceiling}")
    report.append(f"mutation_strength:   {apoptosis_manager.mutation_strength}")

    report.append("\n" + "="*70)
    report.append("END DIAGNOSTICS")
    report.append("="*70 + "\n")

    return "\n".join(report)


# === RUN THIS TO GENERATE REPORT ===
# Paste this entire output into chat for analysis!

diagnostics = generate_diagnostics_report(
    baseline_trainer,
    apoptotic_trainer,
    apoptosis_manager
)

print(diagnostics)

# Also save to file
with open('diagnostics_report.txt', 'w') as f:
    f.write(diagnostics)

print("\n✓ Report also saved to: diagnostics_report.txt")
