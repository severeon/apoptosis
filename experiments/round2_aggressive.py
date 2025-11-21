"""
Aggressive hyperparameters for Round 2 of apoptosis experiment.

Changes from Round 1 (conservative):
- Shorter lifespan: 3000 → 1500 (more deaths)
- Faster checks: 500 → 250 (quicker response)
- Higher plasticity: 0.5 → 0.75 (young layers contribute more)
- Stronger mutation: 0.2 → 0.3 (more diversity)
- Faster maturation: 750 → 500 (young layers grow up faster)
"""

# Round 2: AGGRESSIVE apoptosis
AGGRESSIVE_CONFIG = {
    'max_lifespan': 1500,         # Die faster (was 3000)
    'maturation_period': 500,      # Mature faster (was 750)
    'apoptosis_interval': 250,     # Check more often (was 500)
    'vitality_threshold': 0.15,    # Die a bit earlier (was 0.1)
    'plasticity_ceiling': 0.75,    # Young layers stronger (was 0.5)
    'mutation_strength': 0.3,      # More exploration (was 0.2)
    'base_temp': 1.0,
    'temp_range': 0.5,
    'base_dropout': 0.1,
    'dropout_range': 0.3,
}

# Expected outcomes:
# - ~20-25 apoptosis events (instead of ~10)
# - More dynamic vitality patterns
# - Young layers contribute more meaningfully
# - Potentially see adaptation benefits

# Round 3: EXTREME (if round 2 still doesn't show benefits)
EXTREME_CONFIG = {
    'max_lifespan': 800,           # Die even faster
    'maturation_period': 300,      # Mature very fast
    'apoptosis_interval': 200,     # Check very often
    'vitality_threshold': 0.2,     # Die earlier
    'plasticity_ceiling': 0.9,     # Almost full influence when young
    'mutation_strength': 0.4,      # High exploration
    'base_temp': 1.0,
    'temp_range': 0.7,             # More temperature variation
    'base_dropout': 0.1,
    'dropout_range': 0.4,          # More dropout variation
}

# Expected outcomes:
# - ~40-50 apoptosis events (constant turnover)
# - Very dynamic, potentially chaotic
# - High risk of instability, but high potential reward
# - If this doesn't help, the hypothesis might be wrong


# Alternative experiment: FLIP THE ZONES
FLIP_ZONES_CONFIG = {
    # Same as aggressive, but in the notebook you'll swap:
    # - Death zone: layers 0-1 (was 4-5)
    # - Birth zone: layers 4-5 (was 0-1)
    # Test if killing INPUT layers (instead of OUTPUT layers) helps
    **AGGRESSIVE_CONFIG
}


# Alternative experiment: GROUP APOPTOSIS
GROUP_APOPTOSIS_CONFIG = {
    'max_lifespan': 1500,
    'maturation_period': 500,
    'apoptosis_interval': 500,     # Less frequent
    'vitality_threshold': 0.15,
    'plasticity_ceiling': 0.75,
    'mutation_strength': 0.3,
    'base_temp': 1.0,
    'temp_range': 0.5,
    'base_dropout': 0.1,
    'dropout_range': 0.3,
    'kill_multiple_layers': True,  # NEW: kill both death zone layers at once
}

print("Round 2 Configs Loaded!")
print(f"Aggressive: {AGGRESSIVE_CONFIG['max_lifespan']} steps lifespan")
print(f"Extreme: {EXTREME_CONFIG['max_lifespan']} steps lifespan")
