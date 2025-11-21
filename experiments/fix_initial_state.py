"""
FIX: Start all layers at full vitality, let them naturally cycle.

The problem: Starting with layers 0-1 at 50% vitality cripples the model.
The solution: Start everyone at 100%, let senescence happen naturally.
"""

# === OPTION A: All Start Healthy (Best Fix) ===

def _initialize_senescence_v2(self) -> List[SenescenceMetadata]:
    """Initialize senescence - ALL layers start healthy."""
    metadata = []
    for i in range(self.n_layers):
        if i < 2:  # Layers 0-1: will eventually be birth zone
            zone = "birth"
            vitality = 1.0  # 🔥 START AT 100% (was 0.5)
            influence = 1.0  # 🔥 FULL POWER (was 0.5)
            age = 0  # Young, will grow
        elif i >= 4:  # Layers 4-5: will eventually be death zone
            zone = "death"
            vitality = 1.0  # Start healthy
            influence = 1.0
            age = 0  # Young, will age and die
        else:  # Layers 2-3: stable core
            zone = "stable"
            vitality = 1.0
            influence = 1.0
            age = 0

        meta = SenescenceMetadata(
            age=age,
            layer_zone=zone,
            vitality=vitality,
            influence_weight=influence
        )
        metadata.append(meta)
    return metadata


# === OPTION B: No Zones Initially (Alternative) ===

def _initialize_senescence_neutral(self) -> List[SenescenceMetadata]:
    """Initialize senescence - ALL layers neutral, zones activate later."""
    metadata = []
    for i in range(self.n_layers):
        # Everyone starts as "stable" - zones kick in after 500 steps
        meta = SenescenceMetadata(
            age=0,
            layer_zone="stable",  # No zones yet
            vitality=1.0,
            influence_weight=1.0
        )
        metadata.append(meta)
    return metadata

# Then in ApoptosisManager, after 500 steps, assign zones:
def activate_zones(self):
    """Activate death/birth zones after warmup period."""
    if self.step_count == 500:
        print("\n🔥 Activating senescence zones!")
        for i, meta in enumerate(self.model.senescence):
            if i < 2:
                meta.layer_zone = "birth"
            elif i >= 4:
                meta.layer_zone = "death"
            # 2-3 stay stable


# === OPTION C: Delayed Apoptosis (Simplest) ===

# Just add warmup steps to ApoptosisManager:
class ApoptosisManager:
    def __init__(self, model, warmup_steps=1000, **kwargs):
        # ... existing code ...
        self.warmup_steps = warmup_steps

    def step(self):
        self.step_count += 1

        # Don't apply senescence during warmup
        if self.step_count < self.warmup_steps:
            return False

        # Now apply senescence normally
        for meta in self.model.senescence:
            meta.update(...)

        if self.step_count % self.apoptosis_interval == 0:
            return self.trigger_apoptosis()

        return False


# === TESTING THE FIX ===

# Create model with fixed initialization
class ApoptoticTransformerV2(ApoptoticTransformer):
    def _initialize_senescence(self):
        return _initialize_senescence_v2(self)

# Or just patch it:
apoptotic_model = ApoptoticTransformer(...)
apoptotic_model._initialize_senescence = lambda: _initialize_senescence_v2(apoptotic_model)
apoptotic_model.senescence = apoptotic_model._initialize_senescence()

print("✓ Fixed: All layers start at full vitality")
print("  Layers will naturally age into their roles")
