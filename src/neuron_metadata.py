from attr import dataclass

@dataclass
class NeuronMetadata:
    """Metadata for a single neuron."""
    age: int = 0
    birth_step: int = 0
    fitness_history: list = None # type: ignore

    def __post_init__(self):
        if self.fitness_history is None:
            self.fitness_history = []
