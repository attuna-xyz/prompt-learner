"""A Generic Optimizer class."""
from prompt_learner.prompts.prompt import Prompt


class Optimizer:
    """An optimizer can search for the best hyperparameters for a model.
    It also computes metrics for different configurations."""
    def __init__(self, prompt: Prompt):
        self.prompt = prompt

    def search(self, param_grid: dict):
        """Search for the best hyperparameters given in the 
        parameter grid."""
  
    def compute_metrics(self):
        """Compute metrics for a given prompt and adapter on a task."""