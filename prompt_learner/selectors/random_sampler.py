"""Randomly picks n samples and injects them into the prompt."""
import random
from .selector import Selector
random.seed(42)


class RandomSampler(Selector):
    """Randomly picks n samples and injects them into the prompt."""

    def select_examples(self):
        """Select examples for the task randomly""" 
        random.shuffle(self.all_examples)
        self.selected_examples = self.all_examples[:self.num_samples]
        self.task.selected_examples = self.selected_examples
        return self

    def __repr__(self):
        return f"""RandomSampler(task={self.task},
        num_samples={self.num_samples}"""
