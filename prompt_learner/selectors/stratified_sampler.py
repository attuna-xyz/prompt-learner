"""Randomly picks n samples and injects them into the prompt."""
import random
from .selector import Selector
random.seed(42)


class StratifiedSampler(Selector):
    """Randomly picks n samples for each label.
    Total number of samples selected = num_samples * num_labels"""

    def select_examples(self):
        """Select n examples from each class for the task randomly"""
        self.all_examples = self.task.examples
        self.selected_examples = []
        for label in self.task.allowed_labels:
            examples = [example for example in self.all_examples if example.label == label]
            random.shuffle(examples)
            self.selected_examples += examples[:self.num_samples]
        self.task.selected_examples = self.selected_examples

    def __repr__(self):
        return f"""StratifiedSampler(task={self.task},
        num_samples={self.num_samples}"""
