"""Defines the contract for a Tagging task."""

from prompt_learner.examples.example import Example
from .task import Task


class TaggingTask(Task):
    "Tagging"
    def validate_example(self, example: Example):
        """Validate the example for tagging task."""
        labels = example.label.split(',')  # Handle multiple labels
        allowed_labels_set = set(self.allowed_labels)  # Convert to set
        if not all(label.strip() in allowed_labels_set for label in labels):
            return False
        return True
