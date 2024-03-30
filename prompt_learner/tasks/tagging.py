"""Defines the contract for a Tagging task."""

from tasks.task import Task
from examples.example import Example


class TaggingTask(Task):
    "Tagging"
    def add_example(self, example: Example):
        # Assuming tagging tasks might allow multiple labels per example
        labels = example.label.split(',')  # Handle multiple labels
        allowed_labels_set = set(self.allowed_labels)  # Convert to set
        if not all(label.strip() in allowed_labels_set for label in labels):
            raise ValueError("""One or more labels are not in the
                             allowed labels for this tagging task.""")
        self.examples.append(example)
