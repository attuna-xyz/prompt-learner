from tasks.task import Task
from pydantic import Field, validator
from typing import List
from examples.example import Example

class TaggingTask(Task):
    def add_example(self, example: Example):
        # Assuming tagging tasks might allow multiple labels per example, adjust accordingly.
        labels = example.label.split(',')  # Example way to handle multiple labels
        if not all(label.strip() in self.allowed_labels for label in labels):
            raise ValueError(f"One or more labels are not in the allowed labels for this tagging task.")
        self.examples.append(example)
