from tasks.task import Task
from pydantic import Field, validator
from typing import List
from examples.example import Example

class ClassificationTask(Task):
    """Classification"""
    def add_example(self, example: Example):
        if example.label not in self.allowed_labels:
            raise ValueError(f"Label '{example.label}' is not in the allowed labels for this classification task.")
        self.examples.append(example)