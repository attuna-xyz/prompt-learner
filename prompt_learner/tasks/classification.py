"""Defines the contract for a classification task."""

from typing import List
from tasks.task import Task
from examples.example import Example


class ClassificationTask(Task):
    """Classification"""
    allowed_labels: List[str] = []

    def add_example(self, example: Example):
        if example.label not in self.allowed_labels:
            raise ValueError(f"""Label '{example.label}' is not in
                             allowed labels for this task.""")
        self.examples.append(example)
