"""Defines the contract for a classification task."""

from typing import List
from examples.example import Example
from .task import Task


class ClassificationTask(Task):
    """Classification"""
    allowed_labels: List[str] = []

    def validate_example(self, example: Example):
        """Validate the example for tagging task."""
        labels = example.label.split(',')  # Handle multiple labels
        allowed_labels_set = set(self.allowed_labels)  # Convert to set
        if not all(label.strip() in allowed_labels_set for label in labels):
            return False
        return True
