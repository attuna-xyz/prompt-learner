"""Defines the contract for a text to sql task."""

from prompt_learner.examples.example import Example
from .task import Task


class SQLGenerationTask(Task):
    """SQLGeneration"""

    def validate_example(self, example: Example):
        """Validate the example for SQL Generation task."""
        if example.context is None: #need schema as context
            return False
        return True
