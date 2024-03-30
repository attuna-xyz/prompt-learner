"""Generic Task class."""

from typing import List
from pydantic import BaseModel, Field
from examples.example import Example


class Task(BaseModel):
    """Defines the contract for a Generic task."""
    description: str = Field(description="The name of the task.")
    allowed_labels: List[str] = Field(description="Allowed labels for task.")
    examples: List[Example] = []

    def add_example(self, example: Example):
        """Add an example to the task."""
        # This method will be overridden in subclasses
