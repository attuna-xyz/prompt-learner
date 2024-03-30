"""Class for a Generic prompt Template"""

from typing import List, Any
from pydantic import BaseModel, Field
from tasks.task import Task


class Template(BaseModel):
    """Defines the contract for a Generic template."""
    task_description: str = Field(default="", description="Describes the task")
    task_type: str = Field(default="", description="Type of task")
    allowed_labels: List[Any] = Field(default=[],
                                      description="Allowed labels for task")
    examples: List[Any] = Field(default=[],
                                description="Examples for the task")
    prompt: str = Field(default="", description="Prompt for the task")

    def __init__(self, task: Task, **kwargs):
        super().__init__(**kwargs)
        self.task_description = task.description
        self.task_type = task.__doc__
        self.allowed_labels = task.allowed_labels
        self.examples = task.examples

    def format_examples(self):
        """Add an example to the task."""
        # This method will be overridden in subclasses

    def add_prediction_sample(self, text: str):
        """Add inference instructions to task."""
        # This method will be overridden in subclasses
