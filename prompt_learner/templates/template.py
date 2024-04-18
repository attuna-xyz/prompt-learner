"""Class for a Generic prompt Template"""

from typing import List, Any
from pydantic import BaseModel, Field
from prompt_learner.tasks.task import Task
from prompt_learner.examples.example import Example


class Template(BaseModel):
    """Defines the contract for a Generic template."""
    task_description: str = Field(default="", description="Describes the task")
    task_type: str = Field(default="", description="Type of task")
    task: Task
    allowed_labels: List[Any] = Field(default=[],
                                      description="Allowed labels for task")
    examples: List[Any] = Field(default=[],
                                description="Examples for the task")
    descriptor: str = Field(default="", description="Descriptor for the task")
    examples_preamble: str = Field(default="",
                                   description="Preamble for examples")
    prediction_preamble: str = Field(default="",
                                     description="Preamble forprediction")

    def __init__(self, **kwargs):
    
        super().__init__(**kwargs)
        self.task_description = self.task.description
        self.task_type = self.task.__doc__
        self.allowed_labels = self.task.allowed_labels
        
    def format_examples(self, examples: List[Example]):
        """Add an example to the task."""
        # This method will be overridden in subclasses

    def add_prediction_sample(self, text: str, context: str):
        """Add inference instructions to task."""
        # This method will be overridden in subclasses
