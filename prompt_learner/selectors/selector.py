"""A Generic Example Selector class."""
from typing import List
from pydantic import BaseModel, Field
from prompt_learner.tasks.task import Task
from prompt_learner.examples.example import Example


class Selector(BaseModel):
    """Defines the contract for a Generic Example Selector."""
    all_examples: List[Example] = []
    selected_examples: List[Example] = []
    task: Task
    num_samples: int = Field(description="Number of examples to samples", default=1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_examples = self.task.examples

    def select_examples(self):
        """Select examples for the task."""
        self.all_examples = self.task.examples
        self.selected_examples = self.examples
        self.task.selected_examples = self.select_examples
        # This method will be overridden in subclasses
