# tasks/task.py
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union
from examples.example import Example

class Task(BaseModel):
    description: str = Field(..., description="The name of the task.")
    allowed_labels: List[str] = Field(..., description="Allowed labels for the task.")
    examples: List[Example] = []

    def add_example(self, example: Example):
        # This method will be overridden in subclasses with specific validations
        pass