"""Generic Task class."""

from typing import List
from pydantic import BaseModel, Field
from examples.example import Example
from adapters.adapter import Adapter


class Task(BaseModel):
    """Defines the contract for a Generic task."""
    description: str = Field(description="The name of the task.")
    allowed_labels: List[str] = Field(description="Allowed labels for task.")
    examples: List[Example] = []

    def validate_example(self, example: Example):
        """Validate the example for the task."""
        # This method will be overridden in subclasses

    def add_example(self, example: Example):
        """Add an example to the task."""
        if not self.validate_example(example):
            raise ValueError(f"""Label '{example.label}' is not in
                             allowed labels for this task.""")
        self.examples.append(example)

    def predict(self, adapter: Adapter, prompt: str):
        """Predict the label for the given text using the prompt."""
        return adapter.process_output(adapter.llm.invoke(prompt))
    
    def validate_prediction(self, text: str, prediction: str):
        """Validate the prediction."""
        example = Example(text=text, label=prediction)
        return self.validate_example(example)
