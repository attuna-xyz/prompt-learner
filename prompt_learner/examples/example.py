"""Example Pydantic model for a Task"""
from pydantic import BaseModel, Field


class Example(BaseModel):
    """An Example for a Task"""
    text: str = Field(description="Input text")
    context: str = Field(description="Context for the input text", default="")
    label: str = Field(description="Output label")
