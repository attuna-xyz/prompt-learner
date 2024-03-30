# examples/outputs.py
from pydantic import BaseModel, Field

class Example(BaseModel):
    text: str = Field(description="Input text")
    label: str = Field(..., description="Output label")
