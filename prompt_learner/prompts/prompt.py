"""A class for a Generic Prompt."""
from pydantic import Field, BaseModel
from prompt_learner.optimizers.selectors.selector import Selector
from prompt_learner.optimizers.selectors.random_sampler import RandomSampler
from prompt_learner.templates.template import Template


class Prompt(BaseModel):
    """Defines the contract for a Generic Prompt."""
    template: Template
    selector: Selector = Field(description="Selector for the task.")
    prompt: str = Field(description="Final prompt string.", default="")

    def select_examples(self):
        """Select examples for the task."""
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        self.prompt = f"""{self.template.descriptor}\n{self.template.examples_preamble}
        \n{self.template.format_examples(self.selector.selected_examples)}"""

    def add_inference(self, text: str):
        """Add inference sample"""
        self.prompt = self.prompt + self.template.add_prediction_sample(text)