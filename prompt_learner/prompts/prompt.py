"""A class for a Generic Prompt."""
from pydantic import Field, BaseModel
from prompt_learner.selectors.selector import Selector
from prompt_learner.selectors.random_sampler import RandomSampler
from prompt_learner.templates.template import Template


class Prompt(BaseModel):
    """Defines the contract for a Generic Prompt."""
    template: Template
    prompt: str = Field(description="Final prompt string.", default="")

    def select_examples(self):
        """Select examples for the task."""
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        if self.template.task.selected_examples == []:
            self.prompt = f"""{self.template.descriptor}"""
        else:
            self.prompt = f"""{self.template.descriptor}{self.template.examples_preamble}
        {self.template.format_examples(self.template.task.selected_examples)}"""

    def add_inference(self, text: str, context: str = ""):
        """Add inference sample"""
        self.prompt = self.prompt + self.template.add_prediction_sample(text,context)
    
    def translate(self, template: Template):
        """Translate the prompt to the new template."""
        self.template = template(task=self.template.task)
        self.assemble_prompt()