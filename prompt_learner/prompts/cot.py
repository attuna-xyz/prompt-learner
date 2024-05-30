"""Class for the CoT prompting."""
from pydantic import Field
from .prompt import Prompt


class CoT(Prompt):
    """Think step by step"""
    custom_instructions: str = Field(description="""Custom intructions for Chain
                                 of Thought Prompting""",
                                 default="Think step by step.\n")
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        if self.template.task.selected_examples == []:
            self.prompt = f"""{self.template.descriptor}{self.custom_instructions}"""
        else:
            self.prompt = f"""{self.template.descriptor}{self.template.examples_preamble}
        {self.template.format_examples(self.template.task.selected_examples)}{self.custom_instructions}"""