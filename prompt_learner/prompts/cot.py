"""Class for the CoT prompting."""
from pydantic import Field
from prompts.prompt import Prompt


class CoT(Prompt):
    """Think step by step"""
    custom_intructions: str = Field(description="""Custom intructions for Chain
                                 of Thought Prompting""",
                                 default="Think step by step.")
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        self.prompt = f"""{self.template.descriptor}\n
        {self.template.examples_preamble}
        \n{self.template.format_examples(self.selector.selected_examples)}
        \n{self.custom_intructions}"""
