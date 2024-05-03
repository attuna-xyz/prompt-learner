"""A class for a Generic Translation of String Prompt."""
import json
from typing import List
from pydantic import Field, BaseModel
from prompt_learner.templates.template import Template
from prompt_learner.tasks.task import Task
from prompt_learner.examples.example import Example
from langchain_openai import ChatOpenAI  
    

class GetModules(BaseModel):  
	"""Given a prompt string, extract the different individual modules
    You should extract all 3 modules - task_description, allowed_labels, examples
    If you are unable to extract any of the modules, return an empty string for that module"""  
	task_description: str = Field(description="The description of task being solved in prompt")
	allowed_labels: List[str] = Field(description="A list of permissable labels")
	examples: List[Example] = Field(description="A list of examples given in the prompt")

class Translate(BaseModel):
    """Defines the contract for a Generic Translation."""
    input_prompt: str = Field(description="Input prompt string.", default="")
    task: Task = Field(description="Task type which the string is solving for.",default=None)
    prompt: str = Field(description="Final prompt string.", default="")
    template: Template = Field(description="Template for the prompt.", default=None)
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        self.prompt = f"""{self.template.descriptor}{self.template.examples_preamble}
        {self.template.format_examples(self.template.task.selected_examples)}"""
    
    def create_task(self, task_description: str, allowed_labels: list):
        """Create a task for the prompt."""
        self.task = self.task(description=task_description, allowed_labels=allowed_labels)

    def create_examples(self, examples: list):
        """Create examples for the task."""
        for example in examples:
            self.task.add_example(Example(text=example.text, label=example.label))
        

    def extract_modules(self):
        """Extract different modules from the string using LLM."""
        
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        llm_with_tools = llm.with_structured_output(GetModules,method="json_mode")
        task_message = "You are a a helpful AI Assistant. You have to take this next input prompt and extract the task description, allowed labels and examples from it. Give the result in a json. Here is the input prompt: "
        ai_msg = llm_with_tools.invoke(task_message+self.input_prompt)
        extracted_modules = GetModules(**ai_msg)
        return extracted_modules
        
    
    def translate(self, task: Task, template: Template):
        """Translate the prompt to the new template."""
        self.task = task
        all_modules = self.extract_modules()
        self.create_task(all_modules.task_description, all_modules.allowed_labels)
        self.create_examples(all_modules.examples)
        self.template = template(task=self.task)
        self.assemble_prompt()