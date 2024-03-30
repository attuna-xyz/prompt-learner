class AnthropicCompletionTemplate:
    def __init__(self,task_description:str, task_type:str):
        self.task_description=task_description
        self.task_type=task_type
        self.template= f"""You are a helpful AI assistant. You are helping a user with a {task_type} task.
        You have to perform the following task. 
        <task_description>{task_description}</task_description>"""
        