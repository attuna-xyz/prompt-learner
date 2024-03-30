class OpenAICompletionTemplate:
    def __init__(self,task_description:str, task_type:str):
        self.task_description=task_description
        self.task_type=task_type
        self.template= f"""You are a helpful AI assistant. You are helping a user with a {task_type} task. The user asks you to {task_description}"""
        