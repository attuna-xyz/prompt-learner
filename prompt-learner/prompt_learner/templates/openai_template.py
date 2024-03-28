class OpenAICompletionTemplate:
    def __init__(self):
        self.template= f"""You are a helpful AI assistant. You are helping a user with a task. The user asks you to {self.prompt}. 
        You respond to the user by saying"""
        