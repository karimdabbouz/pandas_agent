from .base import LLMConnector
from openai import OpenAI
import re


class OpenAIConnector(LLMConnector):
    '''
    Connection to OpenAIs API.
    '''
    def __init__(self, api_key:str, model:str):
        self.api_key=api_key
        self.model=model
        self.client=OpenAI(api_key=api_key)
        

    def send_request(self, user_prompt: str, system_prompt: str):
        '''
        Prompts the LLM.
        If sytem_prompt is omitted, it will be included in the user prompt.
        '''
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            store=True,
            messages=messages
        )
        response = re.sub(r'^```(python)?\n|\n```$', '', completion.to_dict()['choices'][0]['message']['content'], flags=re.MULTILINE)
        return response