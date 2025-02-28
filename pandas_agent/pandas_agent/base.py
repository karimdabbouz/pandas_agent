from abc import ABC, abstractmethod
from typing import TypedDict
import datetime


class PandasState(TypedDict):
    name: str
    columns: list[dict]



class Memory(TypedDict):
    prompt: str
    submitted_at: datetime.datetime
    pandas_state: list[PandasState]
    executed: bool
    approved: bool



class LLMConnector(ABC):
    '''
    Defines the connection to an LLM and the methods it needs to implement.
    '''

    @abstractmethod
    def __init__(self, api_key: str, model: str):
        '''
        Requires an OpenAI API key and a model name.
        '''
        pass
    

    @abstractmethod
    def send_request(self, user_prompt: str, system_prompt: str):
        '''
        Prompts the LLM.
        If sytem_prompt is omitted, it will be included in the user prompt.
        '''
        pass



class SharedMemory():
    '''
    Represents shared memory of the whole workflow including prompts, answers and the manipulations done on the data.
    '''
    def __init__(
        self
    ):
        self.state: list[Memory] = []


    def add_memory(self, memory: Memory):
        '''
        Adds a memory.
        '''
        self.state.append(memory)


    def update_latest_memory(self, memory: Memory):
        '''
        Replaces the last memory stored with a new memory.
        '''
        self.state[-1] = memory


    def get_last_memory(self) -> Memory:
        '''
        Returns the last memory.
        '''
        return self.state[-1]


class Agent(ABC):
    '''
    Defines an agent and the methods it needs to implement.
    Expects an instance of LLMConnector with a send_request method.
    '''  
    @abstractmethod
    def __init__(self, llm_connector: LLMConnector):
        pass
    

    @abstractmethod
    def format_memory(self, memory: list[Memory]):
        '''
        Formats the memory including the latest task as LLM-friendly input and returns it.
        Expects a list of memories.
        '''
        pass


    @abstractmethod
    def execute(self, llm_output: str):
        '''
        Takes the output of an LLM and executes it in Jupyter notebook using get_ipython().run_cell().
        '''
        pass


    @abstractmethod
    def update_memory(self, memory: SharedMemory):
        '''
        Parses the new state of the workflow and updates the memory.
        '''
        pass


    @abstractmethod
    def run(self):
        '''
        Entry point of the agent.
        '''
        pass