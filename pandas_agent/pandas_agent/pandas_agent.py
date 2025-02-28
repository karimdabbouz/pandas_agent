from .base import PandasState, Memory, SharedMemory, Agent
from .openai import OpenAIConnector
import pandas as pd
import datetime
from IPython.display import display, Code
from IPython import get_ipython
from IPython.utils.capture import capture_output




class PrimaryAgent(Agent):
    '''
    The primary agent to work with.
    '''
    def __init__(self, llm_connector: LLMConnector, log_responses=False):
        self.llm_connector=llm_connector
        self.log_responses=log_responses
        self.system_prompt='''You are an AI agent for working with the Python pandas library. You will be given a task and the state of previous tasks if there were any. You will provide pure python code to execute the task. Here are your instructions:

1. RESPONSE FORMAT AND RULES
- you must respond with pure python code
- you are not allowed to include comments
- you are not allowed to include a title for your response
- you are not allowed to provide multiple solutions. only ever include the single best code for the task
- when working with dataframes, always show the dataframe by calling it as the last statement in your code
- you can use print statements to display other data
- your code will be executed in a jupyter notebook which will take care of formatting


2. IMPORTS AND STATE
- you must keep track of variables as you set them
- you must take care of all necessary imports
- all input and output files are in the current folder ./
 .
'''


    def format_memory(self, state: list[Memory]):
        '''
        Formats the memory including the latest task as LLM-friendly input and returns it.
        Expects a list of memories.
        '''
        llm_formatted_memory = ''
        last_prompts = ''.join([f'{i}) {v['prompt']} \n' for i, v in enumerate(state[:-1])])
        llm_formatted_memory += f'''Here is what I have asked you so far: {last_prompts}. This is the current state of our data: {str(state[-2]['pandas_state'])}. Please do the following now: {state[-1]['prompt']}'''
        return llm_formatted_memory


    def update_memory(self, memory: SharedMemory):
        '''
        Parses the new state of the workflow and updates the memory.
        '''
        result: PandasState = []
        global_state = globals()
        all_dfs = {k for k, v in global_state.items() if isinstance(v, pd.DataFrame) and not k.startswith('_')}
        for df_name in all_dfs:
            column_list = []
            for i, v in enumerate(global_state[df_name].columns.tolist()):
                column_list.append({
                    'column_name': v,
                    'dtype': str(global_state[df_name].dtypes.tolist()[i])
                })
            result.append({
                'name': df_name,
                'columns': column_list
            })
        current_memory = memory.get_last_memory()
        new_memory = {
            'prompt': current_memory['prompt'],
            'submitted_at': current_memory['submitted_at'],
            'pandas_state': result,
            'executed': True,
            'approved': current_memory['approved']
        }
        memory.update_latest_memory(new_memory)
    

    def execute(self, llm_output: str):
        '''
        Takes the output of an LLM and executes it in Jupyter notebook using get_ipython().run_cell().
        '''
        with capture_output() as captured:
            get_ipython().run_cell(llm_output)
        stdouts = captured.stdout.split('\n')
        print_statements = stdouts[:-1] if len(stdouts[:-1]) > 0 else None
        cell_output = captured.outputs[0] if len(captured.outputs) > 0 else None
        if cell_output:
            display(cell_output)
        if print_statements:
            for statement in print_statements:
                print(statement)

    
    def run(self, memory: SharedMemory):
        '''
        Entry point of the agent.
        '''
        if memory.state[-1]['executed']:
            raise Exception('There is nothing new to do. Add a new task to continue.')
        else:
            if len(memory.state) == 1:
                user_prompt = f'Here is your task: {memory.state[0]['prompt']}'
                llm_output = self.llm_connector.send_request(system_prompt=self.system_prompt, user_prompt=user_prompt) # EXCEPTION HANDLING
                if self.log_responses:
                    display(Code(llm_output, language='python'))
                self.execute(llm_output)
                # HIER MÜSSTE DER REFEREE NOCHMAL DRAUFSCHAUEN
                    # -> vorher oder nachher?
                self.update_memory(memory)
            else:
                user_prompt = self.format_memory(memory.state)
                llm_output = self.llm_connector.send_request(system_prompt=self.system_prompt, user_prompt=user_prompt) # EXCEPTION HANDLING
                if self.log_responses:
                    display(Code(llm_output, language='python'))
                self.execute(llm_output)
                # HIER MÜSSTE DER REFEREE NOCHMAL DRAUFSCHAUEN
                    # -> vorher oder nachher?
                self.update_memory(memory)




class PandasAgent():
    '''
    Main agent the user interacts with.
    Expects an instance of LLMConnection.
    '''
    def __init__(
        self,
        llm_connector,
        log_responses,
        # referee=False
    ):
        self.llm_connector=llm_connector
        self.shared_memory=SharedMemory()
        self.primary_agent=PrimaryAgent(llm_connector=llm_connector, log_responses=log_responses)
        # self.referee_agent=RefereeAgent(llm_connector=llm_connector, log_responses=log_responses) if referee else None


    def add_task(self, prompt: str):
        '''
        Takes a single prompt and adds it to the shared memory.
        '''
        memory: Memory = {
            'prompt': prompt,
            'submitted_at': datetime.datetime.now(),
            'pandas_state': None,
            'executed': False,
            'approved': False
        }
        self.shared_memory.add_memory(memory)


    def get_open_tasks(self) -> list[str]:
        '''
        Returns all open tasks.
        '''
        return self.task_queue


    def action(self):
        '''
        Executes the tasks in task_queue
        '''
        if len(self.shared_memory.state) == 0:
            raise Exception('Memory is empty. Add a task to get started.')
        else:
            self.primary_agent.run(self.shared_memory)