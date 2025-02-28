import pytest
from pandas_agent import PandasAgent
from pandas_agent.openai import OpenAIConnector
import datetime

def test_add_task():
    # Initialize with mock connector
    llm_connector = OpenAIConnector(api_key="fake-key", model="fake-model")
    agent = PandasAgent(llm_connector=llm_connector, log_responses=False)
    
    # Add a task
    test_prompt = "test task"
    agent.add_task(test_prompt)
    
    # Verify task was added correctly
    memory = agent.shared_memory.get_last_memory()
    assert memory["prompt"] == test_prompt
    assert isinstance(memory["submitted_at"], datetime.datetime)
    assert memory["pandas_state"] is None
    assert memory["executed"] is False
    assert memory["approved"] is False 