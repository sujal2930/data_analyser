from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import os

# Mock API key for initialization (won't actually call API)
os.environ["OPENAI_API_KEY"] = "sk-test"

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

try:
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-functions",
        allow_dangerous_code=True
    )
    print("Agent created successfully with string type")
except Exception as e:
    print(f"Error: {e}")
