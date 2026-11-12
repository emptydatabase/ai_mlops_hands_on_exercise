import os

from dotenv import load_dotenv

from task3 import agent

load_dotenv()

agent(
    model=os.environ["OLLAMA_MODEL"]
)
