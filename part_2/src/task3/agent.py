from contextlib import suppress

import requests


SYSTEM_PROMPT = """You are a chat bot.
Your goal is to help the user navigate a dataset.
The name of the dataset is Adult Census Income. It contains records from the 1994 US Census
It has 48k rows and 14 features: age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country and income
It is released under the CC BY 4.0 license"""


class OllamaException(Exception):
    pass


def agent(
        model: str,
        think: bool = False,
        ollama_url: str = "http://localhost:11434"
):
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]
    with suppress(KeyboardInterrupt, EOFError):
        while True:
            user = input(">>> ")
            messages.append({
                "role": "user",
                "content": user
            })
            assistant = chat(
                model=model,
                messages=messages,
                think=think,
                ollama_url=ollama_url
            )
            print(assistant)
            messages.append({
                "role": "assistant",
                "content": assistant
            })


def chat(
        model: str,
        messages: list[dict],
        think: bool,
        ollama_url: str,
):
    with requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "think": think,
                "stream": False
            }
    ) as response:
        packet = response.json()

    if error := packet.get("error"):
        raise OllamaException(error)
    return packet["message"]["content"]
