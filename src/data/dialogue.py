from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json


class BaseDialogueProvider:
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        pass


class DummyDialogueProvider(BaseDialogueProvider):
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        return [('Alice', 'Hello!, how are you'), ('Bob', 'Hello, I am fine, what about you?'), ('Alice', 'I am sick.')]


class MistralDialogueProvider(BaseDialogueProvider):
    def __init__(self, token: str, model_id: str) -> None:
        self._token = token
        self.client = MistralClient(api_key=token)
        self.model_id = model_id
        self.prompt_template = "Generate dialogue with two people - Bob and Alice. They should discuss the following theme: {theme}. Please generate the dialogue in the following json format:"
        self.example = """[
{"speaker": "Bob",
"phrase" : "Hello Alice!"
},
{"speaker" : "Alice",
"phrase" : "Hello Bob!"
}
]
"""

    def generate(self, theme: str):
        messages = [
            ChatMessage(role="user", content=self.prompt_template.format(theme=theme) + self.example)
        ]
        chat_response = self.client.chat(
            model=self.model_id,
            messages=messages,
        ).choices[0].message.content
        return [(x['speaker'], x['phrase']) for x in json.loads(chat_response)]
