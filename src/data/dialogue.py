from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class BaseDialogueProvider:
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        pass


class DummyDialogueProvider(BaseDialogueProvider):
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        return [('Alice', 'Hello!, how are you'), ('Bob', 'Hello, I am fine, what about you?'), ('Alice', 'I am sick.')]


class PromptProvider:
    def __init__(self):
        self.prompt_template = ("Generate dialogue with two people - Bob and Alice. They should discuss the following "
                                "theme: {theme}. The dialogue should contain at least 3 utterances from each speaker. "
                                "Please generate the dialogue in the following json format:")
        self.example = r"""[
        {"speaker": ... 
        "phrase" : ...
        },
        ...
        ]
        """

    def provide_prompt(self, theme: str) -> str:
        return self.prompt_template.format(theme=theme) + self.example


class MistralDialogueProvider(BaseDialogueProvider, PromptProvider):
    def __init__(self, token: str, model_id: str) -> None:
        super().__init__()
        self._token = token
        self.client = MistralClient(api_key=token)
        self.model_id = model_id

    def generate(self, theme: str):
        messages = [
            ChatMessage(role="user", content=self.provide_prompt(theme))
        ]
        chat_response = self.client.chat(
            model=self.model_id,
            messages=messages,
        ).choices[0].message.content
        return [(x['speaker'], x['phrase']) for x in json.loads(chat_response)]


class OpenaiDialogueProvider(BaseDialogueProvider, PromptProvider):
    def __init__(self, host: str = "http://localhost:8000/v1"):
        super().__init__()
        chat = ChatOpenAI(openai_api_base=host, openai_api_key='none')
        prompt = PromptTemplate.from_template(
            "Generate dialogue with two people - Bob and Alice. They should discuss the following "
            "theme: {theme}. The dialogue should contain at least 4 utterances from each speaker. "
            "Please generate the dialogue in the following list json format:" +
            """[[
                {{"speaker": ... 
                "phrase" : ...
                }},
                {{"speaker": ...
                "phrase" : ...
                }}
                ...
                ]]
                """
        )
        parser = JsonOutputParser()

        self.chain = prompt | chat | parser

    def generate(self, theme: str):
        return [(x['speaker'], x['phrase']) for x in self.chain.invoke({'theme': theme})]
