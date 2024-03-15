from typing import List, Tuple, Any

import torch
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class BaseDialogueProvider:
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        pass

    def batch_generate(self, themes: List[str]) -> List[List[Tuple[str, str]]]:
        return [self.generate(theme) for theme in themes]


class DummyDialogueProvider(BaseDialogueProvider):
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        return [('Alice', 'Hello!, how are you'), ('Bob', 'Hello, I am fine, what about you?'), ('Alice', 'I am sick.')]


class PromptProvider:
    def __init__(self):
        self.prompt_template = "Generate dialogue with two people - Bob and Alice. They should discuss the following " + \
                               "theme: {theme}. The dialogue should contain at least 4 utterances from each speaker. " + \
                               "Please generate the dialogue in the following list json format:" + \
                               """[[
                                   {{"speaker": ... 
                                   "phrase" : ...
                                   }},
                                   {{"speaker": ...
                                   "phrase" : ...
                                   }}
                                   ...
                                   ]]
                                   You must return **ONLY** json format.
                                   """

    def provide_prompt(self, theme: str) -> str:
        return self.prompt_template.format(theme=theme)


class MistralDialogueProvider(BaseDialogueProvider, PromptProvider):
    def __init__(self, token: str, model_id: str) -> None:
        super().__init__()
        self._token = token
        self.client = MistralClient(api_key=token)
        self.model_id = model_id
        self.parser = JsonOutputParser()

    def generate(self, theme: str):
        messages = [
            ChatMessage(role="user", content=self.provide_prompt(theme))
        ]
        chat_response = self.client.chat(
            model=self.model_id,
            messages=messages,
        ).choices[0].message.content
        try:
            dialogue = self.parser.invoke(chat_response)
        except Exception:
            return []

        return [(x['speaker'], x['phrase']) for x in dialogue]


class OpenaiDialogueProvider(BaseDialogueProvider):
    def __init__(self, host: str = "http://localhost:8000/v1"):
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
                You must return **ONLY** json format.
                """
        )
        parser = JsonOutputParser()

        self.chain = prompt | chat | parser

    def generate(self, theme: str):
        try:
            return [(x['speaker'], x['phrase']) for x in self.chain.invoke({'theme': theme})]
        except Exception:
            return []


class HuggingfaceDialogueProvider(BaseDialogueProvider, PromptProvider):
    def __init__(self, model_id: str, hf_token: str, quant_config: Any = None):
        super().__init__()
        login(token=hf_token)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quant_config is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device,
                                                              torch_dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device,
                                                              quantization_config=quant_config)
        self.parser = JsonOutputParser()

    def _capture_json_clean(self, content: str) -> List[Tuple[str, str]] | None:
        try:
            pattern = "```json(.*)```"
            jsons = re.findall(pattern, content, re.DOTALL)
            if len(jsons) == 0:
                return None
            elif len(jsons) > 1:
                return None
            dialogue_dict = self.parser.invoke(jsons[0])
            return [(x['speaker'], x['phrase']) for x in dialogue_dict]
        except Exception as ignored:
            return None

    def _capture_json_dirty(self, content: str) -> List[Tuple[str, str]] | Any:
        pattern = r'(?:\[\]|\[\s*?{.*?}\s*\])'
        jsons = re.findall(pattern, content, re.DOTALL)
        for json_ex in jsons[::-1]:
            try:
                dialogue_dict = self.parser.invoke(json_ex)
                return [(x['speaker'], x['phrase']) for x in dialogue_dict]
            except Exception as ignored:
                continue
        return None

    def _capture_json(self, content: str) -> List[Tuple[str, str]]:
        clean_json = self._capture_json_clean(content)
        if clean_json is not None:
            return clean_json
        dirty_json = self._capture_json_dirty(content)
        if dirty_json is not None:
            return dirty_json
        print(f'WARN: json not found in: {content}')
        return []

    def generate(self, theme: str):
        input_ids = self.tokenizer([self.provide_prompt(theme)], return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, max_length=3000, temperature=0.1, do_sample=True)
        output_str = self.tokenizer.batch_decode(outputs)[0]
        return self._capture_json(output_str)

    def batch_generate(self, themes: List[str]) -> List[List[Tuple[str, str]]]:
        input_ids = self.tokenizer([self.provide_prompt(theme) for theme in themes], padding=True,
                                   return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
        with torch.no_grad():
            outputs = self.model.generate(**input_ids, max_new_tokens=1500, temperature=0.1, do_sample=True)
        output_strs = self.tokenizer.batch_decode(outputs)
        return [self._capture_json(output) for output in output_strs]
