from typing import List, Tuple


class BaseDialogueProvider:
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        pass


class DummyDialogueProvider(BaseDialogueProvider):
    def generate(self, theme: str) -> List[Tuple[str, str]]:
        return [('Alice', 'Hello!, how are you'), ('Bob', 'Hello, I am fine, what about you?'), ('Alice', 'I am sick.')]


