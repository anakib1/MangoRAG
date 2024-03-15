from typing import List
import random


class BaseThemeProvider:
    def generate(self) -> str:
        pass

    def batch_generate(self, batch_size: int) -> List[str]:
        pass


class InMemoryThemeProvider(BaseThemeProvider):
    def __init__(self, themes: List[str]) -> None:
        self.themes = themes

    def generate(self) -> str:
        return random.choice(self.themes)

    def batch_generate(self, batch_size: int) -> List[str]:
        return random.choices(self.themes, k=batch_size)


class FileThemeProvider(InMemoryThemeProvider):
    def __init__(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            super().__init__([x.strip() for x in f.readlines()])
