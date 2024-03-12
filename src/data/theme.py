from typing import List
import random
class BaseThemeProvider:
    def generate(self) -> str:
        pass


class InMemoryThemeProvider(BaseThemeProvider):
    def __init__(self, themes : List[str]) -> None:
        self.themes = themes

    def generate(self) -> str:
        return random.choice(self.themes)

class FileThemeProvider(InMemoryThemeProvider):
    def __init__(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            super().__init__(f.readlines())
        