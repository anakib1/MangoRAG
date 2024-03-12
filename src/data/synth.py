import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from .tts import BaseTTSProvider
from .theme import BaseThemeProvider
from .dialogue import BaseDialogueProvider


@dataclass
class SynthExample:
    audio: np.ndarray = None
    theme: str = None
    dialogue: List[Tuple[str, str]] = None


class SynthDataset:
    def __init__(self, theme_provider: BaseThemeProvider, dialogue_provider: BaseDialogueProvider,
                 tts_provider: BaseTTSProvider):
        self.theme_provider = theme_provider
        self.dialogue_provider = dialogue_provider
        self.tts_provider = tts_provider

    def generate(self) -> SynthExample:
        theme = self.theme_provider.generate()
        dialogue = self.dialogue_provider.generate(theme)

        audio = []
        for speaker, utterance in dialogue:
            audio.append(self.tts_provider.generate(text=utterance, speaker=speaker))
            audio.append([0] * 8_000)
        audio.pop()  # remove last silence

        return SynthExample(audio=np.concatenate(audio), theme=theme, dialogue=dialogue)
