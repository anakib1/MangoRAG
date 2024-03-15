import random

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


@dataclass
class SynthConfig:
    pause_len_min: int = 0.5
    pause_len_max: int = 1.2
    sampling_rate: int = 16000


class SynthDataset(SynthConfig):
    def __init__(self, theme_provider: BaseThemeProvider, dialogue_provider: BaseDialogueProvider,
                 tts_provider: BaseTTSProvider, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theme_provider = theme_provider
        self.dialogue_provider = dialogue_provider
        self.tts_provider = tts_provider

    def _merge_dialogue(self, audio_chunks: List[np.array]) -> np.array:
        """
        Merges several audio chunks into one np array with adding pauses according to configuration
        :param audio_chunks: list of audio chunks to merge
        :return: array of concatenated audio
        """
        audio = []
        for chunk in audio_chunks:
            audio.append(np.zeros(random.randint(int(self.sampling_rate * self.pause_len_min),
                                                 int(self.sampling_rate * self.pause_len_max))))
            audio.append(chunk)

        return np.concatenate(audio)

    def generate(self) -> SynthExample:
        theme = self.theme_provider.generate()
        dialogue = self.dialogue_provider.generate(theme)

        audio = [self.tts_provider.generate(text=utterance, speaker=speaker) for speaker, utterance in dialogue]

        return SynthExample(audio=self._merge_dialogue(audio), theme=theme, dialogue=dialogue)

    def batch_generate(self, batch_size: int, tts_batch_size: int = None) -> List[SynthExample]:
        """
        Generates batch of examples.
        :param batch_size: number of examples to generate
        :param tts_batch_size: batch size for tts model
        :return: list of SynthExample - resulting batch
        """

        if tts_batch_size is None:
            tts_batch_size = batch_size

        themes = self.theme_provider.batch_generate(batch_size)
        dialogues = self.dialogue_provider.batch_generate(themes)

        dialogue_chunks = sum(dialogues, [])
        dialogue_id = []
        for i, dialogue in enumerate(dialogues):
            dialogue_id.extend([i] * len(dialogue))

        audio_chunks = []
        for i in range(0, len(dialogue_chunks), tts_batch_size):
            speakers, phrases = list(zip(*dialogue_chunks[i:i + tts_batch_size]))
            audio_chunks.extend(self.tts_provider.batch_generate(phrases, speakers))

        audio_per_dialogue = []
        for dial_id, audio in zip(dialogue_id, audio_chunks):
            if dial_id == len(audio_per_dialogue):
                audio_per_dialogue.append([])
            audio_per_dialogue[-1].append(audio)

        return [SynthExample(audio=self._merge_dialogue(audio), theme=theme, dialogue=dialogue)
                for audio, theme, dialogue in zip(audio_per_dialogue, themes, dialogues)]
