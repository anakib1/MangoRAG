import random

import numpy as np
import torchaudio.transforms
from transformers import VitsModel, AutoTokenizer, BarkModel, BarkProcessor
import torch
from typing import List


class BaseTTSProvider:
    def generate(self, text: str, speaker: str) -> np.array:
        """
        Generates audio corresponding to given speaker and text.
        :param text: text to voice
        :param speaker:speaker identity
        :return: np array of audio in 16_000 sampling rate
        """
        pass

    def batch_generate(self, texts: List[str], speakers: List[str]) -> List[np.array]:
        """
        Generates several audio corresponding to given speakers.
        :param texts: list of texts to generate
        :param speakers: speaker identities
        :return: List of np arrays with audio in 16_000 sampling rate
        """
        return [self.generate(text, speaker) for text, speaker in zip(texts, speakers)]


class DummyTTSProvider(BaseTTSProvider):
    def generate(self, text: str, speaker: str) -> np.array:
        return np.random.randn(16_000 * 10)


class VitsTTSProvider(BaseTTSProvider):
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    def generate(self, text: str, speaker: str):
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform

        return output.flatten().cpu().numpy()


class BarkTTSProvider(BaseTTSProvider):
    def __init__(self, model_id: str = "suno/bark-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BarkModel.from_pretrained(model_id).to(self.device)
        self.processor = BarkProcessor.from_pretrained(model_id)
        self.convert24to16khz = torchaudio.transforms.Resample(24_000, 16_000)

        self.speakers_bank = [f'v2/en_speaker_{i}' for i in range(10)]
        self.speakers_map = {}

    def generate(self, text: str, speaker: str, *args, **kwargs) -> np.array:
        if speaker not in self.speakers_map:
            available_speakers = list(set(self.speakers_bank) - set(self.speakers_map.values()))
            if len(available_speakers) == 0:
                print('No available_speakers speaker found. Reusing from bank')
                available_speakers = self.speakers_bank
            self.speakers_map[speaker] = random.choice(available_speakers)

        speaker_preset = self.speakers_map[speaker]
        inputs = self.processor(text, voice_preset=speaker_preset)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            speech_output = self.model.generate(**inputs)
        speech_output = self.convert24to16khz(speech_output).flatten().cpu().numpy()
        return speech_output
