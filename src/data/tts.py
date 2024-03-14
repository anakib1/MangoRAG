import random

import numpy as np
from transformers import VitsModel, AutoTokenizer, BarkModel, BarkProcessor
import torch


class BaseTTSProvider:
    def generate(self, text: str, *args, **kwargs) -> np.array:
        pass


class DummyTTSProvider(BaseTTSProvider):
    def generate(self, text: str, *args, **kwargs) -> np.array:
        return np.random.randn(16_000 * 10)


class VitsTTSProvider(BaseTTSProvider):
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    def generate(self, text: str, *args, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform

        return output.flatten().cpu().numpy()


class BarkTTSProvider(BaseTTSProvider):
    def __init__(self, model_id: str = "suno/bark-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BarkModel.from_pretrained(model_id).to(self.device)
        self.processor = BarkProcessor.from_pretrained(model_id)

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
            speech_output = self.model.generate(**inputs).flatten().cpu().numpy()
        return speech_output
