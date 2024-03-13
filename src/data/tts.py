import numpy as np
from transformers import VitsModel, AutoTokenizer
import torch


class BaseTTSProvider:
    def generate(self, text: str, *args, **kwargs) -> np.array:
        pass


class DummyTTSProvider(BaseTTSProvider):
    def generate(self, text: str, *args, **kwargs) -> np.array:
        return np.random.randn(16_000 * 10)


class LocalTTSProvider(BaseTTSProvider):
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    def generate(self, text: str, *args, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform

        return output.flatten().cpu().numpy()
