import numpy as np

class BaseTTS:
    def generate(self, text:str, *args, **kwargs) -> np.array:
        pass

class DummyTTS(BaseTTS):
    def generate(self, text:str, *args, **kwargs) -> np.array:
        return np.random.randn(16_000 * 10)