import random

import numpy as np
import torchaudio.transforms
from transformers import VitsModel, AutoTokenizer, BarkModel, BarkProcessor
import torch
from typing import List
from openai import OpenAI
import torchaudio
import os


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


class SpeakerMapper:
    def __init__(self, speakers_bank: List[str]):
        self.speakers_bank = speakers_bank
        self.speakers_map = {}

    def map(self, speaker: str) -> str:
        if speaker not in self.speakers_map:
            available_speakers = list(set(self.speakers_bank) - set(self.speakers_map.values()))
            if len(available_speakers) == 0:
                print('No available_speakers speaker found. Reusing from bank')
                available_speakers = self.speakers_bank
            self.speakers_map[speaker] = random.choice(available_speakers)

        return self.speakers_map[speaker]


class BarkTTSProvider(BaseTTSProvider):
    speakers = {
        "en": [f'v2/en_speaker_{i}' for i in range(10)],
        "ru": [f'v2/ru_speaker_{i}' for i in range(10)]
    }

    def __init__(self, model_id: str = "suno/bark-small", use_fp16: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BarkModel.from_pretrained(model_id,
                                               torch_dtype=torch.float16 if use_fp16 else None,
                                               device_map=self.device)
        self.processor = BarkProcessor.from_pretrained(model_id)
        self.convert24to16khz = torchaudio.transforms.Resample(24_000, 16_000)

        self.speakers_bank = SpeakerMapper([f'v2/en_speaker_{i}' for i in range(10)])

    def batch_processor(self, texts, voice_presets: list[str]):
        list_dict = [self.processor(text, return_tensors="pt", voice_preset=preset) for preset, text in
                     zip(voice_presets, texts)]
        max_length = max(map(lambda dct: dct["attention_mask"].shape[1], list_dict))
        for dct in list_dict:
            for key in ["attention_mask", "input_ids"]:
                empty = torch.zeros(1, max_length - dct[key].shape[1])
                dct[key] = torch.cat([dct[key], empty], dim=1)
        ans_dict = {
            key: torch.cat([dct[key] for dct in list_dict], dim=0)
            for key in ["attention_mask", "input_ids"]
        }
        return ans_dict

    def generate(self, text: str, speaker: str) -> np.array:
        inputs = self.processor(text, voice_preset=self.speakers_bank.map(speaker))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            speech_output = self.model.generate(**inputs)
        speech_output = self.convert24to16khz(speech_output).flatten().cpu().numpy()
        return speech_output

    def batch_generate(self, texts: list[str], speakers: list[int] = None, lang: str = "en") -> list[np.array]:
        """
        :param texts: phrases to transcribe
        :param lang: supported languages ["ru", "en"]
        :param speakers: voices to use for generation. for "ru" in [0 .. 9], for "en" in [0 .. 9]
        """
        batch_size = len(texts)

        if True or speakers is None:
            speakers_preset = [random.choice(BarkTTSProvider.speakers[lang]) for i in range(batch_size)]
        else:
            speakers_preset = [BarkTTSProvider.speakers[lang][speaker] for speaker in speakers]

        inputs = self.batch_processor(texts, speakers_preset)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            speech_output, lengths = self.model.generate(**inputs, return_output_lengths=True)

        list_speech = []
        for i in range(batch_size):
            audio_array = speech_output[i][:lengths[i]]
            audio_array = self.convert24to16khz(audio_array.cpu()).numpy()
            list_speech.append(audio_array)
        return list_speech


class OpenAITTSProvider(BaseTTSProvider):
    speakers: list[str] = [
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer"
    ]

    def __init__(self, api_token: str, cached_file: str = "./speech.mp3"):
        self.client = OpenAI(
            api_key=api_token
        )
        self.cached_file = cached_file
        self.convert24to16khz = torchaudio.transforms.Resample(24_000, 16_000)
        self.speaker_bank = SpeakerMapper(OpenAITTSProvider.speakers)

    def generate(self, text: str, speaker: str = "alloy") -> np.array:
        speaker = self.speaker_bank.map(speaker)
        if speaker not in OpenAITTSProvider.speakers:
            raise Exception("the speaker is not in the list of valid speakers")
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=speaker,
            input=text
        )
        response.stream_to_file(self.cached_file)
        audio, rate = torchaudio.load(self.cached_file)
        audio = self.convert24to16khz(audio).numpy()
        os.remove(self.cached_file)
        return audio.flatten()
