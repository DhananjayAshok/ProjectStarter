from openai import OpenAI
from time import sleep, perf_counter
from typing import Optional, Any
from utils.parameters import load_parameters
from utils.log_handling import log_info, log_warn, log_error
import shutil
from pil import Image
import numpy as np
import base64
import os
from abc import ABC


MIN_QUERIES_PER_MINUTE = 1


class APIModel(ABC):
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        self.parameters = load_parameters(parameters)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_queries_per_minute = max_queries_per_minute
        self.last_query_time = 0
        self.seconds_to_wait = 60 / self.max_queries_per_minute
        # error out if max_queries_per_minute is less than limit
        if self.max_queries_per_minute < MIN_QUERIES_PER_MINUTE:
            log_error(f"max_queries_per_minute must be at least {MIN_QUERIES_PER_MINUTE}, but got {self.max_queries_per_minute}.", parameters=self.parameters)


    def get_encoded_images(self, images: List[Image.Image]) -> List[str]:
        """Encodes images to base64 strings for OpenAI API input.

        :param images: List of images in Pillow Image format.
        :type images: List[Image.Image]
        :return: List of base64 encoded image strings.
        :rtype: List[str]
        """
        cache_dir = self.clear_encoded_image_cache()
        encoded_images = []
        for i, img in enumerate(images):
            img_path = os.path.join(cache_dir, f"image_{i}.jpg")
            img.save(img_path, format="JPEG")
            with open(img_path, "rb") as image_file:
                encoded_images.append(
                    base64.b64encode(image_file.read()).decode("utf-8")
                )
        return encoded_images


    def clear_encoded_image_cache(self) -> str:
        """
        VLM based API models save to a cache for image processing. Clears this cache. 

        :returns: The path to the cache directory for encoded images.
        :rtype: str
        """
        cache_dir = os.path.join(self.parameters["tmp_dir"], "api_image_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        return cache_dir

    def wait(self):
        time_to_wait = self.seconds_to_wait - (
            perf_counter() - self.last_query_time
        )
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.last_query_time = perf_counter()
        return

    def get_output(self, output: list[str]) -> list[str]:
        output_texts = []
        for item in output:
            output_texts.append(item.content[0].text)
        final_texts = []
        for text in output_texts:
            if "[STOP]" in text:
                text = text.split("[STOP]")[0]
            final_texts.append(text.strip())
        return final_texts

    @abstractmethod
    def do_infer(texts: list[str], images: list[list[Image.Image]], max_new_tokens):
        pass




class OpenAIAPIModel(APIModel):
    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        super().__init__(model=model, base_url=base_url, api_key=api_key, max_queries_per_minute=max_queries_per_minute, parameters=parameters)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


class OpenAIModel(OpenAIAPIModel):
    def __init__(self, model: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        super().__init__(model=model, base_url=None, api_key=api_key, max_queries_per_minute=max_queries_per_minute, parameters=parameters)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


class AnthropicModel(APIModel):
    def __init__(self, model: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        super().__init__(model=model, base_url=None, api_key=api_key, max_queries_per_minute=max_queries_per_minute, parameters=parameters)
        self.client = None # TODO


class vLLMModel(OpenAIAPIModel):
    def __init__(self, model: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        base_url = None
        super().__init__(model=model, base_url=base_url, api_key=api_key, max_queries_per_minute=max_queries_per_minute, parameters=parameters)
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)   
        #TODO: MAKE A DEFAULT BASE_URL 


class OpenRouterModel(OpenAIAPIModel):
    def __init__(self, model: str, api_key: Optional[str] = None, max_queries_per_minute: int = 60, parameters: dict[str, Any] = None):
        super().__init__(model=model, base_url="https://openrouter.ai/api/v1", api_key=api_key, max_queries_per_minute=max_queries_per_minute, parameters=parameters)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)