import requests
import base64
from io import BytesIO
import re
import json
from abc import ABC
import torch
import math
from tqdm import tqdm
import time
import concurrent.futures
from multiprocessing import Manager, Lock
import openai
from openai import OpenAI, BadRequestError
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

PROMPT = """Analyze the images to identify any unintentional digital artifacts, concentrating on irregular lighting, object placement, blending errors, or anomalies that could affect realism. Disregard any deliberate artistic styles or intentional surreal elements. Respond with YES if artifacts are detected; if not, respond with NO."""


class BinaryPredictor(ABC):
    def __init__(self):
        self.api_key = "EMPTY"
        self.api_url = "http://127.0.0.1:8070/v1"

    def _encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, prompt, image):
        base64_image = self._encode_image(image)

        content = []
        content.append({"type": "text", "text": prompt})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        return [{"role": "user", "content": content}]

    def process_item(self, num, prompt, image):
        max_retries = 5
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            try:
                message = self.build_messages(prompt, image)

                client = OpenAI(api_key=self.api_key, base_url=self.api_url)

                response = client.chat.completions.create(model="Qwen2.5-VL-7B-Instruct",
                                                          messages=message,
                                                          temperature=0.0,
                                                          max_tokens=1,
                                                          logprobs=True,
                                                          top_logprobs=10,
                                                          )

                top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                target_tokens = {"yes", "no"}
                logprob_map = {tl.token.lower(): tl.logprob for tl in top_logprobs if tl.token.lower() in target_tokens}
                yes_lp = logprob_map.get("yes")
                no_lp = logprob_map.get("no")

                if yes_lp is None:
                    prob_yes = 0.0
                elif no_lp is None:
                    prob_yes = 1.0
                else:
                    prob_yes = 1.0 / (1.0 + math.exp(no_lp - yes_lp))

                result = {"id": num, "response": prob_yes}
                return result

            except BadRequestError as be:
                print(f"BadRequestError: {be}")
                continue
            except openai.RateLimitError as e:
                print("Resource Exhausted, wait for a minute to continue...")
                time.sleep(60)
                continue
            except Exception as e:
                print(f"OpenAI server offers this error: {e}")
                if attempt < max_retries:
                    time.sleep(5)  # Wait for 5 seconds before the next attempt
                continue


class ArtifactsVLLMProb:
    def __init__(self, args):
        self.api_key = "EMPTY"
        self.api_url = "http://127.0.0.1:8070/v1"
        self.max_workers = 8

        self.prompt = PROMPT

        self.predictor = BinaryPredictor()

    @property
    def __name__(self):
        return 'ArtifactsVLLMProb'

    def load_to_device(self, load_device):
        self.device = load_device

    def __call__(self, prompts, images, **kwargs):

        results = [None for _ in range(len(prompts))]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.predictor.process_item, i, self.prompt, image) for i, (prompt, image) in enumerate(zip(prompts, images))]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running vllm prob prediction...'):
                output = future.result()
                if output:
                    results[output['id']] = 1 - output["response"]

        return results
