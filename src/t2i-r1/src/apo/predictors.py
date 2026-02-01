from abc import ABC
import base64
from io import BytesIO
import time
from tqdm import tqdm
import concurrent.futures
import math
import numpy as np
from collections import defaultdict
import PIL.Image
import openai
from openai import OpenAI, BadRequestError


class BinaryPredictor(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.categories = ['No', 'Yes']

        self.model_name = opt["model"]
        self.temperature = opt["temperature"]

        if self.model_name == "Qwen2.5-VL-3B-Instruct":
            self.api_key = "EMPTY"
            self.api_url = "http://127.0.0.1:8090/v1"
        elif self.model_name == "Qwen2.5-VL-7B-Instruct":
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

                response = client.chat.completions.create(model=self.model_name,
                                                          messages=message,
                                                          temperature=self.temperature,
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


class Scorer:

    def __init__(self):
        self.cache = defaultdict(dict)

    def inference(self, ex, predictor, prompt):
        # prompt = prompt + "\nDirectly answer YES if there are artifacts or NO if not."
        output = predictor.process_item(ex["id"], prompt, PIL.Image.open(ex["path"]).convert("RGB"))
        return ex, output

    def __call__(self, predictor, prompt, data, max_threads=8):

        def compute_scores(prompts_exs):
            out_scores = {}
            for ex in prompts_exs:
                out_scores[ex["id"]] = 0

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                futures = [executor.submit(self.inference, ex, predictor, prompt) for ex in prompts_exs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                      desc='pred...'):
                    ex, output = future.result()
                    score = output["response"]
                    loss = ex['label'] * np.log(score + 1e-8) + (1 - ex['label']) * np.log(1 - score + 1e-8)
                    out_scores[ex["id"]] = (score, float(loss))

            return out_scores

        prompts_exs_to_compute = []
        for ex in data:  # for ex, prompt in [(ex, prompt) for ex in data]:
            if ex["id"] not in self.cache[prompt]:
                prompts_exs_to_compute.append(ex)

        computed_scores = compute_scores(prompts_exs_to_compute)

        cached_scores = []
        for ex in data:
            if ex["id"] not in self.cache[prompt]:
                self.cache[prompt][ex["id"]] = computed_scores[ex["id"]]
            cached_scores.append(self.cache[prompt][ex["id"]][1])

        return float(np.mean(cached_scores))
