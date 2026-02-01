import re
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import base64
import time
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import json
import openai
from openai import OpenAI, BadRequestError


class PromptOptimizer(ABC):
    def __init__(self, args, scorer, max_threads=1):
        self.opt = args

        if self.opt['gradient_model'] == "Qwen2.5-VL-3B-Instruct":
            self.api_key = "EMPTY"
            self.api_url = "http://127.0.0.1:8090/v1"
        elif self.opt['gradient_model'] == "Qwen2.5-VL-7B-Instruct":
            self.api_key = "EMPTY"
            self.api_url = "http://127.0.0.1:8070/v1"

        self.scorer = scorer
        self.max_threads = max_threads

    @abstractmethod
    def expand_candidates(self, prompts, train_exs, predictor, scorer):
        pass


class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """

    def build_messages(self, prompt, img_paths=None):
        def encode_image(image_path):
            _, file_extension = os.path.splitext(image_path)
            file_extension = file_extension.lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
                ".svg": "image/svg+xml",
            }
            mime_type = mime_types.get(file_extension)
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_image, mime_type

        if img_paths != None:
            imgs_url = []
            for i in range(len(img_paths)):
                base64_image, mime_type = encode_image(img_paths[i])
                imgs_url.append({"type": "image_url",
                                 "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
            messages = [{"role": "user",
                         "content": [{"type": "text", "text": prompt}] + imgs_url}]
        else:
            messages = [{"role": "user", "content": prompt}]
        return messages

    def get_llm_response(self, prompt, img_paths=None, n=1):
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        messages = self.build_messages(prompt, img_paths)

        num_attempts = 0
        while num_attempts < 5:
            num_attempts += 1
            try:
                response = client.chat.completions.create(model=self.opt['gradient_model'],
                                                          messages=messages,
                                                          n=n,
                                                          temperature=self.opt['gradient_temperature'],
                                                          max_tokens=2048
                                                          )
                num_attempts = 5
                return [response.choices[i].message.content for i in range(n)]

            except BadRequestError as be:
                print(f"BadRequestError: {be}")
                continue
            except openai.RateLimitError as e:
                print("Resource Exhausted, wait for a minute to continue...")
                time.sleep(60)
                continue
            except Exception as e:
                print(f"OpenAI server offers this error: {e}")
                if num_attempts < 5:
                    time.sleep(5)  # Wait for 5 seconds before the next attempt
                continue

    def prepare_samples(self, prompt, minibatch, cache):
        labels, preds, img_paths = [], [], []
        for ex in minibatch:
            score = cache[prompt][ex["id"]][0]
            img_paths.append(ex['path'])
            labels.append(ex['label'])
            pred = 0 if score < 0.5 else 1
            preds.append(pred)
        return img_paths, labels, preds

    def _sample_error_str(self, img_paths, labels, preds, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        # samples a few examples where the model's predictions do not match the true labels
        # then generates a formatted error string that summarizes these mismatches for easier analysis
        label_names = {0: "no artifacts", 1: "artifacts"}
        error_idxs = []
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_img_paths = [img_paths[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]

        error_string = ''
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_img_paths, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx + 1}:\tLabel: {label_names[l]}\tPrediction: {label_names[p]}\n'
            error_idx += 1
        return error_string.strip(), sample_img_paths

    def _get_gradients(self, prompt, error_string, img_paths, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt for detecting whether a given image shows artifacts that don't look realistic.

        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        Give a reasons why the prompt could have gotten these examples wrong.
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])

        res = self.get_llm_response(gradient_prompt, img_paths, n=n)

        with open(self.opt['out'], 'a') as outf:
            outf.write('feedbacks: ' + json.dumps(res) + '\n')
        return res

    def apply_gradient(self, prompt, feedback_str, error_string, img_paths, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier prompt for detecting whether a given image shows artifacts that don't look realistic.
        
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, directly write the improved prompt without any extra words:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])

        new_prompts = self.get_llm_response(transformation_prompt, img_paths, n=n)

        with open(self.opt['out'], 'a') as outf:
            for p in new_prompts:
                outf.write('new prompts: ' + json.dumps(p) + '\n')
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate one variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:\n\n"

        new_instructions = self.get_llm_response(rewriter_prompt, n=n)

        for x in new_instructions:
            with open(self.opt['out'], 'a') as outf:
                outf.write('synonyms: ' + json.dumps(x) + '\n')
        return new_instructions

    def get_gradients(self, prompt, img_paths, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string, sample_img_paths = self._sample_error_str(img_paths, labels, preds,
                                                                    n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(prompt, error_string, sample_img_paths, n=1)
            prompt_feedbacks += [(t, error_string, sample_img_paths) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, train_exs, predictor, scorer):
        """ Expand a list of prompts by generating gradient-based successors and
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            img_paths, labels, preds = self.prepare_samples(prompt, minibatch, scorer.cache)

            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, img_paths, labels, preds)
                new_task_sections = []
                for feedback, error_string, sample_img_paths in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(prompt, feedback, error_string, sample_img_paths,
                                              self.opt['steps_per_gradient'])
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [prompt], desc='mc samples'):
                    mc_sects = self.generate_synonyms(sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections))  # dedup

            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                new_sections = random.sample(new_sections, k=self.opt['max_expansion_factor'])

            new_prompts += new_sections

        new_prompts += prompts  # add originals
        new_prompts = list(set(new_prompts))  # dedup

        return new_prompts
