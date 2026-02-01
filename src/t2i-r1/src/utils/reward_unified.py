import requests
import base64
from io import BytesIO
import torch
import copy
from tqdm import tqdm
import time
import concurrent.futures
from multiprocessing import Manager, Lock

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


class UnifiedReward:
    def __init__(self, args):

        ckpt_path = args.unified_ckpt_path

        pretrained = ckpt_path
        print(f"pretrained path:{pretrained}")
        model_name = "llava_qwen"
        device_map = "auto"

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(pretrained, None, model_name,
                                                                                    device_map=device_map)
        self.config = self.model.config

        self.model.eval()

    @property
    def __name__(self):
        return 'UnifiedReward'

    def load_to_device(self, load_device):
        self.device = load_device
        self.model.to(self.device)

        # freeze all parameters
        # for n, p in self.model.named_parameters():
        #     p.requires_grad = False
        # self.model.eval()

    def __call__(self, prompts, images, **kwargs):

        # Load the image
        results = []
        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):

            # Process the image
            image_tensor = process_images([image], self.image_processor, self.config)[0]
            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)

            question = (f"{DEFAULT_IMAGE_TOKEN}\nYou are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
                        "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
                        "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
                        "   - A higher score indicates that the word is less well represented in the image.\n"
                        "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
                        "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
                        "- Alignment Score: How well the image matches the caption in terms of content.\n"
                        "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
                        "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
                        "Output your evaluation using the format below:\n\n"
                        "---\n\n"
                        "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
                        "Alignment Score (1-5): X\n"
                        "Coherence Score (1-5): Y\n"
                        "Style Score (1-5): Z\n\n"
                        f"Your task is provided as follows:\nText Caption: [{prompt}]")

            # Prepare conversation
            conv_template = "qwen_1_5"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Input question and image to the model
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(self.device)
            image_size = image.size

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    cont = self.model.generate(input_ids,
                                               images=[image_tensor],
                                               image_sizes=[image_size],
                                               do_sample=False,
                                               temperature=0,
                                               max_new_tokens=4096
                                               )

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

            results.append(text_outputs[0])

        return results


class UnifiedRewardVLLM:
    def __init__(self, args):
        self.api_url = "http://127.0.0.1:8080"
        self.max_workers = 8
        self.session = requests.Session()

    @property
    def __name__(self):
        return 'UnifiedRewardVLLM'

    def _encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, prompt, image):
        base64_image = self._encode_image(image)

        content = []
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        content.append({"type": "text", "text": f"You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"})

        return [{"role": "user", "content": content}]

    def process_item(self, id, prompt, image, total_counter, lock):
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            attempt += 1
            try:
                message = self.build_messages(prompt, image)
                payload = {"model": "UnifiedReward", "messages": message, "temperature": 0, "max_tokens": 4096}

                response = self.session.post(f"{self.api_url}/v1/chat/completions", json=payload, timeout=30 + attempt * 5)
                response.raise_for_status()
                output = response.json()["choices"][0]["message"]["content"]
                with lock:
                    total_counter.value += 1
                result = {"id": id, "response": output}
                return result
            except Exception as e:
                if attempt == max_retries:
                    print(f"No prediction for {id}\t{prompt}")
                    raise (e)
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)
        return None

    def __call__(self, prompts, images, **kwargs):
        with Manager() as manager:
            total_counter = manager.Value('i', 0)
            lock = manager.Lock()

            results = [None for _ in range(len(prompts))]
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i, (prompt, image) in enumerate(zip(prompts, images)):
                    futures.append(executor.submit(self.process_item, i, prompt, image, total_counter, lock))
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures),
                                      desc='running vllm prediction...'):
                    output = future.result()
                    if output:
                        results[output['id']] = output["response"]

        return results
