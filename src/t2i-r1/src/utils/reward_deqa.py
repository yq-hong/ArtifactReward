import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


class DeQA:
    def __init__(self, args):
        self.ckpt_path = args.deqa_ckpt_path

    @property
    def __name__(self):
        return 'DeQA'

    def load_to_device(self, load_device):
        self.model = AutoModelForCausalLM.from_pretrained(self.ckpt_path,
                                                          trust_remote_code=True,
                                                          attn_implementation="eager",
                                                          torch_dtype=torch.float16,
                                                          device_map="auto",
                                                          )

        self.model = self.model.to(load_device)
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        # image_list is a list of PIL image
        result = []
        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            with torch.no_grad():
                score = self.model.score([image])[0]
            result.append(score.item())
        return result
