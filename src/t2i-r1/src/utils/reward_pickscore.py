from transformers import AutoProcessor, AutoModel, CLIPModel, AutoConfig
from tqdm import tqdm
import torch


class PickScore:
    def __init__(self, args):
        self.ckpt_path = args.pickscore_ckpt_path
        self.processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

        self.processor = AutoProcessor.from_pretrained(self.processor_name)
        self.model = AutoModel.from_pretrained(self.ckpt_path)

    @property
    def __name__(self):
        return 'PickScore'

    def load_to_device(self, load_device):
        self.model = self.model.to(load_device)
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        result = []
        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            image_inputs = self.processor(images=image,
                                          padding=True,
                                          truncation=True,
                                          max_length=77,
                                          return_tensors="pt").to(device)
            text_inputs = self.processor(text=prompt,
                                         padding=True,
                                         truncation=True,
                                         max_length=77,
                                         return_tensors="pt").to(device)

            with torch.no_grad():
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = self.model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                # scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
                # probs = torch.softmax(scores, dim=-1).cpu().tolist()
                scores = (text_embs @ image_embs.T)[0]

            result.append(scores.item())
        return result
