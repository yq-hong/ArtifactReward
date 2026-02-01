import torch
from tqdm import tqdm
import ImageReward as RM


class ImageRewardModel:
    def __init__(self, args):
        self.ckpt_path = args.imagereward_ckpt_path

    @property
    def __name__(self):
        return 'ImageRewardModel'

    def load_to_device(self, load_device):
        self.model = RM.load("ImageReward-v1.0")

        self.model = self.model.to(load_device)
        self.model.device = load_device
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        result = []
        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            with torch.no_grad():
                score = self.model.score(prompt, image)
            result.append(score)
        return result
