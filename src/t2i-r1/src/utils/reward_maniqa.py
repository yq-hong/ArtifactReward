import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "MANIQA")))
import torch
import cv2
from torchvision import transforms
import numpy as np
from MANIQA.models.maniqa import MANIQA
from MANIQA.config import Config
from MANIQA.utils.inference_process import ToTensor, Normalize
from tqdm import tqdm


class ImagePatches(torch.utils.data.Dataset):
    """Create random patches from a single image for MANIQA scoring."""

    def __init__(self, image_path, transform=None, num_crops=20, crop_size=224):
        super().__init__()
        self.img_name = image_path.split('/')[-1]

        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        c, h, w = self.img.shape
        self.transform = transform

        # Generate patches
        self.img_patches = []
        for _ in range(num_crops):
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)
            patch = self.img[:, top: top + crop_size, left: left + crop_size]
            self.img_patches.append(patch)
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MANIQAReward:
    def __init__(self, args):

        self.ckpt_path = args.maniqa_ckpt_path
        self.num_crops = args.num_crops

        self.config = Config({
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,
        })

        self.model = MANIQA(patch_size=self.config.patch_size, img_size=self.config.img_size,
                            embed_dim=self.config.embed_dim, dim_mlp=self.config.dim_mlp,
                            num_heads=self.config.num_heads, window_size=self.config.window_size,
                            depths=self.config.depths, num_outputs=self.config.num_outputs,
                            num_tab=self.config.num_tab, scale=self.config.scale)

        self.model.eval()

    @property
    def __name__(self):
        return 'MANIQAReward'

    def load_to_device(self, load_device):
        self.device = load_device

        self.model.load_state_dict(torch.load(self.ckpt_path), strict=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, prompts, image_paths, **kwargs):

        results = []
        for prompt, path in tqdm(zip(prompts, image_paths), total=len(prompts)):

            Img = ImagePatches(image_path=path,
                               transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
                               num_crops=self.num_crops
                               )

            sum_score = 0
            for i in range(self.num_crops):
                with torch.no_grad():
                    patch_sample = Img.get_patch(i)
                    patch = patch_sample['d_img_org'].to(self.device)
                    patch = patch.unsqueeze(0)
                    score = self.model(patch)
                    sum_score += score

            results.append(sum_score / self.num_crops)

        return results
