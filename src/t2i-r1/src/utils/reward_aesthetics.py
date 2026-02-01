import torch
import torch.nn as nn
import open_clip


class Aesthetics:
    def __init__(self, args):
        self.ckpt_path = args.aesthetics_ckpt_path

    @property
    def __name__(self):
        return 'aesthetics'

    def load_to_device(self, load_device):
        if "vit_l_14" in self.ckpt_path:
            self.m_head = nn.Linear(768, 1)
        else:
            raise ValueError(f"Unsupported clip_model: {self.ckpt_path}")

        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        self.m_head.load_state_dict(state_dict)
        for param in self.m_head.parameters():
            param.requires_grad = False
        self.m_head = self.m_head.to(load_device)
        self.m_head.eval()

        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='openai',
            device=load_device
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.model = self.model.to(load_device)
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        result = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            with torch.no_grad():
                image = self.preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda'):
                    image_features = self.model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    aesthetic_score = self.m_head(image_features)
            result.append(aesthetic_score.item())
        return result
