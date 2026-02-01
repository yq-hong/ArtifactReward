import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import argparse
import copy
import random
import inference_utils
from eval_utils import get_data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(42)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--reasoning_prompt_path", type=str, default="data/prompt/reasoning_prompt.txt")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--num_generation", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # specify the path to the model
    model_path = args.model_path
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    data = get_data(args)
    # random.shuffle(data)
    print(f"{len(data)} images to be generated...")

    with open(args.reasoning_prompt_path, 'r') as f:
        cot_prompt = f.read().strip()

    for item in data:
        prompt = item['Prompt']
        prompt_id = item['prompt_id']
        img_path = f"{args.image_dir}/ID{prompt_id}_{item['Prompt'].replace(' ', '_')[:100]}_{args.num_generation}.jpg"
        if os.path.exists(img_path):
            continue
        else:
            print(f'Generating prompt {prompt_id}...')
            prompt_text = copy.deepcopy(prompt)
            conversation = [
                {"role": "<|User|>", "content": cot_prompt.format(prompt)},
                {"role": "<|Assistant|>", "content": ""},
            ]
            system_prompt = 'You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.'

            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=conversation,
                                                                                     sft_format=vl_chat_processor.sft_format,
                                                                                     system_prompt=system_prompt)
            prompt = sft_format

            inference_utils.generate(vl_gpt,
                                     vl_chat_processor,
                                     prompt,
                                     prompt_text,
                                     num_generation=args.num_generation,
                                     conversation=conversation,
                                     save_dir=args.image_dir,
                                     prompt_id=prompt_id)

    print("Done!")


if __name__ == "__main__":
    main()
