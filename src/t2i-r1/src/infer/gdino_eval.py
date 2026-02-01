import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import argparse
import PIL.Image
from utils.reward_gdino import GDino
from eval_utils import get_data, load_json, save_results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--gdino_ckpt_path", type=str, default="reward_weight/groundingdino_swint_ogc.pth")
    parser.add_argument("--gdino_config_path", type=str, default="src/t2i-r1/src/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--generate_number', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    exist_full = load_json(os.path.join(args.output_dir, f"GDino_{args.generate_number}.json"))
    done_ids = set(exist_full.keys())

    data = get_data(args)
    pids, prompts, images, img_paths, nouns, task_types, spatial_info, numeracy_info = [], [], [], [], [], [], [], []
    for obj in data:
        pid = obj["prompt_id"]
        if pid in done_ids:
            continue

        prompt = obj["Prompt"]
        image_path = f"{args.image_dir}/ID{pid}_{prompt.replace(' ', '_')}_{args.generate_number}.jpg"
        try:
            image = PIL.Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            try:
                alt_image_path = f"{args.image_dir}/ID{pid}_{prompt.replace(' ', '_')[:100]}_{args.generate_number}.jpg"
                image = PIL.Image.open(alt_image_path).convert("RGB")
                image_path = alt_image_path
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                continue

        pids.append(pid)
        prompts.append(prompt)
        images.append(image)
        img_paths.append(image_path)
        task_types.append(obj["Subcategory"])
        if "nouns" in obj:
            nouns.append(obj["nouns"])
        else:
            nouns.append(None)
        if obj["Subcategory"] == "spatial":
            spatial_info.append(obj["spatial_info"])
        else:
            spatial_info.append(None)
        if obj["Subcategory"] == "numeracy":
            numeracy_info.append(obj["numeracy_info"])
        else:
            numeracy_info.append(None)

    print(f"{len(pids)} images to be evaluated...")
    assert len(pids) == len(prompts) == len(task_types) == len(nouns) == len(spatial_info) == len(numeracy_info)

    gdino_model = GDino(args)
    gdino_model.load_to_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    det_prompt = []
    for i in range(len(nouns)):
        text_prompt, token_spans = gdino_model.make_prompt(nouns[i])
        det_prompt.append({'text_prompt': text_prompt, 'token_spans': str(token_spans)})

    kwargs = {
        "task_type": task_types,
        "nouns": nouns,
        "spatial_info": spatial_info,
        "numeracy_info": numeracy_info,
        "det_prompt": det_prompt
    }
    scores = gdino_model(prompts, images, **kwargs)
    assert len(scores) == len(pids) == len(prompts) == len(img_paths)

    for i in range(len(scores)):
        full_rec = {
            "prompt_id": pids[i],
            "prompt": prompts[i],
            # "key": item["Explanation"],
            "image_path": img_paths[i],
            "score": scores[i]
        }
        exist_full[full_rec["prompt_id"]] = full_rec

    full_sorted = [exist_full[k] for k in sorted(exist_full.keys())]
    save_results(full_sorted, f"GDino_{args.generate_number}.json", args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
