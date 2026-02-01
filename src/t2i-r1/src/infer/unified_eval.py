import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import argparse
from typing import Dict
import PIL.Image
import warnings
from utils.reward_unified import UnifiedReward, UnifiedRewardVLLM, UnifiedRewardVLLM2
from eval_utils import get_data, load_json, save_results, load_jsonl

warnings.filterwarnings("ignore")


def extract_scores(txt: str) -> Dict[str, float]:
    # Pattern to match Alignment, Coherence, and Style scores
    pat = r"(Alignment|Coherence|Style) Score \(1-5\):\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pat, txt)

    out = {}
    for k, v in matches:
        # convert to lowercase with underscores
        out[k.lower().replace(" ", "_")] = float(v)
    return out


def extract_scores_vllm(txt):
    # Pattern to match Alignment, Coherence, and Style scores
    pat = r"Final Score:\s*([0-9]*\.?[0-9]+)"
    match = re.findall(pat, txt)

    if match:
        return float(match[0])
    else:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--unified_ckpt_path", type=str, default="UnifiedReward-7b-v1.5")
    parser.add_argument('--generate_number', type=int, default=1)
    parser.add_argument('--mode', type=str, default="vllm", choices=["vllm", "hf"])
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "hf":
        name = "Unified"
    else:
        name = f"Unified_{args.mode}"

    exist_scores = load_jsonl(os.path.join(args.output_dir, f"{name}_scores_{args.generate_number}.jsonl"))
    exist_full = load_json(os.path.join(args.output_dir, f"{name}_full_{args.generate_number}.json"))
    done_ids = set(exist_full.keys())

    data = get_data(args)
    pids, prompts, images, img_paths = [], [], [], []
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

    print(f"{len(pids)} images to be evaluated...")
    assert len(pids) == len(prompts) == len(img_paths) == len(images)

    if len(pids) > 0:
        if args.mode == "vllm":
            unified_model = UnifiedRewardVLLM(args)
            outputs = unified_model(prompts, images)
            for i in range(len(outputs)):
                if outputs[i] != None:
                    score = extract_scores_vllm(outputs[i])
                    if score != None:
                        full_rec = {
                            "prompt_id": pids[i],
                            "prompt": prompts[i],
                            "image_path": img_paths[i],
                            "score": outputs[i]
                        }
                        score_rec = {
                            "prompt_id": pids[i],
                            "score": score
                        }
                        exist_full[full_rec["prompt_id"]] = full_rec
                        exist_scores[score_rec["prompt_id"]] = score_rec
        else:
            unified_model = UnifiedReward(args)
            unified_model.load_to_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            outputs = unified_model(prompts, images)
            assert len(outputs) == len(pids) == len(prompts) == len(img_paths)

            for i in range(len(outputs)):
                score = extract_scores(outputs[i])

                full_rec = {
                    "prompt_id": pids[i],
                    "prompt": prompts[i],
                    # "key": item["Explanation"],
                    "image_path": img_paths[i],
                    "score": outputs[i]
                }
                score_rec = {
                    "prompt_id": pids[i],
                    "alignment": score.get("alignment", 0),
                    "coherence": score.get("coherence", 0),
                    "style": score.get("style", 0)
                }
                exist_full[full_rec["prompt_id"]] = full_rec
                exist_scores[score_rec["prompt_id"]] = score_rec

        full_sorted = [exist_full[k] for k in sorted(exist_full.keys())]
        score_sorted = [exist_scores[k] for k in sorted(exist_scores.keys())]

        save_results(full_sorted, f"{name}_full_{args.generate_number}.json", args.output_dir)
        save_results(score_sorted, f"{name}_scores_{args.generate_number}.jsonl", args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
