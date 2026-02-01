import sys
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import PIL.Image
from eval_utils import get_data, load_json, save_results

warnings.filterwarnings("ignore")


def get_model(args):
    if args.eval_type == "ORM":
        from utils.reward_orm import ORM
        return ORM(args)
    elif args.eval_type == "HPS":
        from utils.reward_hps import HPSv2
        return HPSv2(args)
    elif args.eval_type == "aesthetics":
        from utils.reward_aesthetics import Aesthetics
        return Aesthetics(args)
    elif args.eval_type == "maniqa":
        from utils.reward_maniqa import MANIQAReward
        args.num_crops = 20
        return MANIQAReward(args)
    elif args.eval_type == "ImageReward":
        from utils.reward_imagereward import ImageRewardModel
        return ImageRewardModel(args)
    elif args.eval_type == "PickScore":
        from utils.reward_pickscore import PickScore
        return PickScore(args)
    elif args.eval_type == "DeQA":
        from utils.reward_deqa import DeQA
        return DeQA(args)
    elif args.eval_type == "artifact_prob_vllm":
        from utils.reward_artifacts import ArtifactsVLLMProb
        return ArtifactsVLLMProb(args)
    elif args.eval_type == "VQAScore":
        from utils.reward_vqa import VQAScore
        return VQAScore(args)
    elif args.eval_type == "Unified_vllm":
        from utils.reward_unified import UnifiedRewardVLLM
        return UnifiedRewardVLLM(args)
    else:
        raise Exception(f'Unsupported task: {args.eval_type}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/GenEval/geneval_and_t2i_data_final.jsonl")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--orm_ckpt_path", type=str, default="reward_weight/ORM-T2I-R1")
    parser.add_argument("--hps_ckpt_path", type=str, default="reward_weight/HPS_v2.1_compressed.pt")
    parser.add_argument("--aesthetics_ckpt_path", type=str, default="reward_weight/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth")
    parser.add_argument("--maniqa_ckpt_path", type=str, default="reward_weight/ckpt_koniq10k.pt")
    parser.add_argument("--imagereward_ckpt_path", type=str, default="ImageReward-v1.0")
    parser.add_argument("--pickscore_ckpt_path", type=str, default="PickScore_v1")
    parser.add_argument("--deqa_ckpt_path", type=str, default="DeQA-Score-Mix3")
    parser.add_argument("--vqa_ckpt_path", type=str, default="clip-flant5-xxl")
    parser.add_argument("--eval_type", type=str, default="ORM",
                        choices=["ORM", "HPS", "aesthetics", "maniqa", "ImageReward", "PickScore", "DeQA", "VQAScore", "artifact_prob_vllm"])
    parser.add_argument('--generate_number', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    filename = f"{args.eval_type}_{args.generate_number}.json"
    exist_full = load_json(os.path.join(args.output_dir, filename))
    done_ids = set(exist_full.keys())

    data = get_data(args)
    pids, prompts, images, img_paths = [], [], [], []
    for obj in data:
        pid = obj["prompt_id"]
        if pid in done_ids:
            continue

        prompt = obj["Prompt"]
        image_path = os.path.join(args.image_dir, f"ID{pid}_{prompt.replace(' ', '_')}_{args.generate_number}.jpg")
        try:
            image = PIL.Image.open(image_path).convert("RGB")
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
        model = get_model(args)
        if "vllm" not in args.eval_type:
            model.load_to_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if args.eval_type == "maniqa":
            scores = model(prompts, img_paths)
        else:
            scores = model(prompts, images)
        assert len(scores) == len(pids) == len(prompts) == len(img_paths)

        for i in range(len(scores)):
            full_rec = {
                "prompt_id": pids[i],
                "prompt": prompts[i],
                # "key": item["Explanation"],
                "image_path": img_paths[i],
                "score": float(scores[i])
            }
            exist_full[full_rec["prompt_id"]] = full_rec

        full_sorted = [exist_full[k] for k in sorted(exist_full.keys())]
        save_results(full_sorted, filename, args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
