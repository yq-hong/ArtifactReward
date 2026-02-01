import json
import os
import re
import argparse
import PIL.Image
from tqdm import tqdm
import concurrent.futures
from typing import Dict, Any, List
from eval_utils import get_data, load_json, load_jsonl, save_results


def get_eval_prompt(prompt_data):
    eval_prompt = f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**  
- PROMPT: [User's original prompt to]  
- EXPLANATION: [Further explanation of the original prompt] 
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

--- 
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt_data['Prompt']}"
EXPLANATION: "{prompt_data['Explanation']}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""

    return eval_prompt


def extract_scores(txt: str) -> Dict[str, float]:
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
    matches = re.findall(pat, txt, re.IGNORECASE)
    out = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    return out


def evaluate_image(prompt_id, item, img_path):
    from gpt_utils import gpt4o
    try:
        eval_prompt = get_eval_prompt(item)
        output = gpt4o(eval_prompt, img_paths=[img_path], model_version="gpt-4o-2024-05-13")[0]
        scores = extract_scores(output)

        return (
                {  # full record
                    "prompt_id": prompt_id,
                    "prompt": item["Prompt"],
                    "key": item["Explanation"],
                    "image_path": img_path,
                    "evaluation": output
                },
                {  # score record
                    "prompt_id": prompt_id,
                    # "tag": item["tag"],
                    "Subcategory": item["Subcategory"],
                    "consistency": scores.get("consistency", 0),
                    "realism": scores.get("realism", 0),
                    "aesthetic_quality": scores.get("aesthetic_quality", 0)
                }
            )

    except Exception as e:
        print(f"[ERR] {prompt_id}: {e}")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Quality Assessment Tool')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--generate_number', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    data = get_data(args)

    # ---- Resume: load existing results ----
    exist_scores = load_jsonl(os.path.join(args.output_dir, f"WISE_scores_{args.generate_number}.jsonl"))
    exist_full = load_json(os.path.join(args.output_dir, f"WISE_full_{args.generate_number}.json"))
    done_ids = set(exist_scores.keys())

    tasks = []
    for item in data:
        pid = item['prompt_id']
        if pid in done_ids:
            continue
        img_path = f"{args.image_dir}/ID{pid}_{item['Prompt'].replace(' ', '_')[:100]}_{args.generate_number}.jpg"
        try:
            image = PIL.Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARN] Missing image: {img_path}")
            continue
        tasks.append((pid, item, img_path))

    # ---- Multi-threaded evaluation ----
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(evaluate_image, pid, pd, ip) for pid, pd, ip in tasks]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='Evaluating (parallel)'):
        # for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is None:
                continue
            full_rec, score_rec = res
            exist_full[full_rec["prompt_id"]] = full_rec
            exist_scores[score_rec["prompt_id"]] = score_rec

    # ---- Merge, sort, and save ----
    full_sorted = [exist_full[k] for k in sorted(exist_full.keys())]
    score_sorted = [exist_scores[k] for k in sorted(exist_scores.keys())]

    save_results(full_sorted, f"WISE_full_{args.generate_number}.json", args.output_dir)
    save_results(score_sorted, f"WISE_scores_{args.generate_number}.jsonl", args.output_dir)


if __name__ == "__main__":
    main()
