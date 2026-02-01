import os.path
import json
import datetime
import argparse
from tqdm import tqdm
import concurrent.futures
import numpy as np
import PIL.Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from predictors import BinaryPredictor


def inference(ex, predictor, prompt):
    # prompt = prompt + "\nDirectly answer YES if there are artifacts or NO if not."
    output = predictor.process_item(ex["id"], prompt, PIL.Image.open(ex["path"]).convert("RGB"))
    return ex, output


def run_evaluate(predictor, prompt, exs, n_worker):
    ids, labels, scores, preds, img_paths = [], [], [], [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_worker) as executor:
        futures = [executor.submit(inference, ex, predictor, prompt) for ex in exs]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='pred...'):
            ex, output = future.result()
            if output != None:
                assert ex['id'] == output["id"]
                score = output["response"]
                img_paths.append(ex['path'])
                labels.append(ex['label'])
                scores.append(score)
                pred = 0 if score < 0.5 else 1
                preds.append(pred)
                ids.append(ex['id'])
            else:
                print(f"No prediction for {ex['id']} {ex['path']}")
                with open(args.out, 'a') as outf:
                    outf.write(f"No prediction for {ex['id']}\t{ex['path']}\n")

    correct_count = sum(1 for a, b in zip(labels, preds) if a == b)
    accuracy = correct_count / len(exs)
    # accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    conf_matrix = confusion_matrix(labels, preds)

    y_true = np.array(labels)
    p1 = np.array(scores)
    eps = 1e-8
    loss = -np.mean(y_true * np.log(p1 + eps) + (1 - y_true) * np.log(1 - p1 + eps))

    return f1, accuracy, conf_matrix, float(loss), 1 - float(np.mean(p1)), img_paths, labels, scores, preds, ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Qwen2.5-VL-7B-Instruct",
                        choices=["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"])
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--result_folder', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--mode', default='test')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--max_threads', default=8, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--exp', default=0, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.out = f"{args.result_folder}/{args.exp}_evaluate/evaluate/exp{args.exp}_prompt{args.prompt_idx}_{args.model}_{args.mode}_{args.out_num}.txt"
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    with open(f"artifacts_curated.json", "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]['id'] = f'{args.mode}-{i}'
        data[i]['label_name'] = {0: "no artifacts", 1: "artifacts"}[data[i]['label']]

    with open(f"{args.result_folder}/{args.exp}_evaluate/{args.exp}_test_attr.json", 'r') as f:
        prompt_all = json.load(f)
    prompt_keys = list(prompt_all.keys())
    pred_prompt = prompt_keys[args.prompt_idx]
    with open(args.out, 'a') as outf:
        outf.write(f'\nprediction_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')

    predictor = BinaryPredictor(configs)
    f1, acc, conf_matrix, loss, mean_score, texts, labels, scores, preds, ids = run_evaluate(predictor, pred_prompt, data, args.max_threads)

    with open(args.out, 'a') as outf:
        outf.write('\n\nNew compute scores: -----------------------')
        outf.write(f'\nAccuracy: {acc}\tF1: {f1}\tLoss: {loss}\tMean score: {mean_score}\n')
        outf.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        outf.write("ID\tPred\tLabel\tProb\tPath\n")
        for i in range(len(labels)):
            if labels[i] != preds[i]:
                outf.write(f'{ids[i]}\t{preds[i]}\t{labels[i]}\t{scores[i]}\t{texts[i]}\n')

    print('DONE!')
