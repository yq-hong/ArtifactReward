import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import datetime
import time
import json
import random
import argparse
from predictors import BinaryPredictor, Scorer
import optimizers

random.seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="Qwen2.5-VL-7B-Instruct",
                        choices=["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"])
    parser.add_argument('--gradient_model', default="Qwen2.5-VL-3B-Instruct",
                        choices=["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"])
    parser.add_argument('--result_folder', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--exp', default='0')
    parser.add_argument('--max_threads', default=8, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--gradient_temperature', default=0.7, type=float)

    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)

    parser.add_argument('--minibatch_size', default=60, type=int, help='# total instances per minibatch')
    parser.add_argument('--n_gradients', default=4, type=int, help='# generated gradients per prompt')
    parser.add_argument('--errors_per_gradient', default=4, type=int,
                        help='# error examples used to generate one gradient')
    parser.add_argument('--gradients_per_error', default=1, type=int, help='# gradient reasons per error')
    parser.add_argument('--steps_per_gradient', default=1, type=int, help='# new prompts per gradient reason')
    parser.add_argument('--mc_samples_per_step', default=1, type=int, help='# synonyms')
    parser.add_argument('--max_expansion_factor', default=5, type=int, help='maximum # prompts after expansion')

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.out = f"{args.result_folder}/{args.exp}_evaluate/apo_log_{args.exp}.txt"
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    with open("artifacts_curated.json", "r") as f:
        data_train = json.load(f)
    for i in range(len(data_train)):
        data_train[i]['id'] = f'train-{i}'
        data_train[i]['label_name'] = {0: "no artifacts", 1: "artifacts"}[data_train[i]['label']]

    with open("artifacts_curated_test.json", "r") as f:
        data_test = json.load(f)
    for i in range(len(data_test)):
        data_test[i]['id'] = f'test-{i}'
        data_test[i]['label_name'] = {0: "no artifacts", 1: "artifacts"}[data_test[i]['label']]

    with open(f"{args.result_folder}/0_evaluate/0_prompts.json", 'r') as f:
        prompt_all = json.load(f)
    pred_prompt = list(prompt_all[0].keys())[0]
    with open(args.out, 'a') as outf:
        outf.write(f'\nprediction_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')
    candidates = [pred_prompt]

    predictor = BinaryPredictor(configs)
    train_scorer = Scorer()
    test_scorer = Scorer()
    train_score0 = train_scorer(predictor, pred_prompt, data_train, args.max_threads)

    optimizer = optimizers.ProTeGi(configs, train_scorer, args.max_threads)

    for round in tqdm(range(configs['rounds'] + 1)):
        print("STARTING ROUND ", round)
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
        start = time.time()

        if round > 0:
            candidates = optimizer.expand_candidates(candidates, data_train, predictor, train_scorer)

            scores = []
            for cand in candidates:
                cand_score = train_scorer(predictor, cand, data_train, args.max_threads)
                scores.append(cand_score)
            [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

            candidates = candidates[:configs['beam_size']]
            scores = scores[:configs['beam_size']]
        else:
            scores = [train_score0]

        with open(args.out, 'a') as outf:
            outf.write(f'{time.time() - start}\n')
            for c in candidates:
                outf.write(json.dumps(c) + '\n')
            outf.write(f'{scores}\n')

        metrics = []
        for candidate, score in zip(candidates, scores):
            cand_score = test_scorer(predictor, candidate, data_test, args.max_threads)
            metrics.append(cand_score)
        with open(args.out, 'a') as outf:
            outf.write(f'{metrics}\n')

        with open(f'{args.exp}_train_attr.json', 'w') as json_file:
            json.dump(train_scorer.cache, json_file)
        with open(f'{args.exp}_test_attr.json', 'w') as json_file:
            json.dump(test_scorer.cache, json_file)

    print("DONE!")
