import os
import json
from typing import Dict, Any, List
from datasets import load_dataset


def load_jsonl(path: str) -> Dict[int, Dict]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        print(f"Can't find the file at {path}")
        return {}
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            records[obj["prompt_id"]] = obj
    return records


def load_json(path: str) -> Dict[int, Dict]:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item["prompt_id"]: item for item in data}


def save_results(data: List[Dict], filename: str, output_dir):
    path = os.path.join(output_dir, filename)
    if filename.endswith('.jsonl'):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")


def get_data(args):
    if 'geneval_and_t2i' in args.image_dir:
        print('Evaluating geneval_and_t2i...')

        if 'jsonl' in args.data_path:
            idx = 1
            data = []
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    obj["Prompt"] = obj.pop("prompt")
                    if "tag" in obj:
                        obj["Subcategory"] = obj.pop("tag")
                    elif "task_type" in obj:
                        obj["Subcategory"] = obj.pop("task_type")
                    obj["prompt_id"] = idx
                    idx += 1
        else:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
    elif 'WISE' in args.image_dir and 'WISE' in args.data_path:
        print('Evaluating WISE...')
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif 'LLM4LLM' in args.image_dir:
        print('Evaluating LLM4LLM...')
        idx = 1
        data = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                obj["Prompt"] = obj.pop("prompt")
                obj["Subcategory"] = obj.pop("tag")
                obj["prompt_id"] = idx
                data.append(obj)
                idx += 1
    else:
        data = []
        print("Task not surported!")

    return data
