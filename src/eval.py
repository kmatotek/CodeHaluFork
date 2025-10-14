import argparse
import json
import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from testing_utils import run_test

os.environ["TOKENIZERS_PARALLELISM"] = "false"
TIMEOUT = 30

# -----------------------------
# Hallucination categories
# -----------------------------
programming_halus = {
    "Data_Compliance_Hallucination": {
        "TypeError": "TypeError",
        "ValueError": "ValueError",
        "ZeroDivisionError": "ZeroDivisionError",
    },
    "Structural_Access_Hallucination": {
        "IndexError": "IndexError",
        "KeyError": "KeyError",
    },
    "Identification_Hallucination": {
        "NameError": "NameError",
        "AttributeError": "AttributeError",
        "UnboundLocalError": "UnboundLocalError",
    },
    "External_Source_Hallucination": {
        "ImportError": "ImportError",
        "ModuleNotFoundError": "ModuleNotFoundError",
    },
    "Physical_Constraint_Hallucination": {
        "RecursionError": "RecursionError",
        "MemoryError": "MemoryError",
    },
    "Calculate_Boundary_Hallucination": {
        "OverflowError": "OverflowError",
        "StopIteration": "StopIteration",
    },
    "Logic_Deviation": {
        "Logic_Deviation": "Logic_Deviation",
    },
    "Logic_Breakdown": {
        "Logic_Breakdown": "Logic_Breakdown",
    },
}

# -----------------------------
# Utility helpers
# -----------------------------
def serialize_errors(errors_dict):
    serialized_errors = {}
    for error_name, (error_values, count) in errors_dict.items():
        serialized_errors[error_name] = {
            "values": list(error_values),
            "count": count,
        }
    return serialized_errors


def add_error(errors_dict, error_name, error_value):
    if error_name not in errors_dict:
        errors_dict[error_name] = (set(), 0)
    errors, count = errors_dict[error_name]
    errors.add(error_value)
    errors_dict[error_name] = (errors, count + 1)
    return errors_dict


# -----------------------------
# Flatten helpers
# -----------------------------
def flatten_error(err_list):
    if not err_list or len(err_list) == 0:
        return {"name": "EvaluationError", "value": "No error info"}
    first = err_list[0]
    if isinstance(first, list) and len(first) > 0:
        return first[0]
    elif isinstance(first, dict):
        return first
    else:
        return {"name": "EvaluationError", "value": str(first)}


def flatten_result(res_list):
    if not res_list or len(res_list) == 0:
        return -1
    first = res_list[0]
    if isinstance(first, list) and len(first) > 0:
        return first[0]
    return first


# -----------------------------
# Evaluation function
# -----------------------------
def check_correctness(sample, generation, timeout, debug=True):
    """Safely run a generated program against test cases with a timeout."""
    def _temp_run(sample, generation, debug, result, error):
        res, err = run_test(sample, test=generation, debug=debug)
        result.append(res)
        error.append(err)

    manager = multiprocessing.Manager()
    result = manager.list()
    error = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(sample, generation, debug, result, error)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    # Normalize empty or malformed results
    if not result or len(result) == 0 or result[0] is None:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        error = [
            [{"name": "TimeError", "value": "global timeout"} for _ in range(len(in_outs["inputs"]))]
        ]

    return result[0], error[0]


# -----------------------------
# Load generation file
# -----------------------------
def load_generation(input_file):
    generations = {}
    in_out = {}
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            res = json.loads(line)
            task_id = res["task_id"]
            output = res["deal_response"]
            input_output = json.loads(res["input_output"])
            generations.setdefault(task_id, []).append(output)
            in_out.setdefault(task_id, []).append(input_output)
            data.append(res)

    return generations, data, in_out


# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate_generations(generations, samples, in_out, debug=False):
    results = {}
    errors = {}
    tokenizer = AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf", use_fast=True, trust_remote_code=True
    )

    for task_id, problem_generations in tqdm(generations.items(), desc="Evaluating"):
        sample = next((s for s in samples if s["task_id"] == task_id), None)
        if not sample:
            continue

        input_output = in_out[task_id]

        results[task_id] = {}
        errors[task_id] = {}

        for o_idx, o in enumerate(problem_generations):
            key = json.dumps(input_output[o_idx])
            curr_res = [-2]
            curr_err = [{"name": "EvaluationError", "value": "Unknown"}]

            try:
                token_len = tokenizer.tokenize(o)
                if len(token_len) >= 1300:
                    curr_res = [-1]
                    curr_err = [{"name": "Logic Breakdown", "value": "Logic Breakdown"}]
                else:
                    curr_res, curr_err = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)

                # Fix numpy types
                fixed_res = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed_res.append(e)
                curr_res = fixed_res

            except Exception as e:
                if debug:
                    print(f"❌ Evaluation failed: {repr(e)}")
                curr_err = [{"name": "RuntimeError", "value": str(e)}]

            finally:
                # Normalize always
                results[task_id][key] = [curr_res] if isinstance(curr_res, list) else [[curr_res]]
                errors[task_id][key] = [curr_err] if isinstance(curr_err, list) else [[curr_err]]

    return results, errors


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code generations.")
    #parser.add_argument("--halu_type", type=str, required=True)
    parser.add_argument("--generation_file", type=str, required=True)
    return parser.parse_args()


def main(args):
    generation_file = args.generation_file
    #halu_type = args.halu_type
    gen_file_basename = os.path.basename(generation_file)

    generations, samples, in_out = load_generation(generation_file)
    results, errors = evaluate_generations(generations, samples, in_out)

    errors_dict = {}
    total_errors = set()
    new_id = 0
    os.makedirs("evaluated_results", exist_ok=True)
    missing_tasks = []

    for sample in samples:
        task_id = sample["task_id"]
        input_outputs = in_out[task_id]

        for key_idx, key_json in enumerate(input_outputs):
            key = json.dumps(key_json)
            err_list = errors.get(task_id, {}).get(key, [])
            res_list = results.get(task_id, {}).get(key, [])

            if not err_list:
                missing_tasks.append(task_id)

            err_entry = flatten_error(err_list)
            res_entry = flatten_result(res_list)

            # Handle correct vs error
            if err_entry is None and (res_entry is True or res_entry > 0):
                err_entry = {"name": "Correct", "value": "Correct"}
            elif err_entry is None:
                err_entry = {"name": "EvaluationError", "value": "Unknown"}

            new_data = {
                "id": new_id,
                "task_id": task_id,
                "prompt": sample["prompt"],
                "input": key_json["inputs"][0] if key_json.get("inputs") and len(key_json["inputs"]) > 0 else "NO_INPUT",
                "output": key_json["outputs"][0] if key_json.get("outputs") and len(key_json["outputs"]) > 0 else "NO_OUTPUT",
                "code": sample.get("deal_response", ""),
                "error_type": err_entry,
            }


            with open(f"evaluated_results/{gen_file_basename}_data.json", "a") as file:
                json.dump(new_data, file)
                file.write("\n")

            error_name = err_entry["name"]
            error_value = err_entry["value"]
            errors_dict = add_error(errors_dict, error_name, error_value)
            total_errors.add(error_name)
            new_id += 1

    # Save missing tasks if any
    if missing_tasks:
        with open(f"evaluated_results/{gen_file_basename}_missing_tasks.json", "w") as f:
            json.dump(list(set(missing_tasks)), f)

    errors_dict = serialize_errors(errors_dict)
    
    print(f"\n📊 Total Samples: {len(samples)}")
    print("\nTop Error Types:")
    for err, stats in sorted(errors_dict.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
        print(f"  {err}: {stats['count']}")



    with open(f"evaluated_results/{gen_file_basename}_errors_dict.json", "w") as json_file:
        json.dump(errors_dict, json_file, indent=4)

    print(f"\n✅ Evaluation complete. Data saved → evaluated_results/{gen_file_basename}_data.json")
    if missing_tasks:
        print(f"⚠️ Missing tasks saved → evaluated_results/{gen_file_basename}_missing_tasks.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
