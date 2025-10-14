import argparse
import json
import os
import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from testing_utils import run_test
import os
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

    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        error = [
            [{"name": "TimeError", "value": "global timeout"} for _ in range(len(in_outs["inputs"]))]
        ]
        if debug:
            print("⚠️ Global timeout reached")

    return result[0], error[0]


# -----------------------------
# Load generation file
# -----------------------------
def load_generation(input_file):
    """Load line-delimited JSONL file of converted generations."""
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

        original_input_output = sample["input_output"]
        input_output = in_out[task_id]

        results[task_id] = {}
        errors[task_id] = {}

        for o_idx, o in enumerate(problem_generations):
            key = json.dumps(input_output[o_idx])
            curr_res = [-2]
            try:
                token_len = tokenizer.tokenize(o)
                if len(token_len) >= 1300:
                    curr_res = [-1]
                    curr_err = [{"name": "Logic Breakdown", "value": "Logic Breakdown"}]
                else:
                    curr_res, curr_err = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)

                if debug:
                    print(f"✓ Evaluated task {task_id}, output #{o_idx}")

                # Clean numpy types
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
            except Exception as e:
                if debug:
                    print(f"❌ Evaluation failed: {repr(e)}")
                curr_err = [{"name": "RuntimeError", "value": str(e)}]
            finally:
                results[task_id][key] = [curr_res]
                errors[task_id][key] = [curr_err]

    return results, errors


# -----------------------------
# Main logic
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code generations.")
    parser.add_argument("--halu_type", type=str, required=True)
    parser.add_argument("--generation_file", type=str, required=True)
    return parser.parse_args()


def main(args):
    generation_file = args.generation_file
    halu_type = args.halu_type
    gen_file_basename = os.path.basename(generation_file)

    generations, samples, in_out = load_generation(generation_file)
    results, errors = evaluate_generations(generations, samples, in_out)

    errors_dict = {}
    total_errors = set()
    new_id = 0
    os.makedirs("evaluated_results", exist_ok=True)

    for task_id, error_map in errors.items():
        for key, err_list in error_map.items():
            err_entry = err_list[0][0]
            result_entry = results[task_id][key][0][0]
            sample = next(s for s in samples if s["task_id"] == task_id)
            input_output = json.loads(key)

            # Determine error type
            if err_entry is None and (result_entry is False or result_entry < 0):
                err_entry = {"name": "Logic_Deviation", "value": "Logic_Deviation"}
            elif err_entry is None and (result_entry is True or result_entry > 0):
                err_entry = {"name": "Correct", "value": "Correct"}
            elif err_entry["name"] in ["TimeError", "TimeoutException"]:
                err_entry = {"name": "Timeout", "value": "Timeout"}

            new_data = {
                "id": new_id,
                "task_id": task_id,
                "prompt": sample["prompt"],
                "input": input_output["inputs"][0],
                "output": input_output["outputs"][0],
                "code": sample["deal_response"],
                "error_type": err_entry,
            }

            with open(f"evaluated_results/{gen_file_basename}_data.json", "a") as file:
                json.dump(new_data, file)
                file.write("\n")
            new_id += 1

            error_name = err_entry["name"]
            error_value = err_entry["value"]
            errors_dict = add_error(errors_dict, error_name, error_value)
            total_errors.add(error_name)

    errors_dict = serialize_errors(errors_dict)
    count = 0
    for _, err_type in programming_halus[halu_type].items():
        count += errors_dict.get(err_type, {}).get("count", 0)

    halu_percentage = round((count / len(samples)) * 100, 2)
    print(f"\n🧠 Halu Type: {halu_type}")
    print(f"🔢 Halu Count: {count}")
    print(f"📊 Total Samples: {len(samples)}")
    print(f"💯 Halu Percentage: {halu_percentage}%")

    with open(f"evaluated_results/{gen_file_basename}_errors_dict.json", "w") as json_file:
        json.dump(errors_dict, json_file, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
