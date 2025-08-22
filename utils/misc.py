import os
import sys
import json
import random
import torch
import re
import concurrent.futures
import time
import pebble
import dill
import numpy as np

TORCH_DTYPE = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def save_llm_logs(messages, log_dir, log_name="llm_logs.json"):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, log_name), "w") as fh:
        fh.write(json.dumps({
            # "n_calls": len(messages), # TODO
            "messages": messages
        }, indent=2))

    cost = aggregate_cost_from_logs(messages)
    if cost is not None:
        print(f"Total cost: {cost:.2f}")

    print(f"LLM logs saved to {os.path.join(log_dir, log_name)}")


def sample_by_strategy(n_cands, candidates, strategy="random"):
    _n_cands = min(n_cands, len(candidates))
    if strategy == "random":
        return random.sample(candidates, _n_cands)
    elif strategy == "top":
        return candidates[:_n_cands]
    elif strategy == "bottom":
        return candidates[-_n_cands:]
    elif strategy == "diverse":
        # Randomly sample equally from the top, middle, and bottom
        top = random.sample(candidates[:len(candidates) // 3], _n_cands // 3)
        middle = random.sample(candidates[len(candidates) // 3: 2 * len(candidates) // 3], _n_cands // 3)
        bottom = random.sample(candidates[2 * len(candidates) // 3:], _n_cands - len(top) - len(middle))
        return top + middle + bottom
    else:
        raise NotImplementedError


def strip_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            strip_dict(v)
        elif isinstance(v, str):
            d[k] = v.strip()
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, str):
                    v[i] = x.strip()
                elif isinstance(x, dict):
                    strip_dict(x)


def extract_json(s):
    depth = 0
    in_string = False
    escape = False
    start_idx = None

    for i, char in enumerate(s):
        if char == '"' and not escape:
            in_string = not in_string

        if not in_string:
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    json_str = s[start_idx:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        if char == '\\' and not escape:
            escape = True
        else:
            escape = False
    return None


def make_set(tuples, index=0):
    seen = set()
    result = []
    for tup in tuples:
        _tup = tup
        if type(tup) not in [tuple, list]:
            _tup = (tup,)
        key = _tup[index]  # Use the value at the specified index as the key
        if key not in seen:
            seen.add(key)
            result.append(tup)
    return set(result)


def stringify_grid(grid, pretty=True, add_shape=True, transpose=False):
    if transpose:
        grid = np.array(grid).T.tolist()
    if pretty:
        return "[" + "\n ".join(str(row) for row in grid) + "]" + \
               (f"\n(shape: {len(grid)}x{len(grid[0])})" if add_shape else "")
    else:
        return str(grid) + (f" (shape: {len(grid)}x{len(grid[0])})" if add_shape else "")


def docstring_to_string(input_string):
    # Regular expression to find triple-quoted multiline strings
    pattern = re.compile(r'(""")(.*?)(\1)', re.DOTALL)

    # Function to replace newlines within the match with escaped newlines
    def replace_newlines(match):
        multiline_str = match.group(2)
        single_line_str = multiline_str.replace('\n', '\\n')
        return f'"{single_line_str}"'

    # Replace all found multiline strings with single line equivalents
    result = pattern.sub(replace_newlines, input_string)
    return result


def _run_serialized_dill_func(serialized_func, arg):
    func = dill.loads(serialized_func)
    return func(arg)


def run_with_timeout(func, args, timeout=5):
    with pebble.ProcessPool(max_workers=1) as pool:
        future = pool.schedule(_run_serialized_dill_func, args=(dill.dumps(func), args), timeout=timeout)
        return future.result()


def aggregate_cost_from_logs(logs):
    # Find "cost" key in logs (arbitrary depth) and sum up all the found values
    def find_cost(log):
        _sum = 0
        if type(log) is dict:
            if "cost" in log:
                return log["cost"]
            for k, v in log.items():
                _sum += find_cost(v)
        elif type(log) is list:
            for v in log:
                _sum += find_cost(v)
        return _sum

    return find_cost(logs)


# Define the context manager
class SilentExecution:
    def __enter__(self):
        self._stdout = sys.stdout  # Save the current stdout
        self._stderr = sys.stderr  # Save the current stderr
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null
        sys.stderr = open(os.devnull, 'w')  # Redirect stderr to null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()  # Close the null file
        sys.stderr.close()  # Close the null file
        sys.stdout = self._stdout  # Restore the original stdout
        sys.stderr = self._stderr  # Restore the original stderr
