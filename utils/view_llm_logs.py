import json
import os
import argparse


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--path", type=str
        )


def show(seed, key, iteration=None):
    for call in (logs["messages"][str(seed)][key][str(iteration)] if iteration is not None else logs["messages"][seed][key]):
        for turn in call:
            print(f"""\n\n###{turn["role"].upper()}###
{turn["content"]}""")


if __name__ == "__main__":
    args = ArgParser().parse_args()
global logs
with open(args.path, "r") as fh:
    logs = json.load(fh)

print("To view logs, use the following function:\n\nshow(seed, key, iteration)")
print(f"Possible values for seed: {logs['messages'].keys()}")
print(f"Possible values for key: {logs['messages'][list(logs['messages'].keys())[0]].keys()}\n\n")

breakpoint()
