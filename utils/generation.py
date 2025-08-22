import csv
import json
import math
import random

import boto3
from botocore.config import Config
from openai import OpenAI
from openai import Client as OpenAIClient
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import re
import copy
import torch
from torch.nn.functional import cosine_similarity
from utils.misc import strip_dict, extract_json, sample_by_strategy, make_set, docstring_to_string

NEWLINE = "\n"

HF_PIPELINE_MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistral-7b-instruct-0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x22b-instruct-0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistral-large-instruct-2407": "mistralai/Mistral-Large-Instruct-2407",
    "codestral-22b-0.1": "mistralai/Codestral-22B-v0.1",
    "t5-small": "google-t5/t5-small",
    "t5-base": "google-t5/t5-base",
    "t5-large": "google-t5/t5-large",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "sentence-t5-base": "sentence-transformers/sentence-t5-base",
    "sentence-t5-large": "sentence-transformers/sentence-t5-large"
}
HF_SFORMER_MODELS = {
    "gte-qwen-1.5-7b-instruct": "Alibaba-NLP/gte-Qwen1.5-7B-instruct",
    "gte-qwen-2-1.5b-instruct": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "gte-qwen-2-7b-instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "molformer": "/home/ubuntu/.cache/huggingface/hub/models--sentence-transformers--ibm--MoLFormer-XL-both-10pct"
    # "ibm/MoLFormer-XL-both-10pct"
}

BEDROCK_CLAUDE_MODELS = {
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0"
}

BEDROCK_LLAMA_MODELS = {
    "llama-3-8b-instruct-bedrock": "meta.llama3-8b-instruct-v1:0",
    "llama-3-70b-instruct-bedrock": "meta.llama3-70b-instruct-v1:0",
    "llama-3.1-8b-instruct-bedrock": "meta.llama3-1-8b-instruct-v1:0",
    "llama-3.1-70b-instruct-bedrock": "meta.llama3-1-70b-instruct-v1:0",
    "llama-3.1-405b-instruct-bedrock": "meta.llama3-1-405b-instruct-v1:0",
}

BEDROCK_MISTRAL_MODELS = {
    "mistral-7b-instruct-0.2": "mistral.mistral-7b-instruct-v0:2",
    "mixtral-8x7b-instruct-0.1": "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral-large-2402": "mistral.mistral-large-2402-v1:0",
    "mistral-large-2407": "mistral.mistral-large-2407-v1:0"
}

BEDROCK_COHERE_MODELS = {
    "embed-english-v3": "cohere.embed-english-v3"
}

OPENAI_MODELS = {
    "gpt-4o": "gpt-4o-2024-08-06"
}

OPENAI_COSTS = {
    "gpt-4o-2024-08-06": {
        "input": 0.0000025,
        "output": 0.00001
    }
}

SGLANG_MODELS = {
    "gemma-2-2b-it": "google/gemma-2-2b-it",
}

NO_CONVERSE_SUPPORT = [
    {**BEDROCK_CLAUDE_MODELS, **BEDROCK_LLAMA_MODELS, **BEDROCK_MISTRAL_MODELS, **BEDROCK_COHERE_MODELS}[k] for k in
    ["mistral-7b-instruct-0.2", "mixtral-8x7b-instruct-0.1", "mistral-large-2402", "embed-english-v3"]]

SYS_PROMPT = {"role": "system", "content": f"""You are a helpful chatbot with high attention to detail \
who is not talkative and responds only with the answer and no additional conversation. All your responses should be in \
JSON format, i.e. {{key: value}}, where the key is always "response" and the value can be a string, int, list, or \
dict, depending on the context."""}

SYS_PROMPT_CODE = {"role": "system", "content": f"""You are a coding wizard who is excellent at solving puzzles. \
Your response first outputs a docstring block (<docstring> and </docstring>) containing a detailed description of the \
logic to be implemented in the code (be specific), and then a triple-backtick python block (```python and ```) containing the code. If the user requests for \
multiple solutions, you should provide a sequence of multiple docstring-python blocks one by one. For example:
<docstring>Test function to print a hello world statement.</docstring>
```python
def test():
    print('Hello, World!')
```"""}

SYS_PROMPT_DOCSTRING = {"role": "system", "content": f"""You are a coding wizard who is excellent at solving puzzles. \
Your response outputs only a docstring block (<docstring> and </docstring>) containing a detailed description of the \
logic to be implemented in code (be specific). If the user requests for multiple solutions, you should provide a \
sequence of multiple docstring blocks one by one. For example:
<docstring>Test function to print a hello world statement.</docstring>"""}

SYS_PROMPT_COT = {"role": "system", "content": f"""You are a helpful chatbot with high attention to detail who is not \
talkative and responds only with the answer and no additional conversation. All your responses should be in JSON \
format, i.e. {{key: value}}, where the key is always "response" and the value can be a string, int, list, or dict, \
depending on the context. Before generating the "response" key, first create a "reasoning" key at the same level as the \
"response" key, where you think step by step about what to put in the "response" by analyzing the available information."""}

SYS_PROMPT_CODE_COT = {"role": "system", "content": f"""You are a coding wizard who is excellent at solving puzzles. \
Your response first outputs a reasoning block (<reasoning> and </reasoning>), followed by a docstring block \
(<docstring> and </docstring>) containing a detailed description of the logic to be implemented in the code (be specific), and then a triple-backtick python \
block (```python and ```) containing the code. If the user requests for multiple solutions, you should provide a sequence of \
multiple reasoning-docstring-python blocks one by one. Think carefully about the reasoning and the code you provide \
each time. For example:
<reasoning>First, we need to define a function that prints a hello world statement.</reasoning>
<docstring>Test function to print a hello world statement.</docstring>
```python
def test():
    print('Hello, World!')
```"""}

SYS_PROMPT_DOCSTRING_COT = {"role": "system", "content": f"""You are a coding wizard who is excellent at solving \
puzzles. Your response first outputs a reasoning block (<reasoning> and </reasoning>), followed by a docstring block \
(<docstring> and </docstring>) containing a detailed description of the logic to be implemented in code (be specific). \
If the user requests for multiple solutions, you should provide a sequence of multiple reasoning-docstring blocks one \
by one. Think carefully about the reasoning and docstring you provide each time. For example:
<reasoning>First, we need to define a function that prints a hello world statement.</reasoning>
<docstring>Test function to print a hello world statement.</docstring>"""}

hf_to_bedrock_keys = {
    "max_new_tokens": "maxTokens",
    "max_tokens": "maxTokens",
    "top_p": "topP",
    "temperature": "temperature",
}

hf_to_openai_keys = {
    "max_new_tokens": "max_tokens",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "temperature": "temperature",
}

DEFAULT_DECODING_ARGS = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9
}


def get_bedrock_client(region='us-west-2', client=None):
    if client is not None:
        return client
    return boto3.client(service_name='bedrock-runtime',
                        region_name=region,
                        config=Config(
                            retries={
                                'max_attempts': 5,
                                'mode': 'standard'
                            },
                            read_timeout=300
                        ))


def get_openai_client(client=None):
    if client is not None:
        return client
    return OpenAI() if os.getenv("OPENAI_API_KEY") is not None else None


def extract_code_blocks(text: str, reasoning=True, docstring_only=False) -> list[tuple[str, str, str]]:
    # Manually correct some common mistakes in generations
    text = text.replace("```python\n<docstring>", "<docstring>")
    text = text.replace("import numpy as np\n", "")

    # Regular expressions to match <reasoning>, <docstring>, and Python code blocks
    reasoning_pattern = r'<reasoning>\s*(.*?)\s*</reasoning>'
    docstring_pattern = r'<docstring>\s*(.*?)\s*</docstring>'
    docstring_fallback_pattern = r'<docstring>\s*(.*?)\s*\n```python'
    code_pattern = r'```python\s*(.*?)\s*```'

    # Find all matches for reasoning, docstring, and code blocks
    docstring_matches = re.findall(docstring_pattern, text, re.DOTALL)
    if len(docstring_matches) == 0:
        docstring_matches = re.findall(docstring_fallback_pattern, text, re.DOTALL)
    code_matches = re.findall(code_pattern, text, re.DOTALL)

    # Combine the matches into a list of tuples
    if reasoning:
        reasoning_matches = re.findall(reasoning_pattern, text, re.DOTALL)
        if docstring_only:
            blocks = list(zip(reasoning_matches, docstring_matches))
        else:
            blocks = list(zip(reasoning_matches, docstring_matches, code_matches))
    else:
        if docstring_only:
            blocks = list(zip(docstring_matches, [None] * len(docstring_matches)))
        else:
            blocks = list(zip(docstring_matches, code_matches))

    return blocks


def invoke_bedrock(query,
                   model_id,
                   bedrock_client=None,
                   system_prompt="You are a helpful assistant.",
                   use_converse=True,
                   **kwargs):
    bedrock = get_bedrock_client(client=bedrock_client)
    accept = 'application/json'
    contentType = 'application/json'
    if use_converse and model_id not in NO_CONVERSE_SUPPORT:
        response = bedrock.converse(
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': query,
                        }
                    ]
                },
            ],
            system=[
                {
                    'text': system_prompt
                },
            ],
            inferenceConfig={
                'maxTokens': 512,
                'temperature': 0.6,
                'topP': 0.9,
                **{hf_to_bedrock_keys[k]: kwargs[k] for k in kwargs if k in hf_to_bedrock_keys}
            }
        )
        return response['output']['message']['content'][0]['text']
    else:
        if model_id in BEDROCK_CLAUDE_MODELS.values():
            decoding_args = {**{"max_tokens": 512}, **kwargs}
            body = json.dumps({
                "system": system_prompt,
                "anthropic_version": "bedrock-2023-05-31",
                **decoding_args,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
                ]})
        elif model_id in BEDROCK_LLAMA_MODELS.values():
            decoding_args = {**{"max_gen_len": 512}, **kwargs}
            body = json.dumps({
                "system": system_prompt,
                **decoding_args,
                "messages": [{
                    "role": "user",
                    "content": query
                }]
            })
        elif model_id in BEDROCK_MISTRAL_MODELS.values():
            decoding_args = {**{"max_tokens": 512}, **kwargs}
            body = json.dumps({
                "prompt": f"[INST] {system_prompt}\n\n{query}\n[/INST]",
                **decoding_args,
            })
        else:
            raise ValueError(f"Model ID {model_id} not supported.")

        response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        if model_id in BEDROCK_MISTRAL_MODELS.values():
            return response_body['outputs'][0]['text']
        return response_body['content'][0]['text']


def invoke_openai(query,
                  model_id,
                  openai_client=None,
                  system_prompt="You are a helpful assistant.",
                  **kwargs):
    client = get_openai_client(client=openai_client)
    decoding_args = {
        'max_tokens': 512,
        'temperature': 0.6,
        'top_p': 0.9,
        **{hf_to_openai_keys[k]: kwargs[k] for k in kwargs if k in hf_to_openai_keys}
    }
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        **decoding_args
    )
    return completion.choices[0].message.content, dict(completion.usage)


def invoke_sglang(query,
                  model_id,
                  system_prompt="You are a helpful assistant.",
                  **kwargs):
    client = OpenAIClient(base_url="http://127.0.0.1:30000/v1", api_key="None")
    decoding_args = {
        'max_tokens': 512,
        'temperature': 0.6,
        'top_p': 0.9,
        **{hf_to_openai_keys[k]: kwargs[k] for k in kwargs if k in hf_to_openai_keys}
    }
    if "gemma" in model_id:
        messages = [
            {
                "role": "user",
                "content": system_prompt + "\n\n" + query
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        **decoding_args
    )
    return completion.choices[0].message.content


def get_response(model, messages, parse_as_json=True, parse_as_code=False, docstring_only=False, silent=True,
                 is_hf=True, msg_log=None, max_retry=5, cur_retry=0, bedrock_client=None, openai_client=None, **kwargs):
    usage = None
    if is_hf:
        # LOCAL
        decoding_args = {**DEFAULT_DECODING_ARGS, **kwargs}
        _messages = copy.deepcopy(messages)
        outputs = model(_messages, **decoding_args)
        _messages = outputs[0]["generated_text"]
        latest_role = _messages[-1]['role']
        latest_message = _messages[-1]['content'].strip()
        _json = None
    else:
        # BEDROCK
        assert messages[0]['role'] == 'system' and messages[1]['role'] == 'user'
        decoding_args = {**DEFAULT_DECODING_ARGS, **kwargs}
        if model in {**BEDROCK_CLAUDE_MODELS, **BEDROCK_LLAMA_MODELS, **BEDROCK_MISTRAL_MODELS,
                     **BEDROCK_COHERE_MODELS}.values():
            decoding_args['max_tokens'] = decoding_args['max_new_tokens']
            if 'max_new_tokens' in decoding_args:
                del decoding_args['max_new_tokens']
            if 'eos_token_id' in decoding_args:
                del decoding_args['eos_token_id']
            if 'do_sample' in decoding_args:
                del decoding_args['do_sample']
            outputs = invoke_bedrock(system_prompt=messages[0]['content'], query=messages[1]['content'],
                                     model_id=model, bedrock_client=bedrock_client, **decoding_args)
        elif model in OPENAI_MODELS.values():
            outputs, usage = invoke_openai(system_prompt=messages[0]['content'], query=messages[1]['content'],
                                           model_id=model, openai_client=bedrock_client, **decoding_args)
            usage["cost"] = OPENAI_COSTS[model]["input"] * usage["prompt_tokens"] + OPENAI_COSTS[model]["output"] * \
                            usage["completion_tokens"]
        elif model in SGLANG_MODELS.values():
            outputs = invoke_sglang(system_prompt=messages[0]['content'], query=messages[1]['content'],
                                    model_id=model, **decoding_args)
        else:
            raise ValueError(f"Model ID {model} not supported.")
        latest_message = outputs
        _messages = messages + [{'role': 'assistant', 'content': latest_message}]
        latest_role = 'assistant'
        _json = None

    if msg_log is not None:
        msg_log.append(_messages)
        if usage is not None:
            msg_log[-1].append({k: usage[k] for k in ["completion_tokens", "prompt_tokens", "total_tokens", "cost"]})

    _response = latest_message.strip()
    if parse_as_code:
        cot_reasoning, docstring, code = None, None, None
        try:
            _response = extract_code_blocks(_response, reasoning=True, docstring_only=docstring_only)
        except:
            raise ValueError(f"Failed to parse code response (1): {latest_message}")
        try:
            assert type(_response) is list
            if len(_response[0]) == 2:
                docstring, code = zip(*_response)
            elif len(_response[0]) == 3:
                cot_reasoning, docstring, code = zip(*_response)
            else:
                raise ValueError(f"Failed to parse code response (2): {latest_message}")
        except:
            raise ValueError(f"Failed to parse code response (3): {_response}")
        _json = {
            "response": list(code),
            "reasoning": list(docstring),  # TODO: Refactor to change "reasoning" to "docstring"
            "cot": list(cot_reasoning)
        }
        # _json = {
        #     "response": [_r["code"].strip() for _r in _response],
        #     "reasoning": [_r["docstring"].strip() for _r in _response]
        # }
        latest_message = json.dumps(_json, indent=2)
    elif parse_as_json:
        try:
            if _response.startswith('"') and _response.endswith('"'):
                _response = json.loads(_response)
            if _response.startswith("```json"):
                _response = _response[len("```json"):]
            if _response.endswith("```"):
                _response = _response[:-len("```")]
            _json = extract_json(_response)
            if _json is None:
                _json = json.loads(_response)
                if type(_json) is str:
                    _json = extract_json(_json)
            assert type(_json) is dict
        except:
            if cur_retry < max_retry:
                cur_retry += 1
                if not silent:
                    print(f"Retry {cur_retry}/{max_retry} for parsing JSON response.")
                return get_response(model, messages, parse_as_json=parse_as_json, parse_as_code=parse_as_code,
                                    docstring_only=docstring_only, silent=silent, is_hf=is_hf, msg_log=msg_log,
                                    max_retry=max_retry, cur_retry=cur_retry, bedrock_client=bedrock_client,
                                    openai_client=openai_client, **kwargs)
            else:
                raise ValueError(f"Failed to parse JSON response: {latest_message}")
        # Strip values in the JSON
        strip_dict(_json)
        latest_message = json.dumps(_json, indent=2)

    if not silent:
        print(f"{latest_role.upper()}: {latest_message}")

    return {"messages": _messages, "last_obj": _json}


def invoke_bedrock_embeddings(model_id, texts, input_type="search_document", bedrock_client=None, batch_size=96):
    bedrock = get_bedrock_client(client=bedrock_client)
    accept = "*/*"
    content_type = 'application/json'
    embeddings = []
    for i in range(0, len(texts), batch_size):
        body = json.dumps({
            "texts": texts[i:i + batch_size],
            "input_type": input_type,
            "truncate": "END"
        })
        response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        embeddings += response_body['embeddings']
    assert len(embeddings) == len(texts)
    return torch.tensor(embeddings)


def get_seq_from_repr(proposed_x_repr, sampled_x, sampled_x_repr, gen_model, prompt_cls, sampled_y=None,
                      n_cands=1, max_retry=5, topk=20, n_rand_samples=None, increasing_order=True, n_low_previous=0,
                      codegen=False, max_score=1., min_normed=0.1, max_normed=0.8, use_sim_for_selection=False,
                      sample_unique=True, multiturn_retry=False, task_demos=None, is_duplicate=None,
                      prev_trials=None, generate_feedback=False, normalize_scores=False, llm_log=None,
                      use_target_score=True, use_exploit_threshold=True, human_input=False, verbose=False,
                      lowercase=True, **instruct_kwargs):
    # Compute scores if not provided
    if sampled_y is not None:
        if use_sim_for_selection:
            selection_scores = cosine_similarity(proposed_x_repr[None, :], sampled_x_repr).squeeze()
        else:
            selection_scores = sampled_y.clone()
        prompt_scores = sampled_y.clone()
    else:
        selection_scores = prompt_scores = cosine_similarity(proposed_x_repr[None, :], sampled_x_repr).squeeze()

    # Keep unique
    _sampled_x, unique_idxs = [], []
    for i in range(len(sampled_x)):
        if (is_duplicate is not None and not is_duplicate(sampled_x[i], _sampled_x)) or (
                is_duplicate is None and sampled_x[i] not in _sampled_x):
            _sampled_x.append(sampled_x[i])
            unique_idxs.append(i)
    sampled_x = _sampled_x
    selection_scores, prompt_scores = selection_scores[torch.tensor(unique_idxs)], prompt_scores[
        torch.tensor(unique_idxs)]

    # Get K nearest neighbors
    k = min(topk, len(sampled_x))
    knn_idxs = selection_scores.topk(len(selection_scores), largest=True).indices
    knn_idxs, not_in_knn_idxs = knn_idxs[:k], knn_idxs[k:]
    # Sort knn_idxs by prompt_scores
    knn_idxs = knn_idxs[torch.argsort(prompt_scores[knn_idxs], descending=not increasing_order)]

    # Normalize scores
    if normalize_scores:
        # Ensure no negatives (except -1)
        not_minus1_idxs = prompt_scores[knn_idxs] != -1
        knn_idxs_not_minus1 = knn_idxs[not_minus1_idxs]
        _min_score = prompt_scores[knn_idxs_not_minus1].min()
        if _min_score < 0:
            prompt_scores[knn_idxs_not_minus1] = prompt_scores[
                                                     knn_idxs_not_minus1] - _min_score  # shift to non-negative
            prompt_scores[knn_idxs_not_minus1] = (
                    prompt_scores[knn_idxs_not_minus1] / (max_score - _min_score))  # scale to [0, 1]
        prompt_scores[knn_idxs_not_minus1] = prompt_scores[knn_idxs_not_minus1] * (
                max_normed - min_normed) + min_normed  # default: scale to [0.1, 0.8]

    # Compute scores for generating the prompt
    target_score, exploit_threshold = None, None
    if use_target_score:
        target_score = max_score  # round(min(max_score, 1.5 * scores.max().item()), 2)
        # Prevent scores from equaling or exceeding the target score
        prompt_scores = prompt_scores.clamp(max=target_score * 0.99)
    if use_exploit_threshold:
        exploit_threshold = 0.95 * (target_score if use_target_score else max_score)

    # Prepare arguments for prompts
    high_scoring = [(sampled_x[i], round(prompt_scores[i].item(), 4)) for i in knn_idxs]
    if n_rand_samples is not None:
        # Used for EA (e.g. LMX)
        high_scoring = random.sample(high_scoring, min(n_rand_samples, len(high_scoring)))
    low_scoring = [sampled_x[i] for i in not_in_knn_idxs[-n_low_previous:]] if (
            n_low_previous > 0 and len(not_in_knn_idxs) > 0) else None

    # Generate feedback (self-refine)
    feedback = None
    if generate_feedback:
        user_msg = {
            "role": "user",
            "content": prompt_cls.feedback(
                high_scoring=high_scoring, low_scoring=low_scoring, increasing_order=increasing_order,
                target_score=target_score, exploit_threshold=exploit_threshold, prev_trials=prev_trials,
                task_demos=task_demos, **instruct_kwargs)
        }
        messages = [
            gen_model["system_prompt_feedback"],
            {"role": user_msg["role"], "content": user_msg["content"] % n_cands}
        ]
        _res = get_response(gen_model["model"], messages, silent=not verbose, is_hf=gen_model["is_hf"],
                            msg_log=llm_log, bedrock_client=gen_model.get("client", None),
                            openai_client=gen_model.get("client", None), **gen_model["decoding_args"])
        feedback = str(_res["last_obj"]["response"])

    # Generate candidate
    user_msg = {
        "role": "user",
        "content": prompt_cls.instruction(
            high_scoring=high_scoring, low_scoring=low_scoring, increasing_order=increasing_order,
            target_score=target_score, exploit_threshold=exploit_threshold, prev_trials=prev_trials, feedback=feedback,
            task_demos=task_demos, **instruct_kwargs)
    }
    messages = [gen_model["system_prompt"], {"role": user_msg["role"], "content": user_msg["content"] % n_cands}]
    if human_input:
        print(json.dumps(messages, indent=2))
        res_val = [input("Enter your guess: ")]
    else:
        _res = get_response(gen_model["model"], messages, silent=not verbose, is_hf=gen_model["is_hf"],
                            parse_as_code=codegen, msg_log=llm_log, bedrock_client=gen_model.get("client", None),
                            openai_client=gen_model.get("client", None),
                            docstring_only=instruct_kwargs.get("output_docstring_only", False),
                            **gen_model["decoding_args"])
        if codegen:
            _response_val = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
                _res["last_obj"]["response"]]
            _reasoning_val = _res["last_obj"]["reasoning"] if type(_res["last_obj"]["reasoning"]) is list else [
                _res["last_obj"]["reasoning"]]
            res_val = list(zip(_reasoning_val, _response_val))
        else:
            res_val = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
                _res["last_obj"]["response"]]

    # Fix for semantle: make sure the response is a list of strings
    if type(res_val[0]) is dict and 'word' in res_val[0]:
        res_val = list(map(lambda x: x['word'], res_val))

    res = list(make_set([r for r in list(map(lambda x: x if (codegen or not lowercase) else x.lower(), res_val))]))
    repeats = [r for r in res if r in sampled_x]
    res = [r for r in res if r not in sampled_x]

    # Retry if the desired number of candidates is not reached
    retry_count = 1
    while len(res) < n_cands and retry_count <= max_retry:
        n_missing = n_cands - len(res)
        repeats = list(make_set(repeats))
        if verbose:
            print(f"Generating seqs from repr: {n_missing} left. Retrying ({retry_count}/{max_retry})")
        if not multiturn_retry:
            messages = [gen_model["system_prompt"],
                        {"role": user_msg["role"], "content": user_msg["content"] % n_missing}]
        else:
            messages = _res["messages"] + [{"role": "user",
                                            "content": f"""Retrying ({retry_count}/{max_retry}). \
Please make %s new guess{'es' if n_missing > 1 else ''} that {'are' if n_missing > 1 else 'is'} not in any of your \
previous guesses and could lead to a higher score.""" % n_missing}]
        if len(repeats) > 0:
            if verbose:
                print(f"Repeats: {', '.join(repeats)}")
            user_msg_for_repeats = prompt_cls.instruction(
                high_scoring=high_scoring, low_scoring=low_scoring, increasing_order=increasing_order,
                target_score=target_score, exploit_threshold=exploit_threshold, prev_trials=prev_trials,
                feedback=feedback, repeats=repeats[::-1], task_demos=task_demos, **instruct_kwargs)
            messages[-1]["content"] = user_msg_for_repeats % n_missing
        if human_input:
            print(json.dumps(messages, indent=2))
            new_res_val = [input("Enter your guess: ")]
        else:
            _res = get_response(gen_model["model"], messages, parse_as_code=codegen,
                                docstring_only=instruct_kwargs.get("output_docstring_only", False),
                                silent=not verbose, is_hf=gen_model["is_hf"], msg_log=llm_log,
                                bedrock_client=gen_model.get("client", None),
                                openai_client=gen_model.get("client", None),
                                **{
                                    **gen_model["decoding_args"],
                                    "temperature": min(
                                        1, gen_model['decoding_args']['temperature'] + (retry_count - 1) * 0.05),
                                    "top_p": min(1, gen_model['decoding_args']['top_p'] + (retry_count - 1) * 0.0125)
                                })
            if codegen:
                _response_val = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
                    _res["last_obj"]["response"]]
                _reasoning_val = _res["last_obj"]["reasoning"] if type(_res["last_obj"]["reasoning"]) is list else [
                    _res["last_obj"]["reasoning"]]
                new_res_val = list(zip(_reasoning_val, _response_val))
            else:
                new_res_val = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
                    _res["last_obj"]["response"]]

            # Fix for semantle: make sure the response is a list of strings
            if type(new_res_val[0]) is dict and 'word' in new_res_val[0]:
                new_res_val = list(map(lambda x: x['word'], new_res_val))

        new_res = list(make_set([r for r in list(map(lambda x: x if (codegen or not lowercase) else x.lower(), new_res_val))]))
        for r in new_res:
            if (r not in res and r not in sampled_x) or (not sample_unique and retry_count == max_retry):
                # Add if unique, or if sample_unique is false and we're on the last retry
                res.append(r)
                if len(res) == n_cands:
                    break
            else:
                repeats.append(r)
        retry_count += 1

    if len(res) > n_cands:
        res = random.sample(res, n_cands)

    assert len(res) > 0

    return res


def get_candidates(n_cands, batch_size, gen_model, prompt_cls, strategy="random", target=None, task_demos=None,
                   codegen=False, max_retry=5, candidates_fname=None, candidates_fname_key=None, llm_log=None,
                   include_scores=False, n_parallel=10, json_keys=None, json_score_key="score", seen_set=None,
                   lowercase=True, **instruct_kwargs):
    res_scores = None
    if candidates_fname is not None and candidates_fname != "":
        # Load candidates from disk
        if candidates_fname.endswith(".json"):
            with open(candidates_fname, "r") as fh:
                res = json.load(fh)
                if type(res) is dict and candidates_fname_key is not None:
                    res = res[candidates_fname_key]
                res = [r for r in res if r != target]
            if n_cands is None:
                n_cands = len(res)
            if n_cands < len(res):
                res = sample_by_strategy(n_cands, res, strategy=strategy)
            if json_keys is not None:
                res = [tuple([r[jkey] for jkey in json_keys]) for r in res]
            if include_scores and json_score_key in res[0]:
                res_scores = [r[json_score_key] for r in res]
        elif candidates_fname.endswith(".csv"):
            with open(candidates_fname, "r") as fh:
                res_val = [row for row in csv.reader(fh) if row[0] != target][1:]  # skip header
            if n_cands is None:
                n_cands = len(res_val)
            if n_cands < len(res_val):
                res_val = sample_by_strategy(n_cands, res_val, strategy=strategy)
            if include_scores:
                res, res_scores = zip(*[(r[0], float(r[1])) for r in res_val])
            else:
                res = [r[0] for r in res_val]
        else:
            raise NotImplementedError
        print(f"Loaded {len(res)} candidates from {candidates_fname}")
    else:
        if n_cands is None:
            raise ValueError("--n_unlabeled must be provided when --candidates_fname is not provided.")
        print(f"Generating candidates...")
        results_set = set()
        retry_count = 0
        while len(results_set) < n_cands and retry_count <= max_retry:
            def _get_response(_batch_size):
                user_msg = {
                    "role": "user",
                    "content": prompt_cls.warmstart(min(n_cands - len(results_set), _batch_size), task_demos=task_demos,
                                                    previous=None if retry_count == 0 else results_set,
                                                    **instruct_kwargs)
                }
                messages = [gen_model["system_prompt"], user_msg]
                res = get_response(gen_model["model"], messages, silent=True, is_hf=gen_model["is_hf"],
                                   parse_as_code=codegen,
                                   docstring_only=instruct_kwargs.get("output_docstring_only", False), msg_log=llm_log,
                                   bedrock_client=gen_model.get("client", None),
                                   openai_client=gen_model.get("client", None), **gen_model["decoding_args"])
                if codegen:
                    _response_val = res["last_obj"]["response"] if type(res["last_obj"]["response"]) is list else [
                        res["last_obj"]["response"]]
                    _reasoning_val = res["last_obj"]["reasoning"] if type(res["last_obj"]["reasoning"]) is list else [
                        res["last_obj"]["reasoning"]]
                    res_val = list(zip(_reasoning_val, _response_val))
                else:
                    res_val = res["last_obj"]["response"] if type(res["last_obj"]["response"]) is list else [
                        res["last_obj"]["response"]]
                if target is not None:
                    res = make_set(r.lower() for r in res_val if r.lower() != target)
                else:
                    res = make_set(res_val)
                return res

            with ThreadPoolExecutor(max_workers=n_parallel) as executor:
                futures = [executor.submit(_get_response,
                                           batch_size if (batch_size is not None and batch_size > 0) else n_cands) for _
                           in
                           range(math.ceil(
                               (n_cands - len(results_set)) / (
                                   batch_size if (batch_size is not None and batch_size > 0) else n_cands)))]
                print(
                    f"Candidates: {n_cands - len(results_set)} left. Attempt ({retry_count}/{max_retry}). Launching {len(futures)} calls (max {n_parallel} in parallel)...")
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        if seen_set is not None:
                            res = [r for r in res if r not in seen_set]
                        results_set = results_set.union(res)
                    except Exception as e:
                        print(f"Exception occurred: {e}")
                retry_count += 1
        res = list(results_set)
        print(f"Generated {len(res)} candidates.")

    if len(res) > n_cands:
        res = random.sample(res, n_cands)
        print(f"Sampled {n_cands} candidates.")

    if res_scores is not None:
        res, res_scores = zip(*sorted(zip(res, res_scores), key=lambda x: x[1], reverse=True))
        res_scores = list(res_scores)

    return list(res), res_scores


def fix_candidate(candidate, model, prompt_cls, codegen=False, error_msg=None, predictions=None, llm_log=None,
                  **instruct_kwargs):
    user_msg = {
        "role": "user",
        "content": prompt_cls.fix(candidate=candidate, error_msg=error_msg, predictions=predictions, **instruct_kwargs)
    }
    messages = [model["system_prompt"], {"role": user_msg["role"], "content": user_msg["content"]}]
    if codegen:
        _res = get_response(model["model"], messages, silent=True, is_hf=model["is_hf"],
                            parse_as_code=codegen, msg_log=llm_log, bedrock_client=model.get("client", None),
                            openai_client=model.get("client", None), **model["decoding_args"])
        _response_val = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
            _res["last_obj"]["response"]]
        _reasoning_val = _res["last_obj"]["reasoning"] if type(_res["last_obj"]["reasoning"]) is list else [
            _res["last_obj"]["reasoning"]]
        res = list(zip(_reasoning_val, _response_val))
    else:
        _res = get_response(model["model"], messages, silent=True, is_hf=model["is_hf"], parse_as_code=codegen,
                            msg_log=llm_log, bedrock_client=model.get("client", None),
                            openai_client=model.get("client", None), **model["decoding_args"])
        res = _res["last_obj"]["response"] if type(_res["last_obj"]["response"]) is list else [
            _res["last_obj"]["response"]]
    return res
