import math
import os
import json
import time
import random
import sys
import atexit
import traceback

import numpy as np
import torch
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.nn.functional import cosine_similarity
from simcse import SimCSE
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch.utils.data as data_utils

from utils.arguments import ArgParser
from utils.bo import get_surrogate, optimize_acq_fn, get_acq_fn, plot_posterior, plot_trace
from utils.generation import HF_PIPELINE_MODELS, BEDROCK_CLAUDE_MODELS, SYS_PROMPT, \
    SYS_PROMPT_COT, BEDROCK_LLAMA_MODELS, BEDROCK_MISTRAL_MODELS, get_seq_from_repr, get_candidates, get_response, \
    SYS_PROMPT_CODE, SYS_PROMPT_DOCSTRING, get_bedrock_client, SYS_PROMPT_CODE_COT, SYS_PROMPT_DOCSTRING_COT, \
    BEDROCK_COHERE_MODELS, invoke_bedrock_embeddings, HF_SFORMER_MODELS, fix_candidate, get_openai_client, \
    OPENAI_MODELS, SGLANG_MODELS
from utils.misc import save_llm_logs, sample_by_strategy, TORCH_DTYPE, stringify_grid, run_with_timeout
from utils import prompts as Prompts
from utils.prompts import NEWLINE

os.environ["TOKENIZERS_PARALLELISM"] = "false"
OPT_SCORE = 1.0
INV_SCORE = -1.0


def python_eval(func_str, arg, func_name="transform", use_numpy=True):
    error_msg = None
    _func_str = func_str.strip().replace("```python", "").replace("```", "")
    _func_str_indented = "\n    ".join(_func_str.splitlines())
    _func_wrapper = f"""\
def wrapper(arg):
    from typing import List
    import itertools
    import pandas as pd
    import math
    {_func_str_indented}
    return {func_name}(arg)
"""
    local_scope = {}
    try:
        exec(_func_wrapper, globals(), local_scope)
        transform = local_scope["wrapper"]
        result = run_with_timeout(transform, arg if not use_numpy else np.array(arg), timeout=20)
        if use_numpy:
            result = result.tolist()
    except:
        error_msg = traceback.format_exc()
        # if args.verbose:
        if args.verbose_codegen_errors:
            print(f"\nError executing generated code:\n{_func_wrapper}")
            print(f"Error:\n{error_msg}")
        result = None
    return result, error_msg


def get_bbox_model(bbox_model_name, enable_cot=False):
    if bbox_model_name == "codegen":
        return python_eval
    else:
        return get_gen_model(gen_model_name=bbox_model_name, enable_cot=enable_cot)


def get_repr_model(repr_model_name):
    if repr_model_name == "simcse":
        return SimCSE("princeton-nlp/sup-simcse-roberta-large")
    elif repr_model_name in HF_PIPELINE_MODELS:
        hf_model = HF_PIPELINE_MODELS[repr_model_name]
        model_pipeline = pipeline(
            model=hf_model,
            task="feature-extraction",
            device_map="auto",
            token=args.hf_access_token
        )
        if "t5" in repr_model_name:
            model_pipeline.model = model_pipeline.model.encoder
        return model_pipeline
    elif repr_model_name in HF_SFORMER_MODELS:
        hf_model = HF_SFORMER_MODELS[repr_model_name]
        model = SentenceTransformer(hf_model,
                                    trust_remote_code=True,
                                    token=args.hf_access_token,
                                    device=args.device)
        return model
    elif repr_model_name in BEDROCK_COHERE_MODELS:
        model = BEDROCK_COHERE_MODELS[repr_model_name]
        return model
    else:
        raise NotImplementedError


def get_gen_model(gen_model_name, enable_cot=False, codegen=False):
    BEDROCK_MODELS = {**BEDROCK_CLAUDE_MODELS, **BEDROCK_LLAMA_MODELS, **BEDROCK_MISTRAL_MODELS}
    if gen_model_name in HF_PIPELINE_MODELS:
        model = HF_PIPELINE_MODELS[gen_model_name]
        model_pipeline = pipeline(
            model=model,
            task="text-generation",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=args.hf_access_token
        )
        model_pipeline.model.generation_config.pad_token_id = model_pipeline.tokenizer.eos_token_id
        terminators = [
            model_pipeline.tokenizer.eos_token_id,
            model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        rtn = {
            "model": model_pipeline,
            "is_hf": True,
            "decoding_args": {
                "max_new_tokens": args.llm_tokens,
                "eos_token_id": terminators,
                "do_sample": True,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p,
            },
            "system_prompt": (SYS_PROMPT_CODE if not enable_cot else SYS_PROMPT_CODE_COT) if codegen else (
                SYS_PROMPT if not enable_cot else SYS_PROMPT_COT),
            "system_prompt_feedback": SYS_PROMPT
        }
    elif gen_model_name in BEDROCK_MODELS:
        model = BEDROCK_MODELS[gen_model_name]
        rtn = {
            "model": model,
            "is_hf": False,
            "client": get_bedrock_client(client=BEDROCK_CLIENT),
            "decoding_args": {
                "max_new_tokens": args.llm_tokens,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p
            },
            "system_prompt": (SYS_PROMPT_CODE if not enable_cot else SYS_PROMPT_CODE_COT) if codegen else (
                SYS_PROMPT if not enable_cot else SYS_PROMPT_COT),
            "system_prompt_feedback": SYS_PROMPT
        }
    elif gen_model_name in OPENAI_MODELS:
        model = OPENAI_MODELS[gen_model_name]
        rtn = {
            "model": model,
            "is_hf": False,
            "client": get_openai_client(client=OPENAI_CLIENT),
            "decoding_args": {
                "max_new_tokens": args.llm_tokens,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p
            },
            "system_prompt": (SYS_PROMPT_CODE if not enable_cot else SYS_PROMPT_CODE_COT) if codegen else (
                SYS_PROMPT if not enable_cot else SYS_PROMPT_COT),
            "system_prompt_feedback": SYS_PROMPT
        }
    elif gen_model_name in SGLANG_MODELS:
        model = SGLANG_MODELS[gen_model_name]
        rtn = {
            "model": model,
            "is_hf": False,
            "client": None,
            "decoding_args": {
                "max_new_tokens": args.llm_tokens,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p
            },
            "system_prompt": SYS_PROMPT if not enable_cot else SYS_PROMPT_COT,
            "system_prompt_feedback": SYS_PROMPT
        }
    else:
        raise NotImplementedError
    return rtn


def get_prompts_for_repr(x, prompt="%s", strategy="docstring"):
    def _get_prompt_by_strategy(input, strat):
        if type(input) in [tuple, list]:
            if strat == "docstring":
                return f"Algorithm: {input[0]}"
            elif strat == "code":
                return f"Program:\n{input[1]}"
            elif strat == "docstring_code":
                return f"Docstring: {input[0]}\nProgram:\n{input[1]}"
            else:
                raise NotImplementedError
        else:
            return input

    # Prompt examples:
    # "%s"
    # "The task is to guess a hidden test word. The next guess is %s."
    return [prompt % _get_prompt_by_strategy(_x, strategy) for _x in (x if type(x) is list else [x])]


def get_representations(x, repr_model, pooling="mean", prompt="%s", proj_matrix=None, normalize=False,
                        diff_repr=None, device='cuda'):
    prompts = get_prompts_for_repr(x, prompt=prompt, strategy=args.repr_codegen_strategy)
    if "simcse" in type(repr_model).__name__.lower():
        _repr = repr_model.encode(prompts, silent=True)
    elif type(repr_model) is str and repr_model in list(BEDROCK_COHERE_MODELS.values()):
        # Bedrock models
        _repr = invoke_bedrock_embeddings(repr_model, prompts, bedrock_client=BEDROCK_CLIENT)
    elif type(repr_model).__name__ == "SentenceTransformer":
        _repr = repr_model.encode(prompts,
                                  prompt={"docstring_code": f"""Instruct: Given the following docstring and program \
for grid transformation, retrieve only those docstrings and programs that perform the same transformation.\n""",
                                          "docstring": f"""Instruct: Given the following grid transformation \
algorithm, retrieve only those algorithms that perform the same transformation.\n""",
                                          "code": f"""Instruct: Given the following grid transformation program, \
retrieve only those programs that perform the same transformation.\n""",
                                          "none": ""
                                          }[args.repr_codegen_strategy if args.repr_codegen_instruct else "none"],
                                  convert_to_tensor=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False,
                                  batch_size=4,
                                  device=device)
    else:
        # LLMs
        last_hidden_state = repr_model(prompts, return_tensors=True)
        if type(last_hidden_state) is not list:
            last_hidden_state = [last_hidden_state]
        if pooling == "mean":
            _repr = torch.cat([last.mean(dim=1) for last in last_hidden_state], dim=0)
        elif pooling == "last":
            _repr = torch.cat([last[:, -1] for last in last_hidden_state], dim=0)
        elif pooling == "first":
            _repr = torch.cat([last[:, 0] for last in last_hidden_state], dim=0)
        else:
            raise NotImplementedError

    if len(_repr.shape) == 1:
        _repr = _repr[None, :]
    if diff_repr is not None:
        _repr = _repr.to(device) - diff_repr.to(device)
    if proj_matrix is not None:
        _repr = _repr.to(device) @ proj_matrix.to(device)
    if normalize:
        _repr = torch.nn.functional.normalize(_repr)
    return _repr.squeeze().to(device)


def get_bbox_values(x=None, task=None, bbox_model=None, prompt_cls=None, mode="train", llm_log=None, device='cuda',
                    bbox_log=None, n_tasks=None, return_predictions=False, invalid_score=None):
    def _get_score(_gold, _prediction, func="hamming"):
        _score = None
        # Check if shapes are the same
        if np.shape(_gold) != np.shape(_prediction):
            raise ValueError(
                f"Shapes of gold ({np.shape(_gold)}) and prediction ({np.shape(_prediction)}) are not the same.")
        if func == "l2":
            _score = np.linalg.norm(_gold - _prediction)
            return 1. / _score if _score != 0 else 1.
        elif func == "l1":
            _score = np.linalg.norm(_gold - _prediction, ord=1)
            return 1. / _score if _score != 0 else 1.
        elif func == "hamming":
            _score = 1. - np.sum(_gold != _prediction) / np.prod(np.shape(_gold))
            return _score
        else:
            raise NotImplementedError

    scores = []
    is_valid = []
    predictions = []
    error_msgs = []
    for _x in x if type(x) is list else [x]:
        error_msg = None
        _scores = []
        _predictions = []
        _error_msgs = []
        _log = {"candidate": _x, "result": []}
        x_tasks = task[mode] if n_tasks is None else task[mode][:n_tasks]
        for _task in x_tasks:
            if bbox_model.__name__ == "python_eval":
                # code generation
                assert type(_x) is tuple
                prediction, error_msg = bbox_model(func_str=_x[1], arg=_task[0], func_name="transform")
            else:
                # prompt optimization
                user_msg = {
                    "role": "user",
                    "content": prompt_cls.blackbox(instruction=_x, input=_task[0])
                }
                messages = [bbox_model["system_prompt"], {"role": user_msg["role"], "content": user_msg["content"]}]
                _res = get_response(bbox_model["model"], messages, silent=not args.verbose, is_hf=bbox_model["is_hf"],
                                    msg_log=llm_log, **bbox_model["decoding_args"])
                prediction = _res["last_obj"]["response"]
            if prediction is None:
                # Error in declaring generated function
                _scores.append(None)
            else:
                try:
                    _scores.append(_get_score(np.array(_task[1]), np.array(prediction)))
                except:
                    error_msg = traceback.format_exc()
                    if args.verbose_codegen_errors:
                        print(f"\nError in output from the generated code:\n{prediction}\nvs.\n{_task[1]}")
                        print(f"Error:\n{error_msg}")
                    _scores.append(None)
            _predictions.append(prediction)
            _error_msgs.append(error_msg)
            if bbox_log is not None:
                _log["result"].append({
                    "input": str(_task[0]),
                    "output": str(_task[1]),
                    "prediction": str(prediction),
                    "score": _scores[-1],
                    "error_msg": error_msg
                })
            if _scores[-1] is None:
                break
        try:
            scores.append(np.mean(_scores))
            is_valid.append(True)
        except:
            scores.append((0. if args.set_invalid_to_zero else INV_SCORE) if invalid_score is None else invalid_score)
            is_valid.append(False)
        predictions.append(_predictions)
        error_msgs.append(_error_msgs)
        if bbox_log is not None:
            bbox_log.append(_log)
    scores = torch.tensor(scores, device=device)
    is_valid = torch.tensor(is_valid, device=device)

    if return_predictions:
        return scores, is_valid, predictions, error_msgs
    return scores, is_valid, error_msgs


def get_projection_matrix(low_dim, strategy='pca', warmstart_repr=None, candidates=None, device='cuda',
                          get_repr_args=None):
    if strategy == 'pca':
        if args.low_dim_pca_ncands == 0:
            # use warmstart set
            _reprs = warmstart_repr.to(device)
        else:
            sampled_cands = sample_by_strategy(args.low_dim_pca_ncands, candidates, strategy="random")
            _reprs = get_representations(sampled_cands, **get_repr_args)
        if len(_reprs) < low_dim:
            if args.verbose:
                print(
                    f"Warning: Number of candidates ({len(_reprs)}) is less than the target low dimension ({low_dim}).")
        _, _, proj_mat = torch.pca_lowrank(_reprs, q=min(low_dim, len(_reprs)))
    elif strategy == 'random':
        proj_mat = torch.randn(warmstart_repr.shape[1], low_dim).to(device)
        torch.nn.init.normal_(proj_mat, 0, 1)
        # Initialization code from InstructZero:
        # mu_hat = proj_mat.reshape(-1).mean().item()
        # std_hat = proj_mat.reshape(-1).std().item()
        # mu = 0.0
        # alpha = 1.
        # sigma = 1.
        # std = alpha * std_hat / (np.sqrt(low_dim) * sigma)
        # print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        # torch.nn.init.normal_(proj_mat, -1, 1)
        # torch.nn.init.uniform_(proj_mat, -1, 1)
    else:
        raise NotImplementedError

    return proj_mat


def get_optimization_hull(ncands, warmstart_repr=None, candidates=None, get_repr_args=None, device='cuda'):
    if ncands == 0:
        # Use warmstart set
        _reprs = warmstart_repr.to(device)
    else:
        sampled_cands = sample_by_strategy(ncands, candidates, strategy="random")
        _reprs = get_representations(sampled_cands, **get_repr_args)
    return _reprs


def get_bounds(ncands, warmstart_repr=None, candidates=None, get_repr_args=None, square_hull=False, hull_margin=1.,
               device='cuda'):
    # Get hull
    repr_hull = get_optimization_hull(ncands=ncands,
                                      warmstart_repr=warmstart_repr,
                                      candidates=candidates,
                                      get_repr_args=get_repr_args,
                                      device=device)
    # Compute bounds
    min_bounds = torch.ones(repr_hull.shape[1]).to(device) * repr_hull.min() if square_hull else repr_hull.min(
        dim=0).values
    max_bounds = torch.ones(repr_hull.shape[1]).to(device) * repr_hull.max() if square_hull else repr_hull.max(
        dim=0).values
    assert hull_margin >= 1
    hull_expansion_margin = (hull_margin - 1) * (max_bounds - min_bounds) / 2.
    bounds = torch.stack([min_bounds - hull_expansion_margin, max_bounds + hull_expansion_margin])
    return bounds.to(device)


def get_visualization_data(ncands, candidates, candidates_fpath, target, candidate_scores, get_repr_args,
                           task, bbox_model, prompt_cls, seed, device, bbox_logs, LLM_LOGS):
    if candidates_fpath is not None:
        if candidates_fpath.endswith(".json"):
            _candidates_fpath = candidates_fpath
        else:
            candidates_task_fpath = os.path.join(candidates_fpath, target.split('.')[0])
            _candidates_fpath = [f for f in os.listdir(candidates_task_fpath) if
                                 (f.startswith("candidates") and f.endswith(".json"))]
            _candidates_fpath = os.path.join(candidates_task_fpath, _candidates_fpath[0]) \
                if len(_candidates_fpath) > 0 else None
        candidates, candidate_scores = get_candidates(n_cands=None, candidates_fname=_candidates_fpath,
                                                      candidates_fname_key=target.split('.')[0],
                                                      strategy="random", json_keys=["docstring", "code"],
                                                      batch_size=None, gen_model=None, prompt_cls=None, n_parallel=None,
                                                      task_demos=None, codegen=None, llm_log=None, include_scores=True,
                                                      use_numpy=None, transpose=None, priors=None)
        if candidate_scores is None:
            candidate_scores, _valid, _, _ = get_bbox_values(x=candidates, task=task, bbox_model=bbox_model,
                                                             prompt_cls=prompt_cls, n_tasks=args.bbox_n_train,
                                                             mode="train", return_predictions=True,
                                                             llm_log=LLM_LOGS[seed]["bbox"], bbox_log=bbox_logs,
                                                             device=device)

    # Sample candidates
    candidates, candidate_scores = zip(
        *sample_by_strategy(ncands, list(zip(candidates, candidate_scores)), strategy="top"))
    # Get representations
    reprs = get_representations(list(candidates), **get_repr_args)
    # Get scores
    scores = torch.tensor(candidate_scores)
    # Sort by scores
    sorted_idxs = scores.argsort(descending=True)
    reprs = reprs[sorted_idxs]
    scores = scores[sorted_idxs]
    candidates = [candidates[i] for i in sorted_idxs]
    return candidates, reprs, scores


def is_duplicate(proposed_x, sampled_x):
    if args.repr_codegen_strategy == "docstring_code":
        return proposed_x in sampled_x
    elif args.repr_codegen_strategy == "docstring":
        return any(proposed_x[0] == _x[0] for _x in sampled_x)
    elif args.repr_codegen_strategy == "code":
        return any(proposed_x[1] == _x[1] for _x in sampled_x)
    else:
        raise NotImplementedError


def decode_repr(proposed_x_repr, sampled_x, sampled_x_repr, sampled_y,
                n_cands, gen_model, prompt_cls, task_demos, llm_log,
                requires_bo, device, counters, use_sim_for_selection,
                handle_duplicates=True):
    rtn = []
    decoded_xs = None
    original_proposed_repr = proposed_x_repr.clone()
    for _attempt in range(_max_attempts := 3):
        is_throttling_error = False
        try:
            decoded_xs = get_seq_from_repr(proposed_x_repr, sampled_x, sampled_x_repr, gen_model, prompt_cls,
                                           sampled_y=sampled_y, generate_feedback=args.generate_feedback,
                                           n_cands=n_cands, topk=args.vec2text_demos,
                                           n_rand_samples=args.vec2text_rand_samples,
                                           normalize_scores=args.vec2text_normalize,
                                           min_normed=args.vec2text_normalize_min, is_duplicate=is_duplicate,
                                           max_normed=args.vec2text_normalize_max,
                                           use_sim_for_selection=use_sim_for_selection,
                                           n_low_previous=args.opro_n_low_previous, task_demos=task_demos,
                                           sample_unique=args.vec2text_sample_unique,
                                           max_retry=args.vec2text_unique_retries, codegen=args.bbox_model == "codegen",
                                           multiturn_retry=args.vec2text_multiturn_retry,
                                           use_target_score=args.vec2text_target_score,
                                           use_exploit_threshold=args.vec2text_exploit_threshold,
                                           llm_log=llm_log, human_input=args.diagnostic_human_input,
                                           use_numpy=args.arc_use_numpy, transpose=args.arc_show_transpose,
                                           priors=args.arc_use_priors, docstring=args.arc_use_docstring,
                                           code=args.arc_use_code, scores=args.arc_use_scores,
                                           mix=args.arc_use_mixing, no_demos=args.arc_no_demos)
        except Exception as e:
            # Check if the exception is botocore.errorfactory.ThrottlingException
            if "ThrottlingException" in str(e):
                is_throttling_error = True
                print(f"Encountered ThrottlingException. Retrying ({_attempt + 1}/{_max_attempts}) in 3 seconds.")
                # Sleep for 3 seconds and retry
                time.sleep(3)
                continue
            if args.verbose:
                print("Error: Failed to decode representation.")
                traceback.print_exc()
            if args.debug:
                traceback.print_exc()
                breakpoint()
            return None
        if decoded_xs is not None:
            break

    if decoded_xs is None:
        if is_throttling_error:
            counters['n_throttling_errors'] += 1
        return None

    for decoded_x in decoded_xs:
        decoded_x_repr = None

        # Handle duplicate candidates
        if is_duplicate(decoded_x, sampled_x):
            if handle_duplicates:
                if requires_bo:
                    if args.repeat_cand_strategy == "bo_proposal":
                        decoded_x_repr = proposed_x_repr
                    elif args.repeat_cand_strategy == "skip":
                        if args.verbose:
                            print("Warning: Skipping repeated candidate")
                        continue
                    if counters['repeat_streak'] >= args.repeat_streak_threshold:
                        counters['n_repeat_streak_reached'] += 1
                        if args.repeat_streak_perturb:
                            # Perturb the proposal and try decoding again
                            for repeat_streak_retry_idx in range(3):
                                _proposed_x_repr = proposed_x_repr + torch.randn_like(proposed_x_repr) * (
                                        sampled_x_repr.std(dim=0) * repeat_streak_retry_idx)
                                decoded_cands = decode_repr(proposed_x_repr=_proposed_x_repr, sampled_x=sampled_x,
                                                            sampled_x_repr=sampled_x_repr, prompt_cls=prompt_cls,
                                                            n_cands=1, task_demos=task_demos,
                                                            sampled_y=sampled_y, gen_model=gen_model, llm_log=llm_log,
                                                            requires_bo=requires_bo, device=device, counters=counters,
                                                            use_sim_for_selection=use_sim_for_selection,
                                                            handle_duplicates=False)
                                if decoded_cands is not None:
                                    decoded_x, decoded_x_repr, _ = decoded_cands[0]
                                    counters['n_repeats_solved_by_perturbing'] += 1
                                    break
                else:
                    if args.verbose:
                        print("Warning: Skipping repeated candidate")
                    continue
            else:
                return None
        rtn.append((decoded_x, decoded_x_repr, original_proposed_repr))

    return rtn


def fix_candidate_wrapper(proposed_x, proposed_x_repr, proposed_y, proposed_preds, is_valid_proposal, gen_model,
                          bbox_model, prompt_cls, task, bbox_n_train, bbox_error_msg, fix_retries, fix_llm_log,
                          bbox_log, bbox_llm_logs, requires_bo, is_codegen, fixed_bbox_trial,
                          device='cuda', verbose=False):
    # Attempt to fix candidate and recompute score
    for fix_retry in range(fix_retries):
        if verbose:
            print(f"\nAttempting fix (retry {fix_retry + 1}/{fix_retries})")
        try:
            _fix_attempt_x = fix_candidate(proposed_x, gen_model, prompt_cls,
                                           task_demos=task["train"][:bbox_n_train],
                                           predictions=proposed_preds[
                                               0] if args.arc_show_pred_in_fix else None,
                                           codegen=is_codegen,
                                           error_msg=bbox_error_msg,
                                           llm_log=fix_llm_log,
                                           use_numpy=args.arc_use_numpy,
                                           transpose=args.arc_show_transpose,
                                           priors=args.arc_use_priors)
        except:
            continue
        assert type(_fix_attempt_x) is list
        _fix_attempt_x = _fix_attempt_x[0]
        _fix_attempt_y, _is_valid_proposal, _fix_attempt_preds, _bbox_error_msg = get_bbox_values(
            x=_fix_attempt_x, task=task,
            bbox_model=bbox_model,
            prompt_cls=prompt_cls,
            mode="train",
            bbox_log=bbox_log,
            llm_log=bbox_llm_logs,
            n_tasks=bbox_n_train,
            return_predictions=True,
            device=device)
        proposed_x = _fix_attempt_x
        proposed_y = _fix_attempt_y
        proposed_preds = _fix_attempt_preds
        bbox_error_msg = _bbox_error_msg[0][0]
        if is_valid_proposal := _is_valid_proposal[0].item():
            if verbose:
                print("\nFixed candidate")
            fixed_bbox_trial.append(fix_retry + 1)
            if requires_bo:
                proposed_x_repr = None
            break
    return proposed_x, proposed_x_repr, proposed_y, proposed_preds, is_valid_proposal


def revise_candidate_wrapper(proposed_x, proposed_x_repr, proposed_y, proposed_preds, gen_model, bbox_model,
                             revise_retries, prompt_cls, task, bbox_n_train, is_codegen, fix_llm_log, bbox_log,
                             bbox_llm_log, revision_improvements, optimum_score, device='cuda', verbose=False):
    revision_improvements.append(0.)
    for revise_retry in range(revise_retries):
        if proposed_y.item() >= optimum_score:
            break
        if verbose:
            print(f"\nRevising (retry {revise_retry + 1}/{revise_retries})")
        try:
            _revise_attempt_x = fix_candidate(proposed_x, gen_model, prompt_cls,
                                              task_demos=task["train"][:bbox_n_train],
                                              predictions=proposed_preds[0] if args.arc_show_pred_in_fix else None,
                                              codegen=is_codegen,
                                              llm_log=fix_llm_log,
                                              use_numpy=args.arc_use_numpy,
                                              transpose=args.arc_show_transpose,
                                              priors=args.arc_use_priors,
                                              improve=True)
        except:
            continue
        assert type(_revise_attempt_x) is list
        _revise_attempt_x = _revise_attempt_x[0]
        _revise_attempt_y, _is_revise_attempt_valid, _revise_attempt_preds, _ = get_bbox_values(
            x=_revise_attempt_x, task=task,
            bbox_model=bbox_model,
            prompt_cls=prompt_cls,
            mode="train",
            bbox_log=bbox_log,
            llm_log=bbox_llm_log,
            n_tasks=bbox_n_train,
            return_predictions=True,
            device=device)
        if _is_revise_attempt_valid[0].item() and _revise_attempt_y.item() > proposed_y.item():
            revision_improvements[-1] += (_revise_attempt_y.item() - proposed_y.item())
            proposed_x, proposed_y = _revise_attempt_x, _revise_attempt_y
            proposed_preds = _revise_attempt_preds
            proposed_x_repr = None
    return proposed_x, proposed_x_repr, proposed_y, proposed_preds


def aggregate_results(results):
    # Aggregate results across seeds
    n_opt = len([r for r in results if r["opt_found"]])
    steps_to_opt = sum([r["steps_to_opt"] for r in results if r["steps_to_opt"] != -1])
    agg_res = {
        "task": results[0]['task'],
        "target": results[0]['target'],
        "n_runs": args.n_seeds,
        "avg_time_elapsed": round(np.mean([r["time_elapsed"] for r in results]), 4),
        "n_opt": n_opt,
        "avg_opt_rate": round(n_opt / args.n_seeds, 4),
        "avg_steps_to_opt": round(steps_to_opt / n_opt, 4) if n_opt > 0 else None,
        "warmstart_avg_best_y": round(np.mean([r["warmstart_best_xy"][1] for r in results]), 4),
        "avg_best_y": round(np.mean([r["best_xy"][1] for r in results]), 4),
        "std_best_y": round(np.std([r["best_xy"][1] for r in results]), 4),
        "avg_gain_from_warmstart": round(np.mean([r["gain_from_warmstart"] for r in results]), 4),
        "std_gain_from_warmstart": round(np.std([r["gain_from_warmstart"] for r in results]), 4),
        "avg_test_score": round(np.mean([r["test_score"] for r in results]), 4),
        "std_test_score": round(np.std([r["test_score"] for r in results]), 4),
        "avg_n_proposals": round(np.mean([r["n_proposals"] for r in results]), 4),
        "avg_invalid_decodings": round(np.mean([r["n_invalid_decodings"] for r in results]), 4),
        "avg_invalid_bbox": round(np.mean([r["n_invalid_bbox"] for r in results]), 4),
        "avg_fixed_bbox": round(np.mean([r["n_fixed_bbox"] for r in results]), 4),
        "avg_revision_improvement": round(np.mean([r["avg_revision_improvement"] for r in results]), 4),
        "avg_repeat_cands": round(np.mean([r["n_repeat_cands"] for r in results]), 4),
        "avg_repeat_decoded_cands": round(np.mean([r["n_repeat_decoded_cands"] for r in results]), 4),
        "avg_repeat_streak_reached": round(np.mean([r["n_repeat_streak_reached"] for r in results]), 4),
        "avg_repeats_solved_by_perturbing": round(np.mean([r["n_repeats_solved_by_perturbing"] for r in results]), 4),
        "avg_throttling_errors": round(np.mean([r["n_throttling_errors"] for r in results]), 4),
        "avg_target_in_bounds": round(np.mean([r["target_in_bounds"] for r in results]), 4) if results[0][
                                                                                                   "target_in_bounds"] is not None else None,
        "trace": None,
        "args": args.__dict__
    }

    # Plot
    bo_best_y = []
    for i in range(args.n_seeds):
        seed_res = results[i]
        bo_best_y.append([tup[1] for tup in seed_res['trace_best_xy']])
        bo_best_y[-1] += [bo_best_y[-1][-1]] * ((args.n_evaluations + 1) - len(bo_best_y[-1]))
        assert len(bo_best_y[-1]) == (args.n_evaluations + 1)  # + 1 because of the initial best warmstart candidate
    plot_trace(np.array(bo_best_y), out_dir=out_dir)

    # Save aggregated results
    agg_res["trace"] = {
        "best_y": bo_best_y
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        fh.write(json.dumps(agg_res, indent=2))
    print(f"Aggregated results saved to {out_dir}/results.json")

    return agg_res


def _bo(target, gen_model, repr_model, bbox_model, prompt_cls, task=None,
        requires_bo=True, candidates_fname=None, seed=17, device='cuda', save_results=True):
    time_start = time.time()
    print(f"\nSEED: {seed}\n")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LLM_LOGS[seed] = {
        "candidates": [],
        "dim_reduction": [],
        "hull": [],
        "bo": {},
        "bbox": []
    }
    bbox_logs = []

    # Load or generate unlabelled candidates
    n_unlabeled = args.n_unlabeled
    unlabeled_cands, unlabeled_cands_scores = get_candidates(n_cands=n_unlabeled,
                                                             batch_size=args.candidate_gen_batch_size,
                                                             gen_model=gen_model, prompt_cls=prompt_cls,
                                                             strategy="random",
                                                             n_parallel=args.candidate_gen_n_parallel,
                                                             task_demos=task["train"][:args.bbox_n_train],
                                                             codegen=args.bbox_model == "codegen",
                                                             candidates_fname=candidates_fname,
                                                             llm_log=LLM_LOGS[seed]["candidates"],
                                                             include_scores=True, json_keys=["docstring", "code"],
                                                             use_numpy=args.arc_use_numpy,
                                                             transpose=args.arc_show_transpose,
                                                             priors=args.arc_use_priors)

    if unlabeled_cands_scores is None and (args.visualize_posterior or args.save_candidates):
        unlabeled_cands_scores, unlabeled_valid, unlabeled_preds, unlabeled_error_msg = get_bbox_values(
            x=unlabeled_cands, task=task, bbox_model=bbox_model, prompt_cls=prompt_cls, n_tasks=args.bbox_n_train,
            mode="train", return_predictions=True, llm_log=LLM_LOGS[seed]["bbox"], bbox_log=bbox_logs, device=device,
            invalid_score=INV_SCORE)

        # Fix and revise generated candidates
        n_invalid_bbox_cands_found, n_fixed_bbox_cands = 0, 0
        for i in tqdm.trange(len(unlabeled_cands), desc="Fix and revise"):
            is_unlabeled_valid = unlabeled_valid[i].item()
            if args.vec2text_fix_retries > 0 and not is_unlabeled_valid:
                n_invalid_bbox_cands_found += 1
                unlabeled_cands[i], _, unlabeled_cands_scores[i], unlabeled_preds[
                    i], is_unlabeled_valid = fix_candidate_wrapper(
                    proposed_x=unlabeled_cands[i], proposed_x_repr=None, proposed_y=unlabeled_cands_scores[i],
                    proposed_preds=unlabeled_preds[i], is_valid_proposal=is_unlabeled_valid, gen_model=gen_model,
                    bbox_model=bbox_model, is_codegen=args.bbox_model == "codegen", prompt_cls=prompt_cls,
                    task=task, bbox_n_train=args.bbox_n_train, bbox_error_msg=unlabeled_error_msg[i][0],
                    fix_retries=args.vec2text_fix_retries, fix_llm_log=LLM_LOGS[seed]["candidates"], bbox_log=bbox_logs,
                    bbox_llm_logs=LLM_LOGS[seed]["bbox"], requires_bo=requires_bo, fixed_bbox_trial=[],
                    device=device, verbose=args.verbose)
                if is_unlabeled_valid:
                    n_fixed_bbox_cands += 1
            if is_unlabeled_valid and args.vec2text_revise_retries > 0 and unlabeled_cands_scores[i].item() < OPT_SCORE:
                unlabeled_cands[i], _, unlabeled_cands_scores[i], unlabeled_preds[i] = revise_candidate_wrapper(
                    proposed_x=unlabeled_cands[i], proposed_x_repr=None, proposed_y=unlabeled_cands_scores[i],
                    proposed_preds=unlabeled_preds[i], gen_model=gen_model, bbox_model=bbox_model,
                    revise_retries=args.vec2text_revise_retries, prompt_cls=prompt_cls, task=task,
                    bbox_n_train=args.bbox_n_train, is_codegen=args.bbox_model == "codegen",
                    fix_llm_log=LLM_LOGS[seed]["candidates"], bbox_log=bbox_logs, bbox_llm_log=LLM_LOGS[seed]["bbox"],
                    revision_improvements=[], optimum_score=OPT_SCORE, device=device, verbose=args.verbose)
        print(f"Fixed {n_fixed_bbox_cands}/{n_invalid_bbox_cands_found} invalid candidates.")

        if not args.keep_invalid_unlabeled_candidates:
            unlabeled_cands = [unlabeled_cands[i] for i in range(len(unlabeled_cands)) if unlabeled_valid[i].item()]
            unlabeled_cands_scores = unlabeled_cands_scores[unlabeled_valid]
        unlabeled_cands_scores = unlabeled_cands_scores.tolist()
        unlabeled_cands, unlabeled_cands_scores = zip(
            *sorted(zip(unlabeled_cands, unlabeled_cands_scores), key=lambda x: x[1], reverse=True))
        unlabeled_cands, unlabeled_cands_scores = list(unlabeled_cands), list(unlabeled_cands_scores)

    assert len(unlabeled_cands) > 1  # Need more than 1 to run PCA

    if candidates_fname is None and args.save_candidates:
        save_cands = [{
            "docstring": _cand[0],
            "code": _cand[1],
            "score": _score
        } for _cand, _score in zip(unlabeled_cands,
                                   unlabeled_cands_scores if unlabeled_cands_scores is not None else [None] * len(
                                       unlabeled_cands))]
        # Sort by scores
        if unlabeled_cands_scores is not None:
            save_cands = sorted(save_cands, key=lambda x: x["score"], reverse=True)
        os.makedirs(out_dir, exist_ok=True)
        cand_fpath = f"candidates_{target.split('.')[0]}_{seed}.json"
        with open(os.path.join(out_dir, cand_fpath), "w") as fh:
            fh.write(json.dumps(save_cands, indent=2))
            print(f"Saved candidates to {os.path.join(out_dir, cand_fpath)}")
        if args.exit_after_candidate_gen:
            return None

    # Obtain warm-start set
    n_warmstart = args.n_warmstart
    sampled = sample_by_strategy(n_cands=n_warmstart,
                                 candidates=unlabeled_cands if unlabeled_cands_scores is None else list(
                                     zip(unlabeled_cands, unlabeled_cands_scores)),
                                 strategy=args.warmstart_strategy)
    if unlabeled_cands_scores is None:
        sampled_x, sampled_y = sampled, None
    else:
        sampled_x, sampled_y = zip(*sampled)
        sampled_x = list(sampled_x)
        sampled_y = torch.tensor(sampled_y, device=device)

    sampled_x_repr, diff_repr, proj_matrix, bounds, target_in_bounds = None, None, None, None, None
    if requires_bo:
        sampled_x_repr = get_representations(sampled_x,
                                             repr_model=repr_model,
                                             pooling=args.repr_llm_pooling,
                                             prompt=args.repr_prompt,
                                             device=device)

        # Subtract a generic representation vector from all representations to get subtle differences

        if args.repr_diff_vector:
            diff_repr = get_representations((
                "This function transforms the input grid into the output grid.",
                "def transform(input: np.ndarray) -> np.ndarray:\n    return output"
            ),
                repr_model=repr_model,
                pooling=args.repr_llm_pooling,
                prompt=args.repr_prompt,
                device=device)
            sampled_x_repr -= diff_repr

        # Optionally, get low-dim projection matrix
        if args.low_dim_strategy is not None and args.low_dim_strategy != "off":
            proj_matrix = get_projection_matrix(low_dim=args.low_dim, strategy=args.low_dim_strategy,
                                                warmstart_repr=sampled_x_repr, candidates=unlabeled_cands,
                                                get_repr_args={
                                                    "repr_model": repr_model,
                                                    "pooling": args.repr_llm_pooling,
                                                    "prompt": args.repr_prompt,
                                                    "diff_repr": diff_repr,
                                                    "device": device
                                                }, device=device)
            # Project already computed representations
            sampled_x_repr = sampled_x_repr @ proj_matrix
        if args.normalize_inputs:
            assert len(sampled_x_repr) == len(sampled_x)
            sampled_x_repr = torch.nn.functional.normalize(sampled_x_repr, dim=-1)

        # Compute optimization bounds
        bounds = get_bounds(ncands=args.repr_hull_ncands,
                            warmstart_repr=sampled_x_repr,
                            candidates=unlabeled_cands,
                            get_repr_args={
                                "repr_model": repr_model,
                                "pooling": args.repr_llm_pooling,
                                "normalize": args.normalize_inputs,
                                "prompt": args.repr_prompt,
                                "proj_matrix": proj_matrix,
                                "diff_repr": diff_repr,
                                "device": device
                            },
                            square_hull=args.opt_square_hull,
                            hull_margin=args.opt_hull_margin,
                            device=device)

    # Observe black-box values for the warmstart candidates
    if sampled_y is None:
        sampled_y, valid_samples, _ = get_bbox_values(x=sampled_x, task=task, bbox_model=bbox_model,
                                                      prompt_cls=prompt_cls,
                                                      mode="train", llm_log=LLM_LOGS[seed]["bbox"], bbox_log=bbox_logs,
                                                      n_tasks=args.bbox_n_train, device=device)
    else:
        valid_samples = sampled_y != INV_SCORE
        if args.set_invalid_to_zero:
            sampled_y[~valid_samples] *= 0.

    n_invalid_warmstart_cands = 0
    if args.surr_skip_invalid_candidates:
        sampled_x = [sampled_x[i] for i in range(len(sampled_x)) if valid_samples[i].item()]
        sampled_y = sampled_y[valid_samples]
        if requires_bo:
            sampled_x_repr = sampled_x_repr[valid_samples]
        n_invalid_warmstart_cands = len(valid_samples) - valid_samples.sum().item()
        if n_invalid_warmstart_cands > 0:
            print(f"Skipped {n_invalid_warmstart_cands} invalid warmstart candidates.")

    assert len(sampled_x) > 0
    if len(sampled_x) != n_warmstart:
        n_warmstart = len(sampled_x)
        print(f"Warning: Setting n_warmstart to {len(sampled_x)}")

    # Initialize surrogate with warmstart candidates (learns hyperparameters for the priors using MLL)
    surrogate = None
    if requires_bo:
        add_kwargs = {
            "ard_num_dims": sampled_x_repr.shape[-1] if args.kernel_per_dim_lengthscale else 1,
            **{k: v for k, v in {"matern_nu": args.kernel_matern_nu,
                                 "lengthscale": args.kernel_lengthscale,
                                 "outputscale": args.kernel_outputscale,
                                 "mean": args.kernel_mean,
                                 "lengthscale_prior_concentration": args.kernel_lengthscale_prior_concentration,
                                 "lengthscale_prior_rate": args.kernel_lengthscale_prior_rate,
                                 "outputscale_prior_concentration": args.kernel_outputscale_prior_concentration,
                                 "outputscale_prior_rate": args.kernel_outputscale_prior_rate,
                                 "period_length_prior_mean": args.kernel_period_length_prior_mean,
                                 "period_length_prior_std": args.kernel_period_length_prior_std,
                                 "mean_prior_mean": args.kernel_mean_prior_mean,
                                 "mean_prior_std": args.kernel_mean_prior_std,
                                 "ladder_kernel": args.ladder_kernel}.items() if v is not None}
        }
        # We set the CLI arg values to -100 if None needs to be passed for the kernel hyperparameters
        add_kwargs = {k: (v if v != -100 else None) for k, v in add_kwargs.items()}

        surrogate = get_surrogate(args.surrogate_fn, sampled_x_repr, sampled_y, train_yvar=args.gp_noise_var,
                                  task="arc", gp_kernel=args.gp_kernel,
                                  normalize_inputs=args.surr_normalize_inputs_botorch,
                                  standardize_outputs=args.surr_standardize_outputs,
                                  bounds=bounds, dtype=args.dtype,
                                  device=device, **add_kwargs)

        # Track posterior values for visualization
        posterior_vals, viz_observed = {}, []
        if args.visualize_posterior:
            viz_cands, viz_repr, viz_scores = get_visualization_data(ncands=args.visualize_posterior_ncands,
                                                                     candidates=unlabeled_cands,
                                                                     candidate_scores=unlabeled_cands_scores,
                                                                     candidates_fpath=args.visualize_posterior_fname,
                                                                     target=target,
                                                                     get_repr_args={
                                                                         "repr_model": repr_model,
                                                                         "pooling": args.repr_llm_pooling,
                                                                         "normalize": args.normalize_inputs,
                                                                         "prompt": args.repr_prompt,
                                                                         "proj_matrix": proj_matrix,
                                                                         "diff_repr": diff_repr,
                                                                         "device": device
                                                                     },
                                                                     task=task, bbox_model=bbox_model,
                                                                     prompt_cls=prompt_cls, seed=seed, device=device,
                                                                     bbox_logs=bbox_logs, LLM_LOGS=LLM_LOGS)

    # Prepare for the BO loop
    opt_batch_size = args.opt_batch_size
    vec2text_batch_size = args.vec2text_batch_size
    n_timesteps = args.n_evaluations // opt_batch_size // vec2text_batch_size
    n_evaluations = n_timesteps * opt_batch_size * vec2text_batch_size
    print(f"\nRunning {n_timesteps} iterations ({n_evaluations} evaluations)")
    best_idx = sampled_y.argmax().item()
    best_x = warmstart_best_x = sampled_x[best_idx]
    best_y = warmstart_best_y = sampled_y[best_idx].item()
    trace_best = [best_idx]
    steps_to_opt = -1
    n_invalid_decodings = 0
    n_invalid_bbox = 0
    n_repeat_cands = 0
    n_repeat_decoded_cands = 0
    n_repeats_solved_by_perturbing = 0
    n_fixed_bbox = 0
    fixed_bbox_trial = []
    revision_improvements = []
    vec2text_sims = []
    per_iteration_logs = []
    repeat_streak = 0
    n_repeat_streak_reached = 0
    n_throttling_errors = 0
    opt_found = (sampled_y.max() >= OPT_SCORE).item()

    # Log initial posterior mean and std for visualization
    if requires_bo and args.visualize_posterior:
        dataloader = data_utils.DataLoader(data_utils.TensorDataset(viz_repr, viz_scores), batch_size=256)
        f_vals = []
        for x, y in dataloader:
            posterior = surrogate.posterior(x.to(device))
            with torch.no_grad():
                f_vals.append(torch.stack(
                    (y.to(device), posterior.mean.squeeze(), posterior.variance.sqrt().squeeze()), dim=-1))
        f_vals = torch.cat(f_vals, dim=0).tolist()
        posterior_vals[0] = f_vals
        if len(viz_observed) == 0:
            viz_observed.append(list(zip(sampled_x, sampled_y.tolist())))  # add warmstart observations

    print(f"""\nBest initial candidate (y={warmstart_best_y:.3f}):
DOCSTRING:
{warmstart_best_x[0]}
CODE:
{warmstart_best_x[1]}\n""")
    pbar = tqdm.trange(n_timesteps, file=sys.stdout)
    pbar.set_description(
        f'[Best f(x="{best_x[0][:8]}...")={best_y:.3f}]'
    )

    # BayesOpt loop
    for t in pbar:
        if opt_found:
            steps_to_opt = t
            break

        LLM_LOGS[seed]["bo"][t] = []

        # Get the acquisition function
        acq_fn = get_acq_fn(acquisition_fn=args.acquisition_fn,
                            surrogate=surrogate,
                            best_y=best_y,
                            d=sampled_x_repr.shape[-1] if requires_bo else None,
                            acq_ucb_beta=args.acq_ucb_beta,
                            batch_size=opt_batch_size)

        if requires_bo:
            # Log posterior mean and std for visualization
            if args.visualize_posterior:
                dataloader = data_utils.DataLoader(data_utils.TensorDataset(viz_repr, viz_scores), batch_size=256)
                f_vals = []
                for x, y in dataloader:
                    posterior = surrogate.posterior(x.to(device))
                    with torch.no_grad():
                        f_vals.append(torch.stack(
                            (y.to(device), posterior.mean.squeeze(), posterior.variance.sqrt().squeeze()), dim=-1))
                f_vals = torch.cat(f_vals, dim=0).tolist()
                posterior_vals[t] = f_vals
                if len(viz_observed) == 0:
                    viz_observed.append(list(zip(sampled_x, sampled_y.tolist())))  # add warmstart observations

            # Optimize acquisition function
            proposed_x_reprs, proposed_acq_vals = optimize_acq_fn(acq_fn, d=sampled_x_repr.shape[-1],
                                                                  opt_num_restarts=args.opt_num_restarts,
                                                                  opt_q=opt_batch_size,
                                                                  return_best_only=args.opt_return_best_only,
                                                                  bounds=bounds, device=device)
            proposed_x_reprs = proposed_x_reprs.to(sampled_x_repr.dtype)
            proposed_acq_vals = torch.cat([proposed_acq_vals.view(-1)] * opt_batch_size, dim=0)

            # Log acq and posterior values
            per_iteration_logs.append({
                "target_posterior_mean": None,
                "target_posterior_std": None,
                "target_acq_val": None,
                "proposed_acq_val": proposed_acq_vals[0].item(),
                "proposed_2_target_sims": None
            })
            if args.debug:
                print(json.dumps(per_iteration_logs[-1], indent=2))

        # Iterate over each proposed candidate
        posterior_update_x, posterior_update_y = [], []
        decoded_cands_in_batch = []
        decoded_cands = []

        # Decode representation
        if args.acquisition_fn == "OPRO":
            arg_sampled_y = sampled_y
        elif args.acquisition_fn == "random":
            arg_sampled_y = sampled_y[torch.randperm(len(sampled_y))]
        elif args.acquisition_fn == "none":
            arg_sampled_y = sampled_y  # won't be used
        else:
            assert requires_bo
            arg_sampled_y = None
            if args.use_sim_for_selection:
                # Setting where the BO vector is used only for selection and the prompt contains black-box fn scores
                arg_sampled_y = sampled_y

        for _i in range(opt_batch_size):
            if requires_bo:
                proposed_x_repr, proposed_acq_val = proposed_x_reprs[_i], proposed_acq_vals[_i]
                # Check if the proposed vector has been seen before
                proposed_seen = any(torch.allclose(proposed_x_repr, _tensor) for _tensor in sampled_x_repr)
                if proposed_seen:
                    n_repeat_cands += 1
                    if args.verbose:
                        print("Warning: Duplicate BO proposal.")

        assert args.vec2text_n_parallel >= opt_batch_size
        _n_parallel_calls = int((opt_batch_size * args.vec2text_batch_size) / max(1, args.vec2text_batch_size / (
                args.vec2text_n_parallel / opt_batch_size)))
        _counters_for_decoding = [{"repeat_streak": repeat_streak,
                                   "n_repeat_streak_reached": n_repeat_streak_reached,
                                   "n_repeats_solved_by_perturbing": n_repeats_solved_by_perturbing,
                                   "n_throttling_errors": n_throttling_errors} for _ in range(_n_parallel_calls)]
        with ThreadPoolExecutor(max_workers=args.vec2text_n_parallel) as executor:
            # Pack counters for decoding (housekeeping)
            futures = [executor.submit(lambda: decode_repr(
                proposed_x_repr=proposed_x_reprs[__i // (_n_parallel_calls // opt_batch_size)] if requires_bo else None,
                sampled_x=sampled_x,
                sampled_x_repr=sampled_x_repr, prompt_cls=prompt_cls,
                n_cands=int(max(1, args.vec2text_batch_size / (
                        args.vec2text_n_parallel / opt_batch_size))),
                task_demos=task["train"][:args.bbox_n_train],
                sampled_y=arg_sampled_y, gen_model=gen_model,
                llm_log=LLM_LOGS[seed]["bo"][t],
                use_sim_for_selection=args.use_sim_for_selection,
                requires_bo=requires_bo, device=device,
                counters=_counters_for_decoding[__i])) for __i in
                       range(_n_parallel_calls)]
            for future in as_completed(futures):
                try:
                    _decoded_cands = future.result()
                    if _decoded_cands is not None:
                        decoded_cands += _decoded_cands
                except:
                    traceback.print_exc()
        if len(decoded_cands) == 0:
            decoded_cands = None
        # Unpack counters
        for _counters in _counters_for_decoding:
            repeat_streak += _counters["repeat_streak"]
            n_repeat_streak_reached += _counters["n_repeat_streak_reached"]
            n_repeats_solved_by_perturbing += _counters["n_repeats_solved_by_perturbing"]
            n_throttling_errors += _counters["n_throttling_errors"]

        if decoded_cands is None:
            n_invalid_decodings += 1
            if args.skip_on_vec2text_errors:
                if args.verbose:
                    print("Warning: Unable to generate candidates. Skipping.")
                continue
            raise ValueError("Failed to decode proposals.")

        for decoded_cand in decoded_cands:
            proposed_x, proposed_x_repr, original_proposed_repr = decoded_cand
            # Observe black-box value
            proposed_y, is_valid_proposal, proposed_preds, bbox_error_msg = get_bbox_values(x=proposed_x, task=task,
                                                                                            bbox_model=bbox_model,
                                                                                            prompt_cls=prompt_cls,
                                                                                            mode="train",
                                                                                            bbox_log=bbox_logs,
                                                                                            llm_log=LLM_LOGS[seed][
                                                                                                "bbox"],
                                                                                            n_tasks=args.bbox_n_train,
                                                                                            return_predictions=True,
                                                                                            device=device)
            if args.vec2text_fix_retries > 0 and not (is_valid_proposal := is_valid_proposal[0].item()):
                proposed_x, proposed_x_repr, proposed_y, proposed_preds, is_valid_proposal = fix_candidate_wrapper(
                    proposed_x=proposed_x, proposed_x_repr=proposed_x_repr, proposed_y=proposed_y,
                    proposed_preds=proposed_preds, is_valid_proposal=is_valid_proposal, gen_model=gen_model,
                    bbox_model=bbox_model, is_codegen=args.bbox_model == "codegen", prompt_cls=prompt_cls,
                    task=task, bbox_n_train=args.bbox_n_train, bbox_error_msg=bbox_error_msg[0][0],
                    fix_retries=args.vec2text_fix_retries, fix_llm_log=LLM_LOGS[seed]["bo"][t], bbox_log=bbox_logs,
                    bbox_llm_logs=LLM_LOGS[seed]["bbox"], requires_bo=requires_bo,
                    fixed_bbox_trial=fixed_bbox_trial,
                    device=device, verbose=args.verbose)
                if is_valid_proposal:
                    n_fixed_bbox += 1

            # Attempt to revise the code (in case the code does not match the docstring in terms of logic)
            if is_valid_proposal and args.vec2text_revise_retries > 0 and proposed_y.item() < OPT_SCORE:
                proposed_x, proposed_x_repr, proposed_y, proposed_preds = revise_candidate_wrapper(
                    proposed_x=proposed_x, proposed_x_repr=proposed_x_repr, proposed_y=proposed_y,
                    proposed_preds=proposed_preds, gen_model=gen_model, bbox_model=bbox_model,
                    revise_retries=args.vec2text_revise_retries, prompt_cls=prompt_cls, task=task,
                    bbox_n_train=args.bbox_n_train, is_codegen=args.bbox_model == "codegen",
                    fix_llm_log=LLM_LOGS[seed]["bo"][t], bbox_log=bbox_logs, bbox_llm_log=LLM_LOGS[seed]["bbox"],
                    revision_improvements=revision_improvements, optimum_score=OPT_SCORE,
                    device=device, verbose=args.verbose)

            if requires_bo:
                if proposed_x_repr is None:
                    proposed_x_repr = get_representations(proposed_x,
                                                          repr_model=repr_model,
                                                          pooling=args.repr_llm_pooling,
                                                          normalize=args.normalize_inputs,
                                                          prompt=args.repr_prompt,
                                                          proj_matrix=proj_matrix,
                                                          diff_repr=diff_repr,
                                                          device=device)
                vec2text_sim = cosine_similarity(proposed_x_repr[None, :],
                                                 original_proposed_repr[None, :]).item()
                vec2text_sims.append(vec2text_sim)

            # Check for duplicates
            if is_duplicate(proposed_x, sampled_x):
                repeat_streak += 1
                n_repeat_decoded_cands += 1
            else:
                repeat_streak = 0

            if proposed_y.item() > best_y:
                print(f"""\n\nNew best (y={proposed_y.item():.3f}) at t={t + 1}:
DOCSTRING:
{proposed_x[0]}
CODE: 
{proposed_x[1]}\n""")

            pbar.set_description(
                f'[Best f(x="{best_x[0][:8]}...")={best_y:.3f}]; Curr f(x="{proposed_x[0][:8]}...")={proposed_y.item():.3f}]'
            )

            if not is_valid_proposal:
                n_invalid_bbox += 1
                if args.surr_skip_invalid_candidates:
                    print(f"\nSkipping proposal (invalid code execution)\n")
                    continue

            # Add to trajectory
            sampled_x.append(proposed_x)
            decoded_cands_in_batch.append((proposed_x, proposed_y.item()))
            sampled_y = torch.cat([sampled_y, proposed_y.view(-1)], dim=0)
            if requires_bo:
                sampled_x_repr = torch.cat([sampled_x_repr, proposed_x_repr.unsqueeze(0)], dim=0)
            best_idx = sampled_y.argmax().item()
            best_x = sampled_x[best_idx]
            best_y = sampled_y[best_idx].item()
            trace_best.append(best_idx)

            if requires_bo:
                posterior_update_x.append(proposed_x_repr.view(1, -1))
                posterior_update_y.append(proposed_y.view(-1))

            # Check if target is found
            if best_y >= OPT_SCORE:
                opt_found = True
                steps_to_opt = t + 1
                break

        # Update surrogate posterior with (x, y) batch
        if requires_bo:
            viz_observed.append(decoded_cands_in_batch)
            if len(posterior_update_x) > 0 and not opt_found:
                posterior_update_x = torch.cat(posterior_update_x, dim=0)
                posterior_update_y = torch.cat(posterior_update_y, dim=0).view(-1, 1)
                if not args.ladder_kernel:
                    condition_args = {}
                    if args.gp_noise_var is not None:
                        condition_args["noise"] = torch.full_like(posterior_update_y, args.gp_noise_var).to(device).to(
                            TORCH_DTYPE[args.dtype]) + 1e-6
                    surrogate = surrogate.condition_on_observations(
                        posterior_update_x.to(device).to(TORCH_DTYPE[args.dtype]),
                        posterior_update_y.to(device).to(TORCH_DTYPE[args.dtype]),
                        **condition_args
                    )
                else:
                    surrogate = get_surrogate(args.surrogate_fn, sampled_x_repr, sampled_y,
                                              train_yvar=args.gp_noise_var,
                                              task="arc", gp_kernel=args.gp_kernel,
                                              normalize_inputs=args.surr_normalize_inputs_botorch,
                                              standardize_outputs=args.surr_standardize_outputs,
                                              bounds=bounds, dtype=args.dtype,
                                              device=device, **add_kwargs)

    pbar.close()
    time_end = time.time()

    if opt_found:
        print(f'OPTIMUM FOUND at t={steps_to_opt}')

    if requires_bo and args.visualize_posterior:
        os.makedirs(out_dir, exist_ok=True)
        posterior_path = os.path.join(out_dir, f'seed-{seed}_posterior.json')
        plot_posterior(posterior_vals=posterior_vals, obs_xy=viz_observed, posterior_cands=viz_cands,
                       animate=args.visualize_posterior_anim, anim_interval=300, anim_repeat=True,
                       path=posterior_path)
        with open(posterior_path, 'w') as fh:
            fh.write(json.dumps({
                "y_mean_std": posterior_vals,
                "obs_xy": viz_observed
            }, indent=2))

    # Test best x on test set
    test_scores, test_scores_valid, _ = get_bbox_values(x=best_x, task=task, bbox_model=bbox_model,
                                                        prompt_cls=prompt_cls,
                                                        mode="test", bbox_log=bbox_logs,
                                                        llm_log=LLM_LOGS[seed]["bbox"],
                                                        device=device)
    test_scores = test_scores.tolist()
    test_predictions = bbox_logs[-len(test_scores):]

    rtn = {
        "task": args.task,
        "target": target.split('.')[0],
        "seed": seed,
        "n_warmstart": n_warmstart,
        "n_evaluations": n_evaluations,
        "opt_batch_size": opt_batch_size,
        "vec2text_batch_size": vec2text_batch_size,
        "n_timesteps": n_timesteps,
        "surrogate_fn": args.surrogate_fn,
        "acquisition_fn": args.acquisition_fn + ("-lmx" if args.vec2text_lmx else ""),
        "vec2text_sim_mean": round(np.mean(vec2text_sims), 4),
        "vec2text_sim_std": round(np.std(vec2text_sims), 4),
        "time_elapsed": time_end - time_start,
        "opt_found": opt_found,
        "steps_to_opt": steps_to_opt,
        "n_proposals": len(sampled_x),
        "n_invalid_warmstart_cands": n_invalid_warmstart_cands,
        "n_invalid_decodings": n_invalid_decodings,
        "n_invalid_bbox": n_invalid_bbox,
        "n_fixed_bbox": n_fixed_bbox,
        "fixed_bbox_trial": fixed_bbox_trial,
        "avg_revision_improvement": round(np.mean(revision_improvements), 4),
        "n_repeat_cands": n_repeat_cands,
        "n_repeat_decoded_cands": n_repeat_decoded_cands,
        "n_repeat_streak_reached": n_repeat_streak_reached,
        "n_repeats_solved_by_perturbing": n_repeats_solved_by_perturbing,
        "n_throttling_errors": n_throttling_errors,
        "target_in_bounds": target_in_bounds,
        "warmstart_best_xy": (warmstart_best_x, warmstart_best_y),
        "warmstart_avg_y": sampled_y[:n_warmstart].mean().item(),
        "best_xy": (best_x, best_y),
        "gain_from_warmstart": best_y - warmstart_best_y,
        "test_score": round(np.mean(test_scores), 4),
        "test_predictions": test_predictions,
        "trace_best_xy": [(sampled_x[t], sampled_y[t].item()) for t in trace_best],
        "trace_xy": list(zip(sampled_x, sampled_y.tolist())),
        "warmstart_xy": sorted(list(zip(sampled_x[:n_warmstart], sampled_y[:n_warmstart].tolist())),
                               key=lambda x: x[1]),
        "vec2text_sims": vec2text_sims,
        "per_iteration": per_iteration_logs
    }

    if save_results:
        os.makedirs(out_dir, exist_ok=True)
        # Save results
        with open(os.path.join(out_dir, f"seed-{seed}.json"), "w") as fh:
            fh.write(json.dumps(rtn, indent=2))
        # Save bbox logs
        with open(os.path.join(out_dir, f"seed-{seed}_bbox.json"), "w") as fh:
            fh.write(json.dumps(bbox_logs, indent=2))
        print(f"Results saved to {out_dir}/seed-{seed}.json")

    # Return results
    return rtn


def run_bo(target, n_runs=1, seed=17, device='cuda'):
    # Optionally, set defaults
    if args.use_method_defaults:
        print(f"Using method defaults for: {args.acquisition_fn}")
        if args.acquisition_fn in NON_BO_ACQS:
            args.vec2text_normalize = False
            args.vec2text_target_score = True
            args.vec2text_exploit_threshold = False
            print("Force set vec2text_normalize, vec2text_exploit_threshold to False; vec2text_target_score to True")
            if args.acquisition_fn == "none":
                args.arc_no_demos = True
                print("Force set arc_no_demos to True")
        else:
            # BO
            args.vec2text_normalize = True
            args.vec2text_target_score = True
            args.vec2text_exploit_threshold = False
            print(
                "Force set vec2text_normalize, vec2text_target_score to True and vec2text_exploit_threshold to False")
        # LMX
        if args.vec2text_lmx:
            args.vec2text_rand_samples = args.vec2text_demos
            args.vec2text_demos *= 2
            args.arc_use_scores = False
            args.arc_use_mixing = True
            print("Force set vec2text_rand_samples to vec2text_demos, vec2text_demos *= 2, arc_use_scores to False, "
                  "arc_use_mixing to True")
    requires_bo = args.acquisition_fn not in NON_BO_ACQS

    # Init models
    gen_model = get_gen_model(args.gen_model, enable_cot=args.llm_enable_cot, codegen=args.bbox_model == "codegen")
    repr_model = get_repr_model(args.repr_model) if requires_bo else None
    bbox_model = get_bbox_model(args.bbox_model, enable_cot=args.llm_enable_cot)
    prompt_cls = Prompts.ARCCode if args.bbox_model == "codegen" else Prompts.ARC

    # Set defaults (temp)
    if args.bbox_model == "codegen":
        args.repr_prompt = "%s"
    else:
        args.repr_prompt = "Transform a 2D input grid into a 2D output grid using the following algorithm:" + \
                           NEWLINE + "%s"

    if target is None:
        targets = os.listdir(args.task_fpath)[args.targets_start_idx:][:args.n_targets]
    else:
        targets = list(map(lambda x: f"{x}.json", target.split(",")))[:args.n_targets]

    print(f"\nStarting runs for {len(targets)} target(s)...")

    global out_dir
    base_out_dir = out_dir
    global LLM_LOGS
    error_targets = []
    all_summary = {}
    for target in targets:
        LLM_LOGS = {}
        try:
            out_dir = os.path.join(base_out_dir, target.split('.')[0])
            # Get task
            print(f"\nLoading the task from {args.task_fpath}: {target}")
            with open(os.path.join(args.task_fpath, target)) as fh:
                task = json.load(fh)
            task = {
                "train": [(t["input"], t["output"]) for t in task["train"]],
                "test": [(t["input"], t["output"]) for t in task["test"]]
            }
            # Print the input-output 2D grids from task["train"] in a nice format to show what the task is
            for i in range(len(task["train"])):
                print(f"INPUT {i + 1}:")
                print(stringify_grid(task["train"][i][0]))
                if args.arc_show_transpose:
                    print(f"INPUT {i + 1} (transposed):")
                    print(stringify_grid(task["train"][i][0], transpose=True))
                print(f"OUTPUT {i + 1}:")
                print(stringify_grid(task["train"][i][1]))
                print()
            # Check if candidates file exists
            candidates_fname = None
            if args.candidates_fname is not None:
                if args.candidates_fname.endswith(".json"):
                    candidates_fname = args.candidates_fname
                else:
                    candidates_task_fname = os.path.join(args.candidates_fname, target.split('.')[0])
                    candidates_fname = [f for f in os.listdir(candidates_task_fname) if
                                        (f.startswith("candidates") and f.endswith(".json"))]
                    candidates_fname = os.path.join(candidates_task_fname, candidates_fname[0]) \
                        if len(candidates_fname) > 0 else None
            results = []
            for i in range(n_runs):
                res = _bo(target=target, gen_model=gen_model, repr_model=repr_model, bbox_model=bbox_model,
                          candidates_fname=candidates_fname, prompt_cls=prompt_cls, task=task, seed=seed + i,
                          requires_bo=requires_bo, device=device)
                if res is not None:
                    results.append(res)
                save_llm_logs(messages=LLM_LOGS, log_dir=out_dir)
            if len(results) > 0:
                print()
                agg_res = aggregate_results(results)
                # Print results
                summary = {k: v for k, v in agg_res.items() if type(v) not in [list, dict]}
                all_summary[target] = summary
                print("\n" + json.dumps(summary, indent=2))
                print("----------------------------------------------\n")
        except:
            error_targets.append(target)
            print(f"\nError occurred for target: {target}")
            traceback.print_exc()
            continue

    out_dir = base_out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "error_targets.json"), "w") as fh:
        fh.write(json.dumps(error_targets, indent=2))

    if len(all_summary) > 1:
        print("\n\nSummary:\n" + json.dumps(all_summary, indent=2))

    if len(error_targets) > 0:
        print(f"\nErrors occurred for the following targets:\n{error_targets}")


if __name__ == '__main__':
    # Setup
    argparser = ArgParser(group="semantle")
    global args
    args = argparser.parse_args()  # TODO: Add ability to read args from a file
    print("Script arguments:")
    print(args.__dict__)
    global RUN_ID
    RUN_ID = str(int(time.time())) if args.run_id is None else args.run_id
    global out_dir
    out_dir = os.path.join(args.out_dir, RUN_ID)
    print(f'\nOutput directory: {out_dir}\n')
    # Log LLM calls
    global LLM_LOGS
    LLM_LOGS = {}
    atexit.register(lambda: save_llm_logs(messages=LLM_LOGS, log_dir=out_dir))
    global BEDROCK_CLIENT
    BEDROCK_CLIENT = get_bedrock_client()
    global OPENAI_CLIENT
    OPENAI_CLIENT = get_openai_client()
    global NON_BO_ACQS
    NON_BO_ACQS = ["OPRO", "random", "none"]

    # Main loop
    run_bo(target=args.target, n_runs=args.n_seeds, seed=args.seed, device=args.device)

    if args.debug:
        breakpoint()
