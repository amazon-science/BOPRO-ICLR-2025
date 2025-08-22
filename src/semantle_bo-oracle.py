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
from gensim.downloader import load as load_word2vec
import torch.utils.data as data_utils
from scipy.special import softmax

from utils.arguments import ArgParser
from utils.bo import get_surrogate, optimize_acq_fn, get_acq_fn, plot_posterior, plot_trace
from utils.generation import HF_PIPELINE_MODELS, BEDROCK_CLAUDE_MODELS, SYS_PROMPT, \
    SYS_PROMPT_COT, BEDROCK_LLAMA_MODELS, BEDROCK_MISTRAL_MODELS, get_seq_from_repr, get_candidates, HF_SFORMER_MODELS, \
    BEDROCK_COHERE_MODELS, invoke_bedrock_embeddings
from utils.misc import save_llm_logs, sample_by_strategy, TORCH_DTYPE
from utils import prompts as Prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"
OPT_SCORE = 1.0
INV_SCORE = -1.0


def get_bbox_model(bbox_model_name):
    if bbox_model_name == "word2vec":
        return load_word2vec("word2vec-google-news-300")
    elif bbox_model_name == "simcse":
        return SimCSE("princeton-nlp/sup-simcse-roberta-large")
    else:
        # LLMs
        if bbox_model_name in HF_PIPELINE_MODELS:
            hf_model = HF_PIPELINE_MODELS[bbox_model_name]
            model_pipeline = pipeline(
                model=hf_model,
                task="feature-extraction",
                device_map="auto",
                token=args.hf_access_token
            )
            return model_pipeline
        else:
            raise NotImplementedError


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


def get_gen_model(gen_model_name, enable_cot=False):
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
            "system_prompt": SYS_PROMPT if not enable_cot else SYS_PROMPT_COT,
            "system_prompt_feedback": SYS_PROMPT
        }
    elif gen_model_name in BEDROCK_MODELS:
        model = BEDROCK_MODELS[gen_model_name]
        rtn = {
            "model": model,
            "is_hf": False,
            "decoding_args": {
                "max_tokens": args.llm_tokens,
                "temperature": args.llm_temperature,
                "top_p": args.llm_top_p
            },
            "system_prompt": SYS_PROMPT if not enable_cot else SYS_PROMPT_COT,
            "system_prompt_feedback": SYS_PROMPT
        }
    else:
        raise NotImplementedError
    return rtn


def get_prompts_for_repr(x, prompt="%s"):
    # Prompt examples:
    # "%s"
    # "The task is to guess a hidden test word. The next guess is %s."
    return [prompt % _x for _x in (x if type(x) is list else [x])]


def get_representations(x, repr_model, pooling="mean", prompt="%s", proj_matrix=None, normalize=False,
                        diff_repr=None, device='cuda', BEDROCK_CLIENT=None):
    prompts = get_prompts_for_repr(x, prompt=prompt)
    if "simcse" in type(repr_model).__name__.lower():
        _repr = repr_model.encode(prompts, silent=True)
    elif type(repr_model) is str and repr_model in list(BEDROCK_COHERE_MODELS.values()):
        # Bedrock models
        _repr = invoke_bedrock_embeddings(repr_model, prompts, bedrock_client=BEDROCK_CLIENT)
    elif type(repr_model).__name__ == "SentenceTransformer":
        _repr = repr_model.encode(prompts,
                                  prompt=f"""Instruct: Given the following English-language word,
retrieve only those words that are similar to it in meaning.\nWord: """,
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


def get_bbox_values(x=None, target=None, bbox_model=None, x_repr=None, target_repr=None):
    if bbox_model is not None:
        if type(bbox_model).__name__ == "KeyedVectors":  # word2vec
            _x = x if type(x) is list else [x]
            vals = []
            valid = []
            for __x in _x:
                try:
                    vals.append(bbox_model.similarity(target, __x))
                    valid.append(True)
                except KeyError:
                    if args.verbose:
                        print(f"Warning: Word '{__x}' not in the word2vec vocabulary. Assigning score=0.")
                    vals.append(0)
                    valid.append(False)
            return torch.tensor(vals).squeeze(), torch.tensor(valid).squeeze()

        x_repr = get_representations(x=x,
                                     repr_model=bbox_model,
                                     pooling=args.repr_llm_pooling,
                                     prompt=args.repr_prompt if args.bbox_prompt is None else args.bbox_prompt,
                                     device=args.device)
        target_repr = get_representations(x=target,
                                          repr_model=bbox_model,
                                          pooling=args.repr_llm_pooling,
                                          prompt=args.repr_prompt if args.bbox_prompt is None else args.bbox_prompt,
                                          device=args.device)

    return cosine_similarity(target_repr,
                             x_repr if len(x_repr.shape) == 2 else x_repr[None, :]).squeeze(), torch.tensor(
        [True] * (1 if len(x_repr.shape) == 1 else x_repr.shape[0])).squeeze()


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
        # torch.nn.init.normal_(proj_mat, 0, 1)
        torch.nn.init.uniform_(proj_mat, -1, 1)
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


def get_visualization_data(ncands, candidates, candidate_scores, get_repr_args):
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


# def decode_repr(proposed_x_repr, sampled_x, sampled_x_repr, sampled_y, gen_model, repr_model, proj_matrix,
#                 prompt_cls, gains_from_vec2text_trials, llm_log, requires_bo, device):
#     per_trial = []
#     for _trial in range(args.vec2text_trials):
#         try:
#             decoded_x = get_seq_from_repr(proposed_x_repr, sampled_x, sampled_x_repr, gen_model, prompt_cls,
#                                           sampled_y=sampled_y, generate_feedback=args.generate_feedback,
#                                           n_cands=1, topk=args.vec2text_demos, normalize_scores=args.vec2text_normalize,
#                                           n_low_previous=args.opro_n_low_previous,
#                                           sample_unique=args.vec2text_sample_unique,
#                                           max_retry=args.vec2text_unique_retries,
#                                           multiturn_retry=args.vec2text_multiturn_retry,
#                                           use_target_score=args.vec2text_target_score,
#                                           use_exploit_threshold=args.vec2text_exploit_threshold,
#                                           prev_trials=sorted(per_trial, key=lambda x: x[-1]),
#                                           llm_log=llm_log, human_input=args.diagnostic_human_input,
#                                           no_demos=args.arc_no_demos)[0]
#         except Exception as e:
#             print("Error: Failed to decode representation.")
#             traceback.print_exc()
#             if args.debug:
#                 breakpoint()
#             elif args.skip_on_vec2text_errors:
#                 print("Skipping this trial.")
#                 continue
#             else:
#                 raise e
#
#         # Encode decoded sequence
#         decoded_x_repr = get_representations(decoded_x,
#                                              repr_model=repr_model,
#                                              pooling=args.repr_llm_pooling,
#                                              normalize=args.normalize_inputs,
#                                              prompt=args.repr_prompt,
#                                              proj_matrix=proj_matrix,
#                                              device=device)
#         # Measure error between the BO proposal and the decoded representation
#         vec2text_sim = cosine_similarity(proposed_x_repr[None, :],
#                                          decoded_x_repr[None, :]).item() if requires_bo else 0.
#         per_trial.append((decoded_x, decoded_x_repr, vec2text_sim))
#
#     if len(per_trial) == 0:
#         return None
#
#     # Choose the best trial
#     decoded_x, decoded_x_repr, vec2text_sim = max(per_trial, key=lambda x: x[-1])
#     gains_from_vec2text_trials.append(vec2text_sim - per_trial[0][-1])
#
#     return decoded_x, decoded_x_repr, vec2text_sim

def is_duplicate(proposed_x, sampled_x):
    return proposed_x in sampled_x


def decode_repr(proposed_x_repr, sampled_x, sampled_x_repr, sampled_y,
                n_cands, gen_model, prompt_cls, llm_log,
                requires_bo, device, counters,
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
                                           n_low_previous=args.opro_n_low_previous,
                                           sample_unique=args.vec2text_sample_unique,
                                           max_retry=args.vec2text_unique_retries,
                                           multiturn_retry=args.vec2text_multiturn_retry,
                                           use_target_score=args.vec2text_target_score,
                                           use_exploit_threshold=args.vec2text_exploit_threshold,
                                           llm_log=llm_log, human_input=args.diagnostic_human_input,
                                           scores=args.arc_use_scores, mix=args.arc_use_mixing,
                                           no_demos=args.arc_no_demos)
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
                                                            n_cands=1, sampled_y=sampled_y, gen_model=gen_model,
                                                            llm_log=llm_log, requires_bo=requires_bo, device=device,
                                                            counters=counters, handle_duplicates=False)
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


def _bo(target, gen_model, repr_model, bbox_model, prompt_cls, task=None, seed=17, device='cuda', save_results=True,
        candidates_fname=None, requires_bo=True):
    print(f"\nSEED: {seed}\n")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LLM_LOGS[seed] = {
        "candidates": [],
        "dim_reduction": [],
        "hull": [],
        "bo": {}
    }
    # TODO: Optionally learn/pre-train the prior

    # Load or generate unlabelled candidates
    n_unlabeled = args.n_unlabeled
    unlabeled_cands, unlabeled_cands_scores = get_candidates(n_cands=n_unlabeled,
                                                             batch_size=args.candidate_gen_batch_size,
                                                             gen_model=gen_model, prompt_cls=prompt_cls,
                                                             strategy="top", target=None,
                                                             candidates_fname=candidates_fname,
                                                             llm_log=LLM_LOGS[seed]["candidates"],
                                                             include_scores=True)

    # Get target representation
    target_repr = get_representations(target,
                                      repr_model=repr_model,
                                      pooling=args.repr_llm_pooling,
                                      prompt=args.repr_prompt,
                                      device=device)
    # Obtain warm-start set
    n_warmstart = args.n_warmstart
    sampled_x = sample_by_strategy(n_cands=n_warmstart, candidates=unlabeled_cands[1:],
                                   strategy=args.warmstart_strategy)
    if len(sampled_x) != n_warmstart:
        n_warmstart = len(sampled_x)
        print(f"Warning: Setting n_warmstart to {len(sampled_x)}")
    sampled_x_repr = get_representations(sampled_x,
                                         repr_model=repr_model,
                                         pooling=args.repr_llm_pooling,
                                         prompt=args.repr_prompt,
                                         device=device)

    # Optionally, get low-dim projection matrix
    proj_matrix = None
    if args.low_dim_strategy is not None and args.low_dim_strategy != "off":
        proj_matrix = get_projection_matrix(low_dim=args.low_dim, strategy=args.low_dim_strategy,
                                            warmstart_repr=sampled_x_repr, candidates=unlabeled_cands[1:],
                                            get_repr_args={
                                                "repr_model": repr_model,
                                                "pooling": args.repr_llm_pooling,
                                                "prompt": args.repr_prompt,
                                                "device": device
                                            }, device=device)
        # Project already computed representations
        sampled_x_repr = sampled_x_repr @ proj_matrix
        target_repr = target_repr @ proj_matrix
    if args.normalize_inputs:
        sampled_x_repr = torch.nn.functional.normalize(sampled_x_repr)
        target_repr = torch.nn.functional.normalize(target_repr.view(1, -1)).squeeze()

    # Compute optimization bounds
    bounds, target_in_bounds = None, None
    if requires_bo or args.acquisition_fn == "random":
        bounds = get_bounds(ncands=0,  # args.repr_hull_ncands,
                            warmstart_repr=sampled_x_repr,
                            candidates=unlabeled_cands[1:],
                            get_repr_args={
                                "repr_model": repr_model,
                                "pooling": args.repr_llm_pooling,
                                "normalize": args.normalize_inputs,
                                "prompt": args.repr_prompt,
                                "proj_matrix": proj_matrix,
                                "device": device
                            },
                            square_hull=args.opt_square_hull,
                            hull_margin=args.opt_hull_margin,
                            device=device)
        # Check if target representation is in bounds
        target_in_bounds = ((target_repr >= bounds[0]) & (target_repr <= bounds[1])).all().item()

    # Observe black-box values for the warmstart candidates
    sampled_y, valid_samples = get_bbox_values(x=sampled_x, target=target, bbox_model=bbox_model,
                                               x_repr=sampled_x_repr, target_repr=target_repr)
    n_invalid_warmstart_cands = 0
    if args.surr_skip_invalid_candidates:
        sampled_x = [sampled_x[i] for i in range(len(sampled_x)) if valid_samples[i].item()]
        sampled_y = sampled_y[valid_samples]
        sampled_x_repr = sampled_x_repr[valid_samples]
        n_invalid_warmstart_cands = len(valid_samples) - valid_samples.sum().item()
        if n_invalid_warmstart_cands > 0:
            print(f"Skipped {n_invalid_warmstart_cands} invalid warmstart candidates.")

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
                                 "mean_prior_std": args.kernel_mean_prior_std}.items() if v is not None}
        }
        # We set the CLI arg values to -100 if None needs to be passed for the kernel hyperparameters
        add_kwargs = {k: (v if v != -100 else None) for k, v in add_kwargs.items()}

        surrogate = get_surrogate(args.surrogate_fn, sampled_x_repr, sampled_y, train_yvar=args.gp_noise_var,
                                  gp_kernel=args.gp_kernel,
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
                                                                     get_repr_args={
                                                                         "repr_model": repr_model,
                                                                         "pooling": args.repr_llm_pooling,
                                                                         "normalize": args.normalize_inputs,
                                                                         "prompt": args.repr_prompt,
                                                                         "proj_matrix": proj_matrix,
                                                                         "device": device
                                                                     })

    # Get representations for unlabeled candidates
    unlabeled_reprs = get_representations(unlabeled_cands,
                                          repr_model=repr_model,
                                          pooling=args.repr_llm_pooling,
                                          normalize=args.normalize_inputs,
                                          prompt=args.repr_prompt,
                                          proj_matrix=proj_matrix,
                                          device=device)

    # Prepare for the BO loop
    opt_batch_size = args.opt_batch_size
    vec2text_batch_size = args.vec2text_batch_size
    n_evaluations = args.n_evaluations
    n_timesteps = args.n_evaluations // opt_batch_size // vec2text_batch_size
    print(f"\nRunning {n_timesteps} iterations ({n_evaluations} evaluations)")
    best_idx = sampled_y.argmax().item()
    best_x = warmstart_best_x = sampled_x[best_idx]
    best_y = warmstart_best_y = sampled_y[best_idx].item()
    trace_best = [best_idx]
    steps_to_opt = -1
    n_invalid_bo_cands = 0
    n_repeat_cands = 0
    n_repeat_decoded_cands = 0
    n_repeats_solved_by_perturbing = 0
    vec2text_sims = []
    per_iteration_logs = []
    gains_from_vec2text_trials = []
    repeat_streak = 0
    n_repeat_streak_reached = 0
    opt_found = (sampled_y.max() >= OPT_SCORE).item()
    n_throttling_errors = 0
    n_invalid_decodings = 0

    print(f"Best initial candidate: {warmstart_best_x} (f={warmstart_best_y:.3f})")
    pbar = tqdm.trange(n_timesteps, file=sys.stdout)
    pbar.set_description(
        f'[Best f(x="{best_x}")={best_y:.3f}]'
    )

    time_start = time.time()
    # BayesOpt loop
    for t in pbar:
        if opt_found:
            break
        LLM_LOGS[seed]["bo"][t] = []

        # Get the acquisition function
        acq_fn = get_acq_fn(acquisition_fn=args.acquisition_fn,
                            surrogate=surrogate,
                            best_y=best_y,
                            d=sampled_x_repr.shape[-1] if requires_bo else None,
                            acq_ucb_beta=args.acq_ucb_beta,
                            batch_size=opt_batch_size)

        # Log posterior mean and std for visualization
        if requires_bo:
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

            # Log the acq value for the target word
            with torch.no_grad():
                target_acq_val = acq_fn(target_repr[None, :].to(device).to(TORCH_DTYPE[args.dtype])).item()
                # Also log the posterior mean and std for the target
                target_posterior = surrogate.posterior(target_repr[None, :].to(device))
                target_posterior_mean = target_posterior.mean.squeeze().item()
                target_posterior_std = target_posterior.variance.sqrt().squeeze().item()
        else:
            # Random
            target_acq_val = 0.
            target_posterior_mean = 0.
            target_posterior_std = 0.

        # Optimize acquisition function
        proposed_x_reprs, proposed_acq_vals = optimize_acq_fn(acq_fn, d=sampled_x_repr.shape[-1],
                                                              opt_num_restarts=args.opt_num_restarts,
                                                              opt_q=opt_batch_size,
                                                              return_best_only=args.opt_return_best_only,
                                                              bounds=bounds, device=device)
        proposed_x_reprs = proposed_x_reprs.to(sampled_x_repr.dtype)
        proposed_acq_vals = torch.cat([proposed_acq_vals.view(-1)] * len(proposed_x_reprs), dim=0)

        if requires_bo:
            # Log acq and posterior values
            proposed_2_target_sims = cosine_similarity(target_repr[None, :], proposed_x_reprs).tolist()
            per_iteration_logs.append({
                "target_posterior_mean": target_posterior_mean,
                "target_posterior_std": target_posterior_std,
                "target_acq_val": target_acq_val,
                "proposed_acq_val": proposed_acq_vals[0].item(),
                "proposed_2_target_sims": proposed_2_target_sims,
            })
            if args.debug:
                print(json.dumps(per_iteration_logs[-1], indent=2))

        # Iterate over each proposed candidate
        posterior_update_x, posterior_update_y = [], []
        decoded_cands_in_batch = []

        # Decode representation using selection from an oracle set instead of with an LLM
        for _i in range(opt_batch_size):
            proposed_x_repr, proposed_acq_val = proposed_x_reprs[_i], proposed_acq_vals[_i]

            if requires_bo:
                # Check if the proposed vector has been seen before
                proposed_seen = any(torch.allclose(proposed_x_repr, _tensor) for _tensor in sampled_x_repr)
                if proposed_seen:
                    n_repeat_cands += 1
                    if args.verbose:
                        print("Warning: Duplicate BO proposal.")

                # Find the candidates that would be included in the prompt
                # We assume these candidates will not be generated
                seen_sims = cosine_similarity(proposed_x_repr[None, :], sampled_x_repr)
                top_seen_idxs = seen_sims.argsort(descending=True).tolist()
                if not args.oracle_only_unseen:
                    # Only prevent the vec2text demos from being regenerated, otherwise prevent all seen
                    top_seen_idxs = top_seen_idxs[:args.vec2text_demos]
                top_seen_cands = [sampled_x[i] for i in top_seen_idxs]
                # Get similarity w.r.t. the BO proposal
                all_sims = cosine_similarity(proposed_x_repr[None, :], unlabeled_reprs)
            elif args.acquisition_fn == "OPRO":
                # Implement an oracle greedy strategy
                # Compute the best set of candidates so far that will not be regenerated
                top_seen_idxs = sampled_y.argsort(descending=True).tolist()
                if not args.oracle_only_unseen:
                    # Only prevent the vec2text demos from being regenerated, otherwise prevent all seen
                    top_seen_idxs = top_seen_idxs[:args.vec2text_demos]
                top_seen_cands = [sampled_x[i] for i in top_seen_idxs]
                # Get similarity w.r.t. the highest scoring candidate
                all_sims = cosine_similarity(sampled_x_repr[top_seen_idxs[0]][None, :], unlabeled_reprs)
            elif args.acquisition_fn == "random":
                top_seen_idxs = sampled_y.argsort(descending=True).tolist()
                if not args.oracle_only_unseen:
                    # Only prevent the vec2text demos from being regenerated, otherwise prevent all seen
                    top_seen_idxs = top_seen_idxs[:args.vec2text_demos]
                top_seen_cands = [sampled_x[i] for i in top_seen_idxs]
                # Get similarity w.r.t. the highest scoring candidate
                all_sims = torch.randn(len(unlabeled_reprs))
            else:
                raise NotImplementedError

            # Create a filtered list of unseen cands and scores from unlabeled_cands and all_sims
            unseen_cands_scores_idxs = [(unlabeled_cands[i], all_sims[i].item(), i) for i in
                                        range(len(unlabeled_cands)) if
                                        not is_duplicate(unlabeled_cands[i], top_seen_cands)]
            target_cand_score_idx = unseen_cands_scores_idxs[0]
            # Sort by similarity (descending)
            unseen_cands_scores_idxs = sorted(unseen_cands_scores_idxs, key=lambda _x: _x[1], reverse=True)
            print(
                f"Rank of target: {unseen_cands_scores_idxs.index(target_cand_score_idx) + 1} (similarity={round(target_cand_score_idx[1], 6)})")

            # Sample batch_size candidates
            if args.oracle_pick_best:
                sampled_idxs = [unseen_cands_scores_idxs[i][2] for i in range(args.vec2text_batch_size)]
            else:
                # Sample weighted by similarity
                sampled_idxs = np.random.choice(
                    [unseen_cands_scores_idxs[i][2] for i in range(len(unseen_cands_scores_idxs))],
                    size=args.vec2text_batch_size,
                    replace=False,
                    p=softmax([unseen_cands_scores_idxs[i][1] / (args.llm_temperature + 1e-6) for i in
                               range(len(unseen_cands_scores_idxs))]))
            decoded_cands = [(unlabeled_cands[i], unlabeled_reprs[i], proposed_x_repr) for i in sampled_idxs]

            if requires_bo:
                # Handle duplicates in the selected random candidates
                if args.repeat_cand_strategy == "bo_proposal":
                    for _i, d in enumerate(decoded_cands):
                        if is_duplicate(d[0], sampled_x):
                            decoded_cands[_i] = (decoded_cands[_i][0], proposed_x_repr, proposed_x_repr)

            # if decoded_cands is None:
            #     n_invalid_decodings += 1
            #     if args.skip_on_vec2text_errors:
            #         if args.verbose:
            #             print("Warning: Unable to generate candidates. Skipping.")
            #         continue
            #     raise ValueError("Failed to decode proposals.")

            for decoded_cand in decoded_cands:
                proposed_x, proposed_x_repr, original_proposed_repr = decoded_cand
                # Observe black-box value
                proposed_y, is_valid_proposal = get_bbox_values(x=proposed_x, target=target, bbox_model=bbox_model,
                                                                x_repr=proposed_x_repr, target_repr=target_repr)

                if requires_bo:
                    if proposed_x_repr is None:
                        proposed_x_repr = get_representations(proposed_x,
                                                              repr_model=repr_model,
                                                              pooling=args.repr_llm_pooling,
                                                              normalize=args.normalize_inputs,
                                                              prompt=args.repr_prompt,
                                                              proj_matrix=proj_matrix,
                                                              device=device)
                    vec2text_sim = cosine_similarity(proposed_x_repr[None, :], original_proposed_repr[None, :]).item()
                    vec2text_sims.append(vec2text_sim)

                # Check for duplicates
                if is_duplicate(proposed_x, sampled_x):
                    repeat_streak += 1
                    n_repeat_decoded_cands += 1
                else:
                    repeat_streak = 0

                if proposed_y.item() > best_y:
                    print(f"""\n\nNEW BEST f(x="{proposed_x}")={proposed_y.item():.3f} AT T={t + 1}\n""")

                pbar.set_description(
                    f'[Best f(x="{best_x}")={best_y:.3f}' + '; ' + f'Curr f(x="{proposed_x}")={proposed_y.item():.3f}]'
                )

                if not is_valid_proposal:
                    n_invalid_bo_cands += 1
                    if args.surr_skip_invalid_candidates:
                        print(f"Skipped invalid proposal.")
                        continue

                # Add to trajectory
                sampled_x.append(proposed_x)
                decoded_cands_in_batch.append((proposed_x, proposed_y.item()))
                sampled_y = torch.cat([sampled_y, proposed_y.view(-1)], dim=0)
                # if requires_bo:  # Need this for oracle mode regardless of if using BO or not
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
                condition_args = {}
                if args.gp_noise_var is not None:
                    condition_args["noise"] = torch.full_like(posterior_update_y, args.gp_noise_var).to(device).to(
                        TORCH_DTYPE[args.dtype]) + 1e-6
                surrogate = surrogate.condition_on_observations(
                    posterior_update_x.to(device).to(TORCH_DTYPE[args.dtype]),
                    posterior_update_y.to(device).to(TORCH_DTYPE[args.dtype]),
                    **condition_args
                )
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
        plot_posterior(posterior_vals=posterior_vals, obs_xy=viz_observed, posterior_cands=viz_cands,
                       animate=args.visualize_posterior_anim, anim_interval=300, anim_repeat=True,
                       path=posterior_path.replace('posterior', 'posterior-500'), top_k=500)
        with open(posterior_path, 'w') as fh:
            fh.write(json.dumps({
                "y_mean_std": posterior_vals,
                "obs_xy": viz_observed
            }, indent=2))

    rtn = {
        "task": args.task,
        "target": target,
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
        "steps_to_opt": steps_to_opt,
        "n_proposals": len(sampled_x),
        "n_invalid_warmstart_cands": n_invalid_warmstart_cands,
        "n_invalid_bo_cands": n_invalid_bo_cands,
        "n_invalid_decodings": n_invalid_decodings,
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
        print(f"Results saved to {out_dir}/seed-{seed}.json")

    # Return results
    return rtn


def aggregate_results(results):
    n_opt = len([r for r in results if r["steps_to_opt"] != -1])
    steps_to_opt = sum([r["steps_to_opt"] for r in results if r["steps_to_opt"] != -1])
    agg_res = {
        "n_runs": args.n_seeds,
        "n_opt": n_opt,
        "avg_opt_rate": round(n_opt / args.n_seeds, 4),
        "avg_steps_to_opt": round(steps_to_opt / n_opt, 4) if n_opt > 0 else None,
        "warmstart_avg_best_y": round(np.mean([r["warmstart_best_xy"][1] for r in results]), 4),
        "avg_best_y": round(np.mean([r["best_xy"][1] for r in results]), 4),
        "std_best_y": round(np.std([r["best_xy"][1] for r in results]), 4),
        "avg_gain_from_warmstart": round(np.mean([r["gain_from_warmstart"] for r in results]), 4),
        "std_gain_from_warmstart": round(np.std([r["gain_from_warmstart"] for r in results]), 4),
        "avg_n_proposals": round(np.mean([r["n_proposals"] for r in results]), 4),
        "avg_invalid_bo_cands": round(np.mean([r["n_invalid_bo_cands"] for r in results]), 4),
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


def run_bo(target, n_runs=1, seed=17, device='cuda'):
    # Optionally, set defaults
    if args.use_method_defaults:
        print(f"Using method defaults for: {args.acquisition_fn}")
        if args.acquisition_fn in NON_BO_ACQS:
            args.vec2text_normalize = False
            args.vec2text_target_score = True
            args.vec2text_exploit_threshold = False
            print("Force set vec2text_normalize, vec2text_exploit_threshold to False; vec2text_target_score to True")
            # Commented below because we want the warmstart set to always be shown in the prompt for repeated sampling
            # if args.acquisition_fn == "none":
            #     args.arc_no_demos = True
            #     print("Force set arc_no_demos to True")
        else:
            # BO
            args.vec2text_normalize = True
            args.vec2text_target_score = True
            args.vec2text_exploit_threshold = False
            print("Force set vec2text_normalize, vec2text_target_score to True; vec2text_exploit_threshold to False")
        # LMX
        if args.vec2text_lmx:
            args.vec2text_rand_samples = args.vec2text_demos
            args.vec2text_demos *= 2
            args.arc_use_scores = False
            args.arc_use_mixing = False  # Setting to true leads to invalid compound word generations
            print(
                "Force set vec2text_rand_samples to vec2text_demos, vec2text_demos *= 2, arc_use_scores to False, "
                "arc_use_mixing to True")
    requires_bo = args.acquisition_fn not in NON_BO_ACQS

    # Init models
    gen_model = get_gen_model(args.gen_model, enable_cot=args.llm_enable_cot)  # Returns a dict
    repr_model = get_repr_model(args.repr_model)  # if requires_bo else None # Commented for oracle setting
    bbox_model = get_bbox_model(args.bbox_model)
    prompt_cls = Prompts.Semantle

    if target is None:
        targets = os.listdir(args.task_fpath)[args.targets_start_idx:][:args.n_targets]
    else:
        targets = list(map(lambda x: f"{x}.csv", target.split(",")))[:args.n_targets]

    print(f"\nStarting runs for {len(targets)} target(s)...")

    global out_dir
    base_out_dir = out_dir
    global LLM_LOGS
    error_targets = []
    all_summary = {}
    for target in targets:
        target_word = target.split('.')[0]
        LLM_LOGS = {}
        try:
            out_dir = os.path.join(base_out_dir, target_word)
            results = []
            for i in range(n_runs):
                res = _bo(target=target_word, gen_model=gen_model, repr_model=repr_model, bbox_model=bbox_model,
                          prompt_cls=prompt_cls, seed=seed + i, requires_bo=requires_bo, device=device,
                          candidates_fname=os.path.join(args.task_fpath, target))
                if res is not None:
                    results.append(res)
                save_llm_logs(messages=LLM_LOGS, log_dir=out_dir)
            if len(results) > 0:
                print()
                agg_res = aggregate_results(results)
                # Print results
                summary = {k: v for k, v in agg_res.items() if type(v) not in [list, dict]}
                all_summary[target_word] = summary
                print("\n" + json.dumps(summary, indent=2))
                print("----------------------------------------------\n")
        except:
            error_targets.append(target_word)
            print(f"\nError occurred for target: {target_word}")
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
    global NON_BO_ACQS
    NON_BO_ACQS = ["OPRO", "random", "none"]

    # Main loop
    run_bo(target=args.target, n_runs=args.n_seeds, seed=args.seed, device=args.device)

    if args.debug:
        breakpoint()
