import argparse

from utils.generation import HF_PIPELINE_MODELS, BEDROCK_CLAUDE_MODELS, BEDROCK_LLAMA_MODELS, BEDROCK_MISTRAL_MODELS, \
    BEDROCK_COHERE_MODELS, HF_SFORMER_MODELS, OPENAI_MODELS, SGLANG_MODELS


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__()
        self.add_argument(
            "--out_dir", type=str, default="outputs"
        )
        self.add_argument(
            "--run_id", type=str
        )
        self.add_argument(
            "--seed", type=int, default=42,
        )
        self.add_argument(
            "--n_seeds", type=int, default=5,
        )
        self.add_argument(
            "--device", type=str, default="cuda",
        )
        self.add_argument(
            "--dtype", type=str, choices=["fp16", "fp32", "fp64"], default="fp64",
            # GPs work better with fp64
        )
        self.add_argument(
            "--debug", action="store_true",
        )
        self.add_argument(
            "--verbose", action="store_true",
        )

        if group == "semantle":
            self.add_argument(
                "--task", type=str, choices=["semantle", "arc", "molopt"], default="semantle"
            )
            self.add_argument(
                "--target", type=str, default=None,
            )
            self.add_argument(
                "--n_targets", type=int, default=None,
            )
            self.add_argument(
                "--targets_start_idx", type=int, default=0,
            )
            self.add_argument(
                "--n_evaluations", type=int, default=100,
            )
            self.add_argument(
                "--n_warmstart", type=int, default=20
            )
            self.add_argument(
                "--candidates_fname", type=str, default=None,
            )
            self.add_argument(
                "--load_from_prev_run", type=str, default=None,
            )
            self.add_argument(
                "--candidate_gen_batch_size", type=int, default=4
            )
            self.add_argument(
                "--candidate_gen_n_parallel", type=int, default=10
            )
            self.add_argument(
                "--save_candidates", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--exit_after_candidate_gen", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--n_unlabeled", type=int, default=None
            )
            self.add_argument(
                "--keep_invalid_unlabeled_candidates", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--task_fpath", type=str, default=None,
            )
            self.add_argument(
                "--warmstart_strategy", type=str, choices=["random", "top", "bottom", "diverse"],
                default="random"
            )
            self.add_argument(
                "--surrogate_fn", type=str, choices=["gp", "laplace"], default="gp"
            )
            self.add_argument(
                "--acquisition_fn", type=str,
                choices=["thompson_sampling", "EI", "logEI", "UCB", "OPRO", "random", "none"],
                default="logEI"
            )
            self.add_argument(
                "--gp_kernel", type=str, choices=["rbf", "matern", "cosine", "materncosine", "cosinedistance"],
                default="matern"
            )
            self.add_argument(
                "--gp_noise_var", type=float, help="Unset for learned, 0 for no noise, >0 for fixed noise."
            )
            self.add_argument(
                "--ladder_kernel", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--kernel_per_dim_lengthscale", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--kernel_matern_nu", type=float
            )
            self.add_argument(
                "--kernel_lengthscale", type=float
            )
            self.add_argument(
                "--kernel_lengthscale_prior_concentration", type=float
            )
            self.add_argument(
                "--kernel_lengthscale_prior_rate", type=float
            )
            self.add_argument(
                "--kernel_outputscale", type=float
            )
            self.add_argument(
                "--kernel_outputscale_prior_concentration", type=float
            )
            self.add_argument(
                "--kernel_outputscale_prior_rate", type=float
            )
            self.add_argument(
                "--kernel_mean", type=float
            )
            self.add_argument(
                "--kernel_mean_prior_mean", type=float
            )
            self.add_argument(
                "--kernel_mean_prior_std", type=float
            )
            self.add_argument(
                "--kernel_period_length_prior_mean", type=float
            )
            self.add_argument(
                "--kernel_period_length_prior_std", type=float
            )
            self.add_argument(
                "--bnn_activation", type=str, choices=["relu", "tanh", "layernorm"], default="relu"
            )
            self.add_argument(
                "--bnn_hidden_dim", type=int, default=50
            )
            self.add_argument(
                "--bbox_model", type=str,
                choices=["word2vec", "simcse", "codegen"] + \
                        list(HF_PIPELINE_MODELS.keys()) + list(BEDROCK_CLAUDE_MODELS.keys()) + \
                        list(BEDROCK_LLAMA_MODELS.keys()) + list(BEDROCK_MISTRAL_MODELS.keys()) + \
                        list(OPENAI_MODELS.keys()),
                default="simcse"
            )
            self.add_argument(
                "--repr_model", type=str,
                choices=["simcse"] + list(HF_PIPELINE_MODELS.keys()) + list(BEDROCK_COHERE_MODELS.keys()) + \
                        list(HF_SFORMER_MODELS.keys()),
                default="simcse"
            )
            self.add_argument(
                "--repr_llm_pooling", type=str, choices=["mean", "last", "first"], default="mean"
            )
            self.add_argument(
                "--repr_prompt", type=str, default="The answer is %s."
            )
            self.add_argument(
                "--repr_codegen_strategy", type=str, choices=["docstring", "code", "docstring_code", "none"],
                default="docstring"
            )
            self.add_argument(
                "--repr_codegen_instruct", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--repr_diff_vector", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--bbox_prompt", type=str, default="What is a %s?"
            )
            self.add_argument(
                "--bbox_n_train", type=int, default=None
            )
            self.add_argument(
                "--gen_model", type=str,
                choices=list(HF_PIPELINE_MODELS.keys()) + list(BEDROCK_CLAUDE_MODELS.keys()) + \
                        list(BEDROCK_LLAMA_MODELS.keys()) + list(BEDROCK_MISTRAL_MODELS.keys()) + \
                        list(OPENAI_MODELS.keys()) + list(SGLANG_MODELS.keys()),
                default="llama-3.1-8b-instruct-bedrock"
            )
            self.add_argument(
                "--hf_access_token", type=str
            )
            self.add_argument(
                "--opt_square_hull", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--opt_hull_margin", type=float, default=1.5
            )
            self.add_argument(
                "--opt_num_restarts", type=int, default=10
            )
            self.add_argument(
                "--opt_batch_size", type=int, default=2
            )
            self.add_argument(
                "--opt_return_best_only", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--acq_ucb_beta", type=float, default=0.3
            )
            self.add_argument(
                "--vec2text_sample_unique", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--vec2text_unique_retries", type=int, default=5
            )
            self.add_argument(
                "--vec2text_fix_retries", type=int, default=0
            )
            self.add_argument(
                "--vec2text_revise_retries", type=int, default=0
            )
            self.add_argument(
                "--vec2text_multiturn_retry", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--vec2text_demos", type=int, default=10
            )
            self.add_argument(
                "--vec2text_rand_samples", type=int, default=None
            )
            self.add_argument(
                "--vec2text_trials", type=int, default=1,
                help="Number of vec2text trials to run for each BO proposal"
            )
            self.add_argument(
                "--vec2text_batch_size", type=int, default=1,
                help="Number of candidates to generate per iteration"
            )
            self.add_argument(
                "--vec2text_n_parallel", type=int, default=10
            )
            self.add_argument(
                "--skip_on_vec2text_errors", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--vec2text_normalize", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--vec2text_normalize_min", type=float, default=0.1
            )
            self.add_argument(
                "--vec2text_normalize_max", type=float, default=0.8
            )
            self.add_argument(
                "--vec2text_target_score", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--vec2text_exploit_threshold", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--vec2text_lmx", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--use_sim_for_selection", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--opro_n_low_previous", type=int, default=0,
                help="Number of low-scoring previous guesses to show in the prompt."
            )
            self.add_argument(
                "--llm_enable_cot", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--llm_tokens", type=int, default=512
            )
            self.add_argument(
                "--llm_temperature", type=float, default=0.6
            )
            self.add_argument(
                "--llm_top_p", type=float, default=0.9
            )
            self.add_argument(
                "--normalize_inputs", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--surr_normalize_inputs_botorch", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--surr_standardize_outputs", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--surr_skip_invalid_candidates", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--set_invalid_to_zero", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--repeat_cand_strategy", type=str, choices=["bo_proposal", "skip"],
                default="bo_proposal"
            )
            self.add_argument(
                "--repeat_streak_threshold", type=int, default=3
            )
            self.add_argument(
                "--repeat_streak_perturb", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--low_dim_strategy", type=str, choices=["pca", "random", "off"], default="pca"
            )
            self.add_argument(
                "--low_dim", type=int, default=10
            )
            self.add_argument(
                "--low_dim_pca_ncands", type=int, default=100,
                help="Number of candidates to sample for PCA projection matrix estimation. If 0, use warmstart set."
            )
            self.add_argument(
                "--repr_hull_ncands", type=int, default=100,
                help="Number of candidates to sample for optimization hull estimation. If 0, use warmstart set."
            )
            self.add_argument(
                "--diagnostic_human_input", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--visualize_posterior", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--visualize_posterior_top", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--visualize_posterior_fname", type=str, default=None,
            )
            self.add_argument(
                "--visualize_posterior_ncands", type=int, default=3000,
                help="Number of candidates to sample for visualization."
            )
            self.add_argument(
                "--visualize_posterior_anim", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--generate_feedback", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--use_method_defaults", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_use_numpy", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_show_transpose", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--arc_use_priors", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_use_docstring", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_use_code", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_use_scores", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--arc_use_mixing", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--arc_show_pred_in_fix", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--arc_no_demos", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--verbose_codegen_errors", action=argparse.BooleanOptionalAction, default=False
            )
            # oracle debugging flags
            self.add_argument(
                "--rand_cands_path", type=str
            )
            self.add_argument(
                "--all_cands_path", type=str
            )
            self.add_argument(
                "--oracle_pick_best", action=argparse.BooleanOptionalAction, default=True
            )
            self.add_argument(
                "--oracle_only_unseen", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--oracle_only_valid", action=argparse.BooleanOptionalAction, default=False
            )
            self.add_argument(
                "--oracle_fix_all_cands", action=argparse.BooleanOptionalAction, default=False
            )
            # /oracle debugging flags
