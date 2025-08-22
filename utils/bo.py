import os
import numpy as np
import torch
import math

from matplotlib import pyplot as plt
import matplotlib.animation as animation

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from gpytorch.kernels import ScaleKernel, MaternKernel, CosineKernel, RBFKernel
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.kernels import Kernel, MultiDeviceKernel
from gpytorch.means import ConstantMean
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from laplace_bayesopt.botorch import LaplaceBoTorch

from utils.misc import TORCH_DTYPE

# For semantle
KERNEL_DEFAULTS = {
    "semantle": {
        'matern_nu': 2.5,
        # Lengthscale gamma prior
        'lengthscale_prior_concentration': 3.0,
        'lengthscale_prior_rate': 6.0,
        # Outputscale gamma prior
        'outputscale_prior_concentration': 5.0,
        'outputscale_prior_rate': 0.15,
        # Mean normal prior
        'mean_prior_mean': None,
        'mean_prior_std': None
    },
    "arc": {
        'matern_nu': 2.5,
        # Lengthscale gamma prior
        'lengthscale_prior_concentration': 35.0,
        'lengthscale_prior_rate': 30.0,
        # Outputscale gamma prior
        'outputscale_prior_concentration': 3.0,
        'outputscale_prior_rate': 0.15,
        # Mean normal prior
        'mean_prior_mean': 0.5,
        'mean_prior_std': 0.01
    }
}


# From: https://github.com/wiseodd/laplace-bayesopt/blob/master/laplace_bayesopt/acqf.py
class ThompsonSampling(AnalyticAcquisitionFunction):
    """
    Thompson sampling acquisition function. While it uses a posterior sample, it is an analytic one.
    I.e. once we pick a sample of the posterior f_s ~ p(f | D), f_s is a deterministic function over x.

    Parameters:
    -----------
    model: botorch.models.model.Model

    posterior_transform: botorch.acquisition.objective.PosteriorTransform
        Optional

    maximize: bool, default = True
        Whether to maximize the acqf f_s or minimize it

    random_state: int, default = 123
        The random state of the sampling f_s ~ p(f | D). This is to ensure that for any given x,
        the sample from p(f(x) | D) comes from the same sample posterior sample f_s ~ p(f | D).
    """

    def __init__(
            self, model, posterior_transform=None, maximize=True, random_state=123
    ):
        super().__init__(model, posterior_transform)
        self.maximize = maximize
        self.random_state = random_state

    def forward(self, x):
        """
        Parameters:
        -----------
        x: torch.Tensor
            Shape (n, 1, d)

        Returns:
        --------
        f_sample: torch.Tensor
            Shape (n,)
        """
        mean, std = self._mean_and_sigma(x)

        if len(mean.shape) == 0:
            mean = mean.unsqueeze(0)
        if len(std.shape) == 0:
            std = std.unsqueeze(0)

        generator = torch.Generator(device=x.device).manual_seed(self.random_state)
        eps = torch.randn(*std.shape, device=x.device, generator=generator)
        f_sample = mean + std * eps

        # BoTorch assumes acqf to be maximization
        # https://github.com/pytorch/botorch/blob/0e74bb60be3492590ea88d6373d89a877c6a52c1/botorch/generation/gen.py#L249-L252
        return f_sample if self.maximize else -f_sample


class MaternCosineKernel(MaternKernel):
    def __init__(self, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternCosineKernel, self).__init__(**{**kwargs, "ard_num_dims": 1})
        self.nu = nu

    # def cosine_similarity(self, x1, x2):
    #     x1_norm = x1 / x1.norm(dim=-1, keepdim=True)
    #     x2_norm = x2 / x2.norm(dim=-1, keepdim=True)
    #     return torch.matmul(x1_norm.squeeze(), x2_norm.T.squeeze())
    def cosine_similarity(self, x1, x2):
        # Normalize inputs to unit vectors
        x1_norm = x1 / x1.norm(dim=-1, keepdim=True)  # Shape: [*, n, d]
        x2_norm = x2 / x2.norm(dim=-1, keepdim=True)  # Shape: [*, m, d]
        # Compute the cosine similarity
        # The shape of x1_norm: [..., n, d], x2_norm: [..., m, d]
        # We need to perform batch matrix multiplication along the last two dimensions.
        # Reshape x1_norm to [..., n, 1, d] for broadcasting
        x1_norm = x1_norm.unsqueeze(-2)  # Shape: [..., n, 1, d]
        # Reshape x2_norm to [..., 1, m, d] for broadcasting
        x2_norm = x2_norm.unsqueeze(-3)  # Shape: [..., 1, m, d]
        # Perform batch matrix multiplication
        # Resulting shape: [..., n, m]
        cosine_matrix = torch.matmul(x1_norm, x2_norm.transpose(-1, -2))  # Transpose x2_norm to shape [..., d, m]
        return cosine_matrix

    def cosine_distance(self, x1, x2):
        return 1 - self.cosine_similarity(x1, x2)

    def matern_kernel(self, distance):
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)
        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        return constant_component * exp_component

    def forward(self, x1, x2, diag=False, **params):
        # Compute the cosine "distance"
        cosine_dist = self.cosine_distance(x1, x2)
        # Scale the cosine distance by lengthscale
        scaled_dist = cosine_dist.div(self.lengthscale)
        with torch.no_grad():
            distance_shape = self.covar_dist(x1, x2, diag=diag, **params).shape
        scaled_dist = scaled_dist.reshape(distance_shape)
        # Apply the Matern kernel to the scaled cosine distance
        return self.matern_kernel(scaled_dist)


class CosineDistanceKernel(CosineKernel):
    def __init__(self, **kwargs):
        super(CosineDistanceKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, **params):
        return 1. - super(CosineDistanceKernel, self).forward(x1, x2, **params)


class LadderKernel(Kernel):
    def __init__(self, latent_kernel, structure_kernel, latent_train, structure_train, **kwargs):
        super().__init__(**kwargs)
        self.latent_kernel = latent_kernel  # Kernel on the latent space
        self.structure_kernel = structure_kernel  # Kernel on the structured space
        self.latent_train = latent_train
        self.lp_dim = self.latent_train.shape[-1]
        self.structure_train = structure_train

    def forward(self, z1, z2, **params):
        check_dim = 0
        if len(z1.shape) > 2:
            check_dim = z1.shape[0]
            z1 = z1.squeeze(1)
        if len(z2.shape) > 2:
            check_dim = z2.shape[0]
            z2 = z2[0]
        latent_train_z1 = z1[:, :self.lp_dim]
        latent_train_z2 = z2[:, :self.lp_dim]

        K_train_struct = self.structure_kernel.forward(self.structure_train, self.structure_train, **params)
        latent_kernel = self.latent_kernel.forward(self.latent_train, self.latent_train, **params)
        K_z1_training = self.latent_kernel.forward(latent_train_z1, self.latent_train, **params)
        K_z2_training = self.latent_kernel.forward(latent_train_z2, self.latent_train, **params)
        latent_kernel_inv = torch.inverse(
            latent_kernel + 0.0001 * torch.eye(len(self.latent_train)).to(latent_kernel.device))

        kernel_val = K_z1_training @ latent_kernel_inv @ K_train_struct @ latent_kernel_inv @ K_z2_training.T
        if check_dim > 0:
            kernel_val = kernel_val.unsqueeze(1)
        return kernel_val


def get_surrogate(model, train_x, train_y, train_yvar=None, gp_kernel=None, n_objs=1,
                  standardize_outputs=True, normalize_inputs=False, bounds=None,
                  task="semantle", device='cpu', dtype="fp32", **kwargs):
    if train_y.size(-1) != 1:
        train_y = train_y.unsqueeze(-1)

    # Transforms
    outcome_transform = Standardize(m=1) if standardize_outputs else None
    input_transform = None
    if normalize_inputs:
        input_transform = Normalize(d=train_x.shape[-1], bounds=bounds)

    hyperparams = {**KERNEL_DEFAULTS[task], **kwargs}

    if model == "gp":
        # Noise
        # None: learned during MLL
        if type(train_yvar) is not torch.Tensor:  # else: fixed noise per observation
            if train_yvar == 0:  # no noise
                train_yvar = torch.full_like(train_y, 1e-6)
            elif train_yvar is not None:  # fixed noise
                train_yvar = torch.full_like(train_y, train_yvar)
        if train_yvar is not None:
            train_yvar.to(TORCH_DTYPE[dtype])

        add_args = {}
        # Kernel
        if gp_kernel is not None:
            kernel = {
                'rbf': RBFKernel(ard_num_dims=hyperparams.get('ard_num_dims', train_x.shape[-1]),
                                 **{k: v for k, v in {"lengthscale_prior": GammaPrior(
                                     hyperparams.get('lengthscale_prior_concentration'),
                                     hyperparams.get('lengthscale_prior_rate')) if (hyperparams.get(
                                     'lengthscale_prior_concentration') is not None and hyperparams.get(
                                     'lengthscale_prior_rate') is not None) else None}.items() if
                                    v is not None}
                                 ),
                'matern': MaternKernel(nu=hyperparams.get('matern_nu'),
                                       ard_num_dims=hyperparams.get('ard_num_dims', train_x.shape[-1]),
                                       **{k: v for k, v in {"lengthscale_prior": GammaPrior(
                                           hyperparams.get('lengthscale_prior_concentration'),
                                           hyperparams.get('lengthscale_prior_rate')) if (hyperparams.get(
                                           'lengthscale_prior_concentration') is not None and hyperparams.get(
                                           'lengthscale_prior_rate') is not None) else None}.items() if
                                          v is not None}),
                'materncosine': MaternCosineKernel(nu=hyperparams.get('matern_nu'),
                                                   ard_num_dims=hyperparams.get('ard_num_dims', train_x.shape[-1]),
                                                   **{k: v for k, v in {"lengthscale_prior": GammaPrior(
                                                       hyperparams.get('lengthscale_prior_concentration'),
                                                       hyperparams.get('lengthscale_prior_rate')) if (hyperparams.get(
                                                       'lengthscale_prior_concentration') is not None and hyperparams.get(
                                                       'lengthscale_prior_rate') is not None) else None}.items() if
                                                      v is not None}),
                'cosine': CosineKernel(**{k: v for k, v in {"period_length_prior": NormalPrior(
                    hyperparams.get('period_length_prior_mean'),
                    hyperparams.get('period_length_prior_std')) if (hyperparams.get(
                    'period_length_prior_mean') is not None and hyperparams.get(
                    'period_length_prior_std') is not None) else None}.items() if v is not None}),
                'cosinedistance': CosineDistanceKernel(**{k: v for k, v in {"period_length_prior": NormalPrior(
                    hyperparams.get('period_length_prior_mean'),
                    hyperparams.get('period_length_prior_std')) if (hyperparams.get(
                    'period_length_prior_mean') is not None and hyperparams.get(
                    'period_length_prior_std') is not None) else None}.items() if v is not None}),
            }[gp_kernel]

            if not hyperparams.get('ladder_kernel', False):
                add_args["covar_module"] = ScaleKernel(base_kernel=kernel,
                                                       **{k: v for k, v in {"outputscale_prior": GammaPrior(
                                                           hyperparams.get('outputscale_prior_concentration'),
                                                           hyperparams.get('outputscale_prior_rate')) if (
                                                               hyperparams.get(
                                                                   'outputscale_prior_concentration') is not None and hyperparams.get(
                                                           'outputscale_prior_rate') is not None) else None}.items()
                                                          if
                                                          v is not None})
            else:
                structure_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_y.shape[-1],
                    **{k: v for k, v in {"lengthscale_prior": GammaPrior(3, 6)}.items() if
                       v is not None}
                )
                add_args["covar_module"] = ScaleKernel(
                    base_kernel=LadderKernel(
                        latent_kernel=kernel,
                        structure_kernel=structure_kernel,
                        latent_train=train_x,
                        structure_train=kwargs.get("ladder_kernel_data", train_y).view(len(train_x), -1)
                    ),
                    **{k: v for k, v in {"outputscale_prior": GammaPrior(
                        hyperparams.get('outputscale_prior_concentration'),
                        hyperparams.get('outputscale_prior_rate')) if (
                            hyperparams.get(
                                'outputscale_prior_concentration') is not None and hyperparams.get(
                        'outputscale_prior_rate') is not None) else None}.items()
                       if v is not None})
            # Set prior on mean
            if (mean_prior_mean := hyperparams.get('mean_prior_mean')) is not None and (
                    mean_prior_std := hyperparams.get('mean_prior_std')) is not None:
                add_args["mean_module"] = ConstantMean(constant_prior=NormalPrior(mean_prior_mean, mean_prior_std))

        if torch.cuda.device_count() > 1:
            add_args["covar_module"] = MultiDeviceKernel(add_args["covar_module"],
                                                         device_ids=range(torch.cuda.device_count()),
                                                         output_device=torch.device('cuda:0'))

        # Model
        if n_objs == 1:
            model = SingleTaskGP(train_x.to(TORCH_DTYPE[dtype]), train_y.to(TORCH_DTYPE[dtype]),
                                 train_Yvar=train_yvar.to(TORCH_DTYPE[dtype]), outcome_transform=outcome_transform,
                                 input_transform=input_transform, **add_args)
            _covar_module_ref = model.covar_module if not isinstance(model.covar_module, MultiDeviceKernel) else \
                model.covar_module.module
        else:
            # Multi-objective GP
            raise NotImplementedError

        if hyperparams.get("mean", None) is not None:
            # Set to the constant value and don't optimize
            model.mean_module.constant = hyperparams["mean"]
            model.mean_module.constant.requires_grad_(False)
        if hyperparams.get("lengthscale", None) is not None:
            if not hyperparams.get('ladder_kernel', False):
                # Set to the constant value and don't optimize
                _covar_module_ref.base_kernel.lengthscale = hyperparams["lengthscale"]
                _covar_module_ref.base_kernel.raw_lengthscale.requires_grad_(False)
            else:
                # Set to the constant value and don't optimize
                _covar_module_ref.base_kernel.latent_kernel.lengthscale = hyperparams["lengthscale"]
                _covar_module_ref.base_kernel.latent_kernel.raw_lengthscale.requires_grad_(False)
                _covar_module_ref.base_kernel.structure_kernel.lengthscale = hyperparams["lengthscale"]
                _covar_module_ref.base_kernel.structure_kernel.raw_lengthscale.requires_grad_(False)
        if hyperparams.get("outputscale", None) is not None:
            # Set to the constant value and don't optimize
            _covar_module_ref.outputscale = hyperparams["outputscale"]
            _covar_module_ref.raw_outputscale.requires_grad_(False)

        requires_optim = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                requires_optim = True
                # print(f"Requires optim: {(name, param)}")
                break

        # Fit
        if requires_optim:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        print(f"Learned GP mean = {model.mean_module.constant.item()}")
        if not hyperparams.get('ladder_kernel', False):
            print(f"Learned GP lengthscale = {_covar_module_ref.base_kernel.lengthscale}")
        else:
            print(f"Learned GP lengthscale = {_covar_module_ref.base_kernel.latent_kernel.lengthscale}")
            print(f"Learned GP lengthscale (struct) = {_covar_module_ref.base_kernel.structure_kernel.lengthscale}")
        print(f"Learned GP outputscale = {_covar_module_ref.outputscale.item()}")
    elif model == "laplace":
        bnn_hidden_dim = hyperparams.get('bnn_hidden_dim', 50)
        bnn_hess_factorization = 'kron'
        feature_dim = train_x.shape[-1]
        bnn_activation = {
            'relu': torch.nn.ReLU,
            'tanh': torch.nn.Tanh,
            'layernorm': lambda: torch.nn.LayerNorm(bnn_hidden_dim),
        }[hyperparams.get('bnn_activation', 'relu')]

        def get_net():
            return torch.nn.Sequential(
                torch.nn.Linear(feature_dim, bnn_hidden_dim),
                bnn_activation(),
                torch.nn.Linear(bnn_hidden_dim, bnn_hidden_dim),
                bnn_activation(),
                torch.nn.Linear(bnn_hidden_dim, n_objs)
            ).to(TORCH_DTYPE[dtype]).to(device)

        model = LaplaceBoTorch(
            get_net, train_x.to(TORCH_DTYPE[dtype]), train_y.to(TORCH_DTYPE[dtype]), noise_var=train_yvar,
            hess_factorization=bnn_hess_factorization, outcome_transform=outcome_transform,
            input_transform=input_transform, device=device
        )
    else:
        raise NotImplementedError

    return model.to(device)


def get_acq_fn(acquisition_fn, surrogate=None, best_y=None, d=None, acq_ucb_beta=None, batch_size=1):
    if acquisition_fn == "logEI":
        if batch_size > 1:
            return qLogExpectedImprovement(model=surrogate, best_f=best_y,
                                           sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500, d])))
        return LogExpectedImprovement(model=surrogate, best_f=best_y)
    elif acquisition_fn == "EI":
        if batch_size > 1:
            return qExpectedImprovement(model=surrogate, best_f=best_y,
                                        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500, d])))
        return ExpectedImprovement(model=surrogate, best_f=best_y)
    elif acquisition_fn == "UCB":
        if batch_size > 1:
            return qUpperConfidenceBound(model=surrogate, beta=acq_ucb_beta,
                                         sampler=SobolQMCNormalSampler(sample_shape=torch.Size([500, d])))
        return UpperConfidenceBound(model=surrogate, beta=acq_ucb_beta)
    elif acquisition_fn == "thompson_sampling":
        return ThompsonSampling(model=surrogate)
    elif acquisition_fn == "OPRO":
        return "OPRO"
    elif acquisition_fn == "random":
        return "random"
    elif acquisition_fn == "none":
        return "none"
    else:
        raise NotImplementedError


def optimize_acq_fn(acq_fn, d, opt_q=1, opt_num_restarts=10, opt_raw_samples=512,
                    bounds=None, return_best_only=True, normalize=False, device='cuda'):
    """Optimizes the acquisition function, and returns a new candidate"""

    # Use 0-1 bounds by default
    if bounds is None:
        _bounds = torch.stack([torch.zeros(d, device=device), torch.ones(d, device=device)])
    else:
        _bounds = bounds.to(device)

    if type(acq_fn) is not str:
        n_iters, _opt_q = 1, opt_q
        if type(acq_fn).__name__ == "ThompsonSampling":
            n_iters, _opt_q = opt_q, 1
        _candidates, _acq_vals = [], []
        for _ in range(n_iters):
            # Multi-start gradient ascent
            __candidates, __acq_vals = optimize_acqf(
                acq_function=acq_fn,
                bounds=_bounds,
                q=_opt_q,
                num_restarts=opt_num_restarts,
                raw_samples=opt_raw_samples,
                return_best_only=return_best_only
            )
            _candidates.append(__candidates)
            _acq_vals.append(__acq_vals)
        _candidates = torch.stack(_candidates).view(opt_q, -1)
        acq_vals = torch.stack(_acq_vals)
        try:
            # If normalized, transform candidates back to original input scale
            candidates = acq_fn.model.input_transform.untransform(_candidates)
        except AttributeError:
            candidates = _candidates
        if normalize:
            candidates = torch.nn.functional.normalize(candidates)
    elif acq_fn == "OPRO":
        # OPRO
        candidates = torch.zeros(opt_q, d, device=device)  # not used
        acq_vals = torch.zeros(opt_q, device=device)  # not used
    elif acq_fn == "random":
        # Random search
        candidates = torch.rand(opt_q, d, device=device) * (bounds[1] - bounds[0]) + bounds[0]
        acq_vals = torch.zeros(opt_q, device=device)  # not used
    elif acq_fn == "none":
        # No examples in prompt
        candidates = torch.zeros(opt_q, d, device=device)  # not used
        acq_vals = torch.zeros(opt_q, device=device)  # not used
    else:
        raise NotImplementedError

    return candidates, acq_vals


def plot_posterior(posterior_vals, posterior_cands, path, animate=False, anim_interval=300, anim_repeat=True,
                   obs_xy=None, trend_over_unobserved=True, conf_intervals=[95], top_k=None):
    # Bayesian credible interval to std map
    conf_to_std = {
        95: 1.96,
        99: 2.576
    }
    all_vals = np.array([posterior_vals[k] for k in posterior_vals])
    plt.clf()
    fig, ax = plt.subplots()
    obs_xy_in_viz = []
    warmstart_obs_xy_in_viz = []

    def update(frame):
        ax.clear()
        vals = all_vals[len(all_vals) - 1 if not animate else (frame if frame < len(all_vals) else (frame - 1))]
        _obs_xy = obs_xy[frame] if obs_xy is not None else []
        y, mean, std = vals[:, 0], vals[:, 1], vals[:, 2]
        x = 1 + np.arange(len(y))
        ax.plot(x, y, label="True")
        ax.plot(x, mean, label="Posterior", color="orange", alpha=0.8)
        # Plot observations
        _obs_xy_in_viz = []
        for xy in _obs_xy:
            try:
                _obs_xy_in_viz.append((posterior_cands.index(xy[0]), xy[1]))
            except:
                continue
        if frame == 0:
            warmstart_obs_xy_in_viz.extend(_obs_xy_in_viz)
        else:
            obs_xy_in_viz.extend(_obs_xy_in_viz)
        if len(warmstart_obs_xy_in_viz) > 0:
            obs_x, obs_y = zip(*warmstart_obs_xy_in_viz)
            ax.scatter(obs_x, obs_y, label="Warmstart", color="gray")
        if len(obs_xy_in_viz) > 0:
            obs_x, obs_y = zip(*obs_xy_in_viz)
            ax.scatter(obs_x, obs_y, label="BO Observation", color="red")
        # Plot uncertainty
        for conf_interval in conf_intervals:
            ax.fill_between(x,
                            mean - (conf_to_std[conf_interval] * std),
                            mean + (conf_to_std[conf_interval] * std),
                            alpha=0.4, color="orange")
        # Plot trend
        if trend_over_unobserved:
            obs_idxs = [o[0] for o in obs_xy_in_viz]
            unobs_idxs = np.array([i for i in range(len(x)) if i not in obs_idxs]).astype(int)
        posterior_trend = np.polyfit(x[unobs_idxs] if trend_over_unobserved else x,
                                     mean[unobs_idxs] if trend_over_unobserved else mean,
                                     8)
        posterior_trend = np.poly1d(posterior_trend)
        posterior_trend_y = posterior_trend(x)
        ax.plot(x, posterior_trend_y, "r--", label=f"Posterior Mean Trend")  # (s={_posterior_trend[0]})
        ax.legend()
        ax.set_title(f"t = {frame}")
        ax.set_ylabel("Objective")
        # Set y limit to be the same for all frames
        ax.set_ylim([-0.1, 1.1])  # ax.set_ylim([-5, 5] if args.surrogate_fn == "laplace" else [-0.1, 1.1])
        ax.set_xlabel("Rank")
        if top_k is not None:
            ax.set_xlim([0, top_k + 1])
        ax.grid()

    update(frame=0)  # initial plot

    if animate:
        ani = animation.FuncAnimation(fig=fig, func=update, frames=range(len(all_vals) + 1), interval=anim_interval,
                                      repeat=anim_repeat, repeat_delay=50)
        ani.save(path.replace(".json", ".gif"), writer="pillow")  # imagemagick
    else:
        plt.savefig(path.replace(".json", ".png"), bbox_inches="tight")


def plot_trace(traces, legend=None, ylim=(0, 1.05), out_dir=None, fname="aggregate.png", silent=False,
               with_std=True, xlabel="Iteration", ylabel="Objective"):
    # Retrieve the default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.clf()
    if type(traces) is not list:
        traces = [traces]
    for i, trace in enumerate(traces):
        color = default_colors[i % len(default_colors)]
        # Create the plot
        plt.plot(np.mean(trace, axis=0), linestyle='-', label=legend[i] if legend is not None else f"{i}",
                 color=color)
        if with_std:
            plt.fill_between(np.arange(trace.shape[1]),
                             np.mean(trace, axis=0) - np.std(trace, axis=0),
                             np.mean(trace, axis=0) + np.std(trace, axis=0),
                             alpha=0.1, color=color)
    # Add titles and labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Fix the y-axis to be between 0 and 1
    plt.ylim([ylim[0], ylim[1]])
    plt.grid()
    if legend is not None:
        plt.legend()
    # Save plot
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname))
        if not silent:
            print(f"Saved trace plot to {out_dir}/{fname}")
    else:
        plt.show()


def plot_molopt_trace(trace_xy, n_warmstart, out_dir=None, fname="trace.png", xlabel="Druglikeness (QED)",
                      xlim=(-0.05, 1.05), ylim=(-12.5, -2.5), ylabel="Binding Affinity (Vina score)",
                      silent=False):
    scalar_qed_vina = [[r[1]] + list(r[2]) for r in trace_xy]
    best_so_far_sqv = [scalar_qed_vina[0]]
    for i in range(1, len(scalar_qed_vina)):
        if scalar_qed_vina[i][0] > best_so_far_sqv[-1][0]:
            best_so_far_sqv.append(scalar_qed_vina[i])
    solutions = np.array(scalar_qed_vina)
    solutions_best = np.array(best_so_far_sqv)

    plt.clf()
    plt.scatter(solutions[:n_warmstart, 1], solutions[:n_warmstart, 2], label="Warmstart", color="gray", s=25,
                marker='s')
    plt.scatter(solutions[n_warmstart:, 1], solutions[n_warmstart:, 2], label="Observation", color="black", s=25)
    plt.plot(solutions_best[:, 1], solutions_best[:, 2], label="Best Trace", color="red", linestyle="--",
             alpha=1)
    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.xlim(*xlim)

    # Save plot
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname))
        if not silent:
            print(f"Saved trace plot to {out_dir}/{fname}")
    else:
        plt.show()
