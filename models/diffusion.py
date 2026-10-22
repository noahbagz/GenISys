"""Reusable implementation copied from Ada; see docs/SOURCE_PROVENANCE.md."""
import numpy as np


import json


import math


import matplotlib.pyplot as plt


from contextlib import contextmanager


import torch


import torch.nn as nn


import torch.nn.functional as F


from tqdm import tqdm


import sklearn.preprocessing as PP


class SampleResult:
    def __init__(self, scaled, raw):
        self.scaled = scaled
        self.raw = raw

    def __iter__(self):
        yield self.scaled
        yield self.raw

    def __len__(self):
        return len(self.scaled)

    def __getitem__(self, item):
        if item == 0:
            return self.scaled
        if item == 1:
            return self.raw
        raise IndexError(item)

    def __array__(self, dtype=None):
        return np.asarray(self.scaled, dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self.scaled).shape


from torch.utils.data import  DataLoader


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def make_beta_schedule(config, device):
    """Create the configured diffusion schedule on the requested device."""

    timesteps = int(config['Diffusion_Timesteps'])
    schedule = str(config.get('beta_schedule', 'linear')).lower()

    if schedule == 'linear':
        beta_start = float(config.get('beta_start', 0.0001))
        beta_end = float(config.get('beta_end', 0.02))
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    elif schedule == 'cosine':
        cosine_s = float(config.get('cosine_s', 0.008))
        values = betas_for_alpha_bar(
            timesteps,
            lambda t: np.cos((t + cosine_s) / (1.0 + cosine_s) * np.pi / 2) ** 2,
        )
        betas = torch.from_numpy(values.astype('float32'))
    else:
        raise ValueError(f'Unknown beta_schedule {schedule!r}.')

    if not bool(torch.all((betas > 0.0) & (betas < 1.0))):
        raise ValueError('Every diffusion beta must be strictly between zero and one.')
    return betas.to(device)


def timestep_embedding(timesteps, dim, max_period=10000, device=torch.device('cuda:0')):
    """
    From https://github.com/rotot0/tab-ddpm
    
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1,device=device)
    return embedding


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


class Denoise_MLP_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.cdim = DDPM_Dict['cdim']
        self.tdim  = DDPM_Dict['tdim']
        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        self.fc.append(nn.Sequential(nn.Linear(self.tdim, self.xdim)))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        '''
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
        

    def forward(self, x, timesteps):
        a = self.X_embed(x)
        #print(a.dtype)
        x = a +  self.time_embed(timestep_embedding(timesteps, self.tdim))
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    

        return x


class Denoise_ResNet_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        #self.ydim = DDPM_Dict['ydim']
        self.tdim  = DDPM_Dict['tdim']
        #self.cdim = DDPM_Dict['cdim']
        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        '''
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
    
    def forward(self, x, timesteps):
    
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim)) 
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x


class Cond_Denoise_ResNet_Model(torch.nn.Module):
    #Conditional Denoising ResNet Model for Tabular Diffusion, allows for conditioning on other variables (e.g. labels or other features)
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.tdim  = DDPM_Dict['tdim']
        self.cdim = DDPM_Dict['cdim']

        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        self.X_embed = nn.Linear(self.xdim, self.tdim)


        self.Con_embed = nn.ModuleList()
        for i in range(len(self.cdim)):
            self.Con_embed.append(nn.Sequential(
                nn.Linear(self.cdim[i], self.tdim),
                nn.SiLU(),
                nn.Linear(self.tdim, self.tdim)))
        

        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       

        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        


    def forward(self, x, cond, timesteps):
        
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim))
        for i in range(len(self.cdim)):
            x = x + self.Con_embed[i](cond[i])

        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        # A diffusion step updates every EMA tensor. Grouping equal-device
        # operations cuts Python and CUDA launch overhead for small denoisers.
        # Keep the original multiply-then-add order for checkpoint continuity.
        parameters = [
            (name, param.data)
            for name, param in module.named_parameters()
            if param.requires_grad
        ]
        names = [name for name, _ in parameters]
        live_values = [value for _, value in parameters]
        shadow_values = [self.shadow[name].data for name in names]
        live_parts = torch._foreach_mul(live_values, 1. - self.mu)
        shadow_parts = torch._foreach_mul(shadow_values, self.mu)
        updated = torch._foreach_add(live_parts, shadow_parts)
        for name, value in zip(names, updated, strict=True):
            self.shadow[name] = value

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """Return an EMA copy when a module supplies a construction config.

        The historical implementation assumed every denoiser exposed
        ``module.config`` even though the GenISys denoisers do not.  Production
        sampling uses :meth:`average_parameters`, which does not reconstruct
        the module and therefore works for every existing architecture.
        """

        if not hasattr(module, "config"):
            raise AttributeError(
                "ema_copy requires module.config; use average_parameters() "
                "for the GenISys denoisers."
            )
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    @contextmanager
    def average_parameters(self, module):
        """Temporarily sample with EMA weights and restore live weights."""

        live = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self.ema(module)
        try:
            yield module
        finally:
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    parameter.data.copy_(live[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class Data_Normalizer:
    def __init__(self, X_LL_Scaled, X_UL_Scaled,datalength, trans_type='quantile'):
        '''
        types: quantile, minmax
        '''
        self.type = trans_type
        self.normalizer = PP.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=min(
                max(min(datalength // 30, 1000), 10),
                max(int(datalength), 1),
            ),
            subsample= int(1e9)
            )
        
        
        self.X_LL_Scaled = np.asarray(X_LL_Scaled, dtype=np.float32).reshape(-1)
        self.X_UL_Scaled = np.asarray(X_UL_Scaled, dtype=np.float32).reshape(-1)
        
        self.X_LL_norm = np.zeros((1,len(X_LL_Scaled)))
        self.X_UL_norm = np.zeros((1,len(X_LL_Scaled)))
        
        self.X_mean = np.zeros((1,len(X_LL_Scaled)))
        self.X_std = np.zeros((1,len(X_LL_Scaled)))
        
    def _coerce_input_shape(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        expected_dim = len(self.X_LL_Scaled)
        if X.ndim == 2 and X.shape[-1] != expected_dim and X.size % expected_dim == 0:
            X = X.reshape(-1, expected_dim)

        return X

    def fit_Data(self,X):
        X = self._coerce_input_shape(X)
        if self.type=='quantile':
            # guard against zero-width features by replacing zero denominators
            denom = (self.X_UL_Scaled - self.X_LL_Scaled).astype(np.float64)
            denom_safe = np.where(np.isclose(denom, 0.0), 1.0, denom)
            X = 2.0*(X - self.X_LL_Scaled) / denom_safe - 1.0  # scale data between -1 and 1 before fitting quantile transformer
            self.normalizer.fit(X)
            z = self.normalizer.transform(X) # Scale Dataset between 

            self.X_LL_norm = self.normalizer.transform(self.X_LL_Scaled.reshape(1,-1))
            self.X_UL_norm = self.normalizer.transform(self.X_UL_Scaled.reshape(1,-1))
            #Scale items between -1 and 1
            #z = 2.0*(z - self.X_LL_norm)/(self.X_UL_norm - self.X_LL_norm) - 1.0
        elif self.type=='minmax':
            lower = self.X_LL_Scaled.reshape(1, -1)
            upper = self.X_UL_Scaled.reshape(1, -1)
            denom = (upper - lower).astype(np.float64)
            denom_safe = np.where(np.isclose(denom, 0.0), 1.0, denom)
            z = (X - lower) / denom_safe
        return z
    
    def transform_Data(self,X):
        #X = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        X = self._coerce_input_shape(X)

        if self.type=='quantile':
            denom = (self.X_UL_Scaled - self.X_LL_Scaled).astype(np.float64)
            denom_safe = np.where(np.isclose(denom, 0.0), 1.0, denom)
            X = 2.0*(X - self.X_LL_Scaled) / denom_safe - 1.0
            z = self.normalizer.transform(X)

            #z = 2.0*(z - self.X_LL_norm)/(self.X_UL_norm - self.X_LL_norm) - 1.0

        elif self.type=='minmax':
            lower = self.X_LL_Scaled.reshape(1, -1)
            upper = self.X_UL_Scaled.reshape(1, -1)
            denom = (upper - lower).astype(np.float64)
            denom_safe = np.where(np.isclose(denom, 0.0), 1.0, denom)
            z = (X - lower) / denom_safe
    
        return z

    def scale_X(self,z):
        #rescales data
        z = self._coerce_input_shape(z)
        if self.type=='quantile':

            #x = 0.5*(z+1.0)*(self.X_UL_norm - self.X_LL_norm) + self.X_LL_norm # scale back from -1,1 to original scale
            x = self.normalizer.inverse_transform(z)
            x = 0.5*(x+1.0)*(self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled # scale back from -1,1 to original scale
        elif self.type=='minmax':
            lower = self.X_LL_Scaled.reshape(1, -1)
            upper = self.X_UL_Scaled.reshape(1, -1)
            x = z * (upper - lower) + lower
        #x = 0.5*(x+1.0)*(self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled # scale back from -1,1 to original scale
        return x     

    def normalized_bounds(self):
        """Return the declared physical bounds in model coordinates."""

        physical_bounds = np.vstack(
            (self.X_LL_Scaled, self.X_UL_Scaled)
        )
        normalized = self.transform_Data(physical_bounds)

        # Quantile transforms can reverse coordinates only if their fitted
        # input is malformed. Taking coordinate-wise extrema also keeps the
        # contract explicit for either supported normalizer.
        return normalized.min(axis=0), normalized.max(axis=0)


class ConstraintClassifier(torch.nn.Module):
    """
    Neural network classifier for predicting design feasibility/constraint satisfaction.
    Used for classifier-guided diffusion to condition sample generation on constraint satisfaction.
    """
    def __init__(self, classifier_dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = classifier_dict['xdim']           # Design vector dimension (e.g., 45)
        self.tdim = classifier_dict['tdim']           # Hidden embedding dimension
        self.cdim = classifier_dict.get('cdim', 2)    # Class dimension (binary: feasible/infeasible)
        self.net = classifier_dict['net']             # Hidden layer architecture
        self.device = device
        self.time_conditioned = bool(classifier_dict.get('time_conditioned', False))
        self.cshipgen_layers = bool(classifier_dict.get('cshipgen_layers', False))
        
        # Input embedding layer
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        if self.time_conditioned:
            self.time_embed = nn.Sequential(
                nn.Linear(self.tdim, self.tdim),
                nn.SiLU(),
                nn.Linear(self.tdim, self.tdim),
            )
        
        # Hidden layers
        self.fc = nn.ModuleList()
        self.fc.append(self.LinLayer(self.tdim, self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1], self.net[i]))
        
        # Output layer with sigmoid for binary classification
        self.fc.append(nn.Sequential(nn.Linear(self.net[-1], self.cdim), nn.Sigmoid()))
    
    def LinLayer(self, dimi, dimo):
        """Build a classifier layer with the selected training recipe."""
        if self.cshipgen_layers:
            return nn.Sequential(
                nn.Linear(dimi, dimo), nn.SiLU(), nn.Dropout(p=0.1)
            )
        return nn.Sequential(
            nn.Linear(dimi, dimo),
            nn.SiLU(),
            nn.LayerNorm(dimo),
            nn.Dropout(p=0.1)
        )
    
    def forward(self, x, timesteps=None):
        """Forward pass through classifier."""
        x = self.X_embed(x)
        if self.time_conditioned:
            if timesteps is None:
                raise ValueError('A timestep-conditioned classifier requires timesteps.')
            x = x + self.time_embed(timestep_embedding(timesteps, self.tdim))
        
        for layer in self.fc:
            x = layer(x)
        
        return x


def classifier_probability_hinge_gradient(
    classifier,
    values,
    timestep,
    target=None,
    *,
    probability_threshold=0.5,
    relu_mode="hard",
    probability_floor=1.0e-6,
    gradient_floor=1.0e-12,
    probability_objective="log",
    return_diagnostics=False,
):
    """Return a gated gradient that raises unsatisfied class probabilities.

    An active output contributes either ``grad(log(p_target))`` or the direct
    sigmoid-output ``grad(p_target)``, selected by ``probability_objective``.
    It contributes only while its target probability is below the threshold. In
    ``hard`` mode every active output has unit weight. In ``tapered`` mode its
    weight decreases linearly with the normalized ReLU shortfall, so guidance
    approaches zero smoothly as the probability reaches the threshold.

    The classifier is evaluated at the supplied current noisy state ``values``
    and matching diffusion ``timestep``. Gradients that are non-finite or
    below ``gradient_floor`` are set to zero instead of being amplified.
    """

    with torch.enable_grad():
        values_for_gradient = values.detach().requires_grad_(True)
        classifier_time = timestep if classifier.time_conditioned else None
        prediction = classifier(values_for_gradient, classifier_time)

        # The direct-sigmoid arm uses the classifier output as written.
        # Keep the original temperature path unchanged for the log arm.
        if probability_objective not in {"log", "sigmoid"}:
            raise ValueError("probability_objective must be 'log' or 'sigmoid'.")
        temperature = getattr(classifier, "guidance_temperature", None)
        if probability_objective == "log" and temperature is not None:
            temperature = torch.as_tensor(
                temperature,
                dtype=prediction.dtype,
                device=prediction.device,
            ).reshape(1, -1).clamp_min(float(probability_floor))
            bounded = prediction.clamp(
                min=float(probability_floor),
                max=1.0 - float(probability_floor),
            )
            prediction = torch.sigmoid(torch.logit(bounded) / temperature)

        if target is None:
            target_tensor = torch.zeros_like(prediction)
            target_tensor[:, 0] = 1.0
        else:
            target_tensor = torch.as_tensor(
                target,
                dtype=prediction.dtype,
                device=prediction.device,
            )
            if target_tensor.ndim == 0:
                class_index = int(target_tensor.item())
                target_tensor = torch.zeros_like(prediction)
                target_tensor[:, class_index] = 1.0
            elif target_tensor.ndim == 1:
                target_tensor = target_tensor.reshape(1, -1).expand_as(prediction)
            else:
                target_tensor = target_tensor.expand_as(prediction)

        target_probability = (
            target_tensor * prediction
            + (1.0 - target_tensor) * (1.0 - prediction)
        )

        # ReLU determines which outputs remain unsatisfied. The weighting is
        # detached before differentiating log probability so it controls the
        # positive guidance magnitude without reversing the desired gradient.
        shortfall = F.relu(float(probability_threshold) - target_probability)
        active = (shortfall > 0.0).to(prediction.dtype).detach()
        if relu_mode == "hard":
            relu_weight = active
        elif relu_mode == "tapered":
            threshold_scale = max(
                float(probability_threshold),
                float(probability_floor),
            )
            relu_weight = (shortfall / threshold_scale).detach()
        else:
            raise ValueError(
                "guidance ReLU mode must be 'hard' or 'tapered'; "
                f"received {relu_mode!r}."
            )

        # Outputs that contain only one class have no empirical decision
        # boundary.  Classifier training attaches this frozen mask from the
        # training labels so constant-pass and constant-fail outputs cannot
        # inject an arbitrary learned gradient during sampling.
        output_mask = getattr(classifier, "guidance_output_mask", None)
        if output_mask is not None:
            output_mask = torch.as_tensor(
                output_mask,
                dtype=prediction.dtype,
                device=prediction.device,
            ).reshape(1, -1)
            active = active * output_mask
            relu_weight = relu_weight * output_mask

        # Keep the on/off mask detached. Differentiating the direct sigmoid
        # output avoids the 1/p amplification of the log arm.
        if probability_objective == "log":
            score = torch.log(target_probability.clamp_min(float(probability_floor)))
        else:
            score = target_probability
        objective = (relu_weight * score).sum()
        gradient = torch.autograd.grad(objective, values_for_gradient)[0]

    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    gradient_rms = torch.sqrt(torch.mean(gradient**2, dim=1, keepdim=True))
    usable = gradient_rms > float(gradient_floor)
    gradient = torch.where(usable, gradient, torch.zeros_like(gradient))

    if not return_diagnostics:
        return gradient

    diagnostics = {
        "active_output_fraction": float(active.mean().detach().cpu()),
        "guidance_eligible_output_fraction": float(
            (
                torch.ones_like(prediction)
                if output_mask is None
                else output_mask.expand_as(prediction)
            ).mean().detach().cpu()
        ),
        "active_sample_fraction": float(
            (active.sum(dim=1) > 0).float().mean().detach().cpu()
        ),
        "mean_relu_weight": float(relu_weight.mean().detach().cpu()),
        "zero_gradient_fraction": float((~usable).float().mean().detach().cpu()),
        "mean_target_probability": float(target_probability.mean().detach().cpu()),
        "mean_gradient_rms": float(gradient_rms.mean().detach().cpu()),
    }
    return gradient, diagnostics


def cap_guidance_displacement(displacement, max_l2):
    """Cap each sample's classifier-induced reverse-step move in L2 norm."""

    magnitude = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
    fraction = (max_l2 / magnitude.clamp_min(1.0e-12)).clamp(max=1.0)
    return displacement * fraction


def normalize_classifier_gradient_l2(gradient, floor=1.0e-8):
    """Give each sample's active classifier gradient unit L2 length.

    A zero or nearly zero ReLU-gated gradient stays zero, so normalization
    cannot reactivate a satisfied classifier or amplify numerical noise.
    """

    magnitude = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
    usable = magnitude > float(floor)
    return torch.where(
        usable,
        gradient / magnitude.clamp_min(float(floor)),
        torch.zeros_like(gradient),
    )


def posterior_mean_guided_sample(
    model_mean,
    posterior_variance,
    innovation,
    guidance_gradient,
    guidance_strength,
    *,
    max_displacement_rms=None,
    max_displacement_to_noise_rms_ratio=None,
):
    """Shift the reverse posterior mean without changing stochastic noise.

    The optional relative cap limits each sample's guidance displacement RMS
    to a multiple of that sample's posterior stochastic displacement RMS.
    This makes a cap comparable across diffusion timesteps and model blocks.
    """

    displacement = (
        posterior_variance
        * float(guidance_strength)
        * guidance_gradient
    )
    displacement = torch.nan_to_num(
        displacement,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if max_displacement_rms is not None:
        displacement_rms = torch.sqrt(
            torch.mean(displacement**2, dim=1, keepdim=True)
        )
        scale = torch.clamp(
            float(max_displacement_rms) / (displacement_rms + 1.0e-12),
            max=1.0,
        )
        displacement = displacement * scale

    stochastic_displacement = torch.sqrt(posterior_variance) * innovation
    if max_displacement_to_noise_rms_ratio is not None:
        displacement_rms = torch.sqrt(
            torch.mean(displacement**2, dim=1, keepdim=True)
        )
        stochastic_rms = torch.sqrt(
            torch.mean(stochastic_displacement**2, dim=1, keepdim=True)
        )
        maximum_rms = (
            float(max_displacement_to_noise_rms_ratio) * stochastic_rms
        )
        scale = torch.clamp(
            maximum_rms / (displacement_rms + 1.0e-12),
            max=1.0,
        )
        displacement = displacement * scale

    return (
        model_mean
        + displacement
        + stochastic_displacement
    )


class DiffusionEnv:
    def __init__(self, DDPM_Dict, X):
        
        
        self.DDPM_Dict = DDPM_Dict
        
        self.device =torch.device(self.DDPM_Dict['device_name'])
        
        #Build the Diffusion Network
        self.diffusion = Denoise_ResNet_Model(self.DDPM_Dict,device=self.device)

        self.dataLength = len(X)
        self.batch_size = min(self.DDPM_Dict['batch_size'], self.dataLength)


        if 'idx_class_OHE' in self.DDPM_Dict:
            self.idx_class_OHE = self.DDPM_Dict['idx_class_OHE'] # indices of the columns to use for classifier free guidance conditioning
            zeros = np.zeros((len(self.idx_class_OHE),), dtype='float32')
            self.zeros_class_OHE = zeros # zero vector for the class label indices, used in run_CFG_step function
            #print('zeros ohe shape: ', self.zeros_class_OHE.shape)

            self.CFG_prob = self.DDPM_Dict.get('CFG_prob', 0.2)
            self.OG_CLS = X[:,self.idx_class_OHE].copy()
        
        self.data_norm = self.DDPM_Dict.get('pre_fitted_data_normalizer')
        if self.data_norm is None:
            self.data_norm = Data_Normalizer(
                np.array(self.DDPM_Dict['X_LL']),
                np.array(self.DDPM_Dict['X_UL']),
                self.dataLength,
                self.DDPM_Dict.get('data_normalizer_type', 'quantile'),
            )
            self.X = self.data_norm.fit_Data(X)
        else:
            self.X = self.data_norm.transform_Data(X)

        if hasattr(self, 'idx_class_OHE'):
            null_physical = np.zeros((1, len(self.DDPM_Dict['X_LL'])))
            null_normalized = self.data_norm.transform_Data(null_physical)
            self.null_class_normalized = torch.from_numpy(
                null_normalized[0, self.idx_class_OHE].astype('float32')
            ).to(self.device)

        #X needs to be a torch tensor of type float32 for the model, and on the correct device
        self.X = torch.from_numpy(self.X.astype('float32'))
         
        self.diffusion.to(self.device)
        
        self.X = self.X.to(self.device)

        # Sampling behavior is configured independently of learned weights so
        # existing checkpoints remain loadable.
        self._configure_sampling_options()

        # Initialize classifier for guided sampling (optional)
        self.classifier = None
        self.classifier_optimizer = None
        self.classifier_trained = False
    
        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_epochs = self.DDPM_Dict['Training_Epochs']
        self.init_lr = self.DDPM_Dict['lr']
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.init_lr, weight_decay=self.DDPM_Dict['weight_decay'])
       
        
        self.log_every = 100
        if 'rebatch_every' in self.DDPM_Dict:
            self.rebatch_every = self.DDPM_Dict['rebatch_every']
        else:
            self.rebatch_every = 100
        self.print_every = 10000
        self.loss_history = []
        

        #Set up alpha terms
        self.betas = make_beta_schedule(self.DDPM_Dict, self.device)
        # The optional L2 limit is sqrt(beta_t) * sqrt(generated coordinates).
        self.guidance_step_l2_limits = (
            torch.sqrt(self.betas) * math.sqrt(self.DDPM_Dict['xdim'])
        )
        
        self.alphas = 1. - self.betas
        
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        
        self.log_1_min_alpha = log_1_min_a(self.log_alpha)
        self.log_1_min_cumprod_alpha = log_1_min_a(self.log_cumprod_alpha)
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # At t=0, alpha_bar_{t-1}=1.  Padding with zero incorrectly gives a
        # nonzero posterior variance for the final reverse transition.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 1)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
   
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Start the training model functions
    """
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)

    def _configure_sampling_options(self):
        """Load reverse-sampling controls without changing checkpoint state."""

        # A design must retain the coordinates produced by the denoiser.
        # Reject stale configurations that request predicted-x0 clipping.
        self.sampling_x0_clip = bool(
            self.DDPM_Dict.get('sampling_x0_clip', False)
        )
        if self.sampling_x0_clip:
            raise ValueError(
                'Predicted-x0 clipping is disabled for GenISys sampling.'
            )

        self.classifier_guidance_mode = str(
            self.DDPM_Dict.get(
                'classifier_guidance_mode',
                'legacy_noise_blend',
            )
        )
        self.guidance_probability_threshold = float(
            self.DDPM_Dict.get('guidance_probability_threshold', 0.5)
        )
        self.guidance_relu_mode = str(
            self.DDPM_Dict.get('guidance_relu_mode', 'hard')
        )
        self.guidance_probability_objective = str(
            self.DDPM_Dict.get('guidance_probability_objective', 'log')
        )
        self.guidance_step_cap = bool(
            self.DDPM_Dict.get('guidance_step_cap', False)
        )
        self.guidance_normalize_l2 = bool(
            self.DDPM_Dict.get('guidance_normalize_l2', False)
        )
        self.guidance_probability_floor = float(
            self.DDPM_Dict.get('guidance_probability_floor', 1.0e-6)
        )
        self.guidance_gradient_floor = float(
            self.DDPM_Dict.get('guidance_gradient_floor', 1.0e-12)
        )
        self.guidance_max_displacement_rms = self.DDPM_Dict.get(
            'guidance_max_displacement_rms'
        )
        self.guidance_max_displacement_to_noise_rms_ratio = (
            self.DDPM_Dict.get(
                'guidance_max_displacement_to_noise_rms_ratio'
            )
        )
        self.guidance_active_timestep_fraction = float(
            self.DDPM_Dict.get('guidance_active_timestep_fraction', 1.0)
        )
        self.guidance_active_timestep_window = self.DDPM_Dict.get(
            'guidance_active_timestep_window'
        )
        self.guidance_balance_combined_gradient_rms = bool(
            self.DDPM_Dict.get(
                "guidance_balance_combined_gradient_rms",
                False,
            )
        )
        self.guidance_collect_diagnostics = bool(
            self.DDPM_Dict.get('guidance_collect_diagnostics', False)
        )
        self.guidance_diagnostics = []

    def reverse_step_indices(self):
        """Use C_ShipGen's t=1 stop only when its recipe requests it."""

        final_stop = 0 if self.DDPM_Dict.get("sampling_omit_t0", False) else -1
        return range(self.timesteps - 1, final_stop, -1)

    def _guidance_timestep_is_active(
        self,
        timestep,
        *,
        active_timestep_fraction=None,
        active_timestep_window=None,
    ):
        """Return whether guidance is active at one reverse-process step.

        Reverse progress is zero at the initial, highest-noise state and one
        at the final denoising state. A window is ``[start, end]`` in those
        coordinates. A fraction is shorthand for the final fraction of the
        reverse process. A call-level window or fraction takes precedence over
        the corresponding environment settings.
        """

        denominator = max(self.timesteps - 1, 1)
        reverse_progress = (
            self.timesteps - 1 - int(timestep)
        ) / denominator

        if active_timestep_window is not None:
            start, end = active_timestep_window
            return float(start) <= reverse_progress <= float(end)

        if active_timestep_fraction is not None:
            fraction = float(active_timestep_fraction)
            return reverse_progress >= 1.0 - fraction

        if self.guidance_active_timestep_window is not None:
            start, end = self.guidance_active_timestep_window
            return float(start) <= reverse_progress <= float(end)

        return (
            reverse_progress
            >= 1.0 - self.guidance_active_timestep_fraction
        )

    def _record_guidance_diagnostics(self, timestep, diagnostics):
        if self.guidance_collect_diagnostics:
            self.guidance_diagnostics.append(
                {'timestep': int(timestep), **diagnostics}
            )
    
    def _anneal_lr(self, epoch_step):
        #Update the learning rate
        frac_done = epoch_step / self.num_epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _as_condition_list(self, cond):
        if not isinstance(cond, (list, tuple)):
            return [cond]
        return list(cond)

    def _prepare_condition_batch(self, cond, num_samples):
        if torch.is_tensor(cond):
            cond = cond.detach().cpu().numpy()
        else:
            cond = np.asarray(cond)

        if cond.ndim == 1:
            cond = cond.reshape(1, -1)

        if cond.shape[0] == 1 and num_samples > 1:
            cond = np.repeat(cond, num_samples, axis=0)
        elif cond.shape[0] not in (1, num_samples):
            raise ValueError(
                f"Condition batch has incompatible shape {cond.shape}; expected 1 or {num_samples} rows"
            )

        return cond

    def _class_target_to_targets(self, class_target):
        if not hasattr(self, 'idx_class_OHE'):
            raise ValueError('class_target was provided, but this model does not define idx_class_OHE')

        if torch.is_tensor(class_target):
            class_target = class_target.detach().cpu().numpy()

        class_target = np.asarray(class_target, dtype=np.float32)
        if class_target.ndim == 2 and class_target.shape[0] == 1:
            class_target = class_target[0]

        targets = np.full((self.diffusion.xdim,), np.nan, dtype=np.float32)
        targets[self.idx_class_OHE] = class_target
        return targets
    '''
    =========================================================================
    Vanilla Diffusion
    ==========================================================================
    '''
    def q_sample(self,x_start, t, noise=None):
        """
        qsample from https://huggingface.co/blog/annotated-diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
    
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    def p_loss(self,x_start,t, noise=None,loss_type='l1'):
        '''
        from https://huggingface.co/blog/annotated-diffusion
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, t)
        
        #predicted_noise = predicted_noise.clamp(-3,3)
        
        if loss_type == 'l1':
            loss1 = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss1 = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss1 = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        #loss2 = F.mse_loss(predicted_noise.mean(dim=0),self.mean_zeros) #batch mean loss
        
        #loss3 = F.mse_loss(predicted_noise.var(dim=0),self.var_sample) #batch mean variance loss suggesting tha
        
        
        return loss1 #, loss2 #, loss3
        


    def run_step(self, x):
        self.optimizer.zero_grad()

        x_train = x
        if hasattr(self, 'idx_class_OHE') and self.CFG_prob > 0.0:
            x_train = x.clone()
            dropped_rows = torch.rand(x.size(0), device=self.device) < self.CFG_prob
            row_indices = torch.nonzero(dropped_rows, as_tuple=False).flatten()
            if row_indices.numel() > 0:
                column_indices = torch.as_tensor(
                    self.idx_class_OHE,
                    dtype=torch.long,
                    device=self.device,
                )
                x_train[
                    row_indices[:, None],
                    column_indices[None, :],
                ] = self.null_class_normalized

        t = torch.randint(0,self.timesteps,(x.size(0),),device=self.device)
        loss1 = self.p_loss(x_train,t,loss_type='l2')
        
        loss = loss1 #+ loss2 #+ loss3
        loss.backward()
        self.optimizer.step()

        return loss
    
    def run_train_loop(self, batches_per_epoch=100):
        
        self.diffusion.train()

        # Training batches are sampled with replacement below, so the requested
        # optimizer budget must not be capped at one nominal pass through the
        # data.  The V1.1 comparison matches the *total* number of optimizer
        # updates across monolithic and decomposed architectures.
        batches_per_epoch = max(1, int(batches_per_epoch))

        x_batches = torch.full((batches_per_epoch, self.batch_size, self.diffusion.xdim), 0.0, dtype = torch.float32, device=self.device)

        


        for i in tqdm(range(self.num_epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            loss = 0
            if i % self.rebatch_every == 0: 
                
                for j in range(batches_per_epoch):
                    A = np.random.randint(0,len(self.X), self.batch_size)
                    x_batches[j] = self.X[A]
                
            #iterate over number of batches per epoch, and get a new batch each time from the dataloader
            # get random batches from train loader
        
            for x_batch in x_batches:
                
        
                loss = self.run_step(x_batch)
        

            self._anneal_lr(i)

            self.loss_history.append(
                [i + 1, float(loss.to('cpu').detach().numpy())]
            )
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)

    
    
    @torch.no_grad()
    def _reverse_mean_from_epsilon(self, x, t, predicted_noise):
        """Return the configured reverse posterior mean."""

        time = torch.full(
            (x.size(dim=0),),
            int(t),
            dtype=torch.int64,
            device=self.device,
        )
        betas_t = self.extract(self.betas, time, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        sqrt_recip_alphas_t = self.extract(
            self.sqrt_recip_alphas, time, x.shape
        )

        # Use the unconstrained epsilon-parameterized reverse mean. No
        # predicted clean design is projected into a coordinate interval.
        return sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

    @torch.no_grad()
    def _reverse_from_epsilon(self, x, t, predicted_noise, innovation=None):
        """Perform one reverse transition from one predicted-noise tensor."""

        time = torch.full(
            (x.size(dim=0),),
            int(t),
            dtype=torch.int64,
            device=self.device,
        )
        model_mean = self._reverse_mean_from_epsilon(
            x, t, predicted_noise
        )

        if t == 0:
            return model_mean

        posterior_variance_t = self.extract(
            self.posterior_variance, time, x.shape
        )
        noise = (
            torch.randn_like(x, device=self.device)
            if innovation is None
            else innovation
        )
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample(self, x, t):
        time = torch.full(
            (x.size(dim=0),),
            t,
            dtype=torch.int64,
            device=self.device,
        )
        predicted_noise = self.diffusion(x, time)
        return self._reverse_from_epsilon(x, t, predicted_noise)
       
    @torch.no_grad()
    def gen_samples(self, num_samples):
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):
            
          
            x_gen = self.p_sample(x_gen, i)
        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)

    @torch.no_grad()
    def gen_targeted_samples(self, num_samples, targets=None, class_target=None, mask=None, alpha=0.1, CFG_W=None, **kwargs):
        #targets is a numpy array with target values for certain parameters. all other parameters are nan.
        #mask is a numpy array with 1 where we want to target and 0 where we want regular sampling

        if targets is None and class_target is not None:
            if CFG_W is not None:
                return self.gen_CFG_samples(num_samples, class_target, CFG_W)
            targets = self._class_target_to_targets(class_target)

        if targets is None:
            raise ValueError('targets or class_target must be provided')

        # Assert that targets and mask have the same shape as the data
        #print(targets.shape)
        #print((self.diffusion.xdim,))
        assert targets.shape == (self.diffusion.xdim,)

        if mask is None:
            mask = np.zeros_like(targets)
            mask[~np.isnan(targets)] = 1
        else:
            assert mask.shape == (self.diffusion.xdim,)
        
        #Assign zero to any nan values in targets for the model to use, but we will ignore these in the loss and sampling
        targets = np.nan_to_num(targets, nan=0.0)


        TARG = self.data_norm.transform_Data(targets.reshape(1,-1)) # scale targets to same scale as data using the data normalizer
        #print(COND[0])

        TARG = torch.from_numpy(TARG.astype('float32')).to(self.device)
        MASK = torch.from_numpy(mask.astype('float32')).to(self.device)
        alpha = alpha
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):
            x_gen = self.p_sample(x_gen, i)
            x_gen = x_gen + alpha * (TARG - x_gen) * MASK 
            #x_gen[MASK==1] = COND[MASK==1]

        #print('final x_gen before forcing targ: ', x_gen[0])
        # Force TARG to x_gen
        x_gen = x_gen * (1-MASK) + TARG * MASK

        output = x_gen.cpu().detach().numpy()

            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)
    

    ''' 
    =========================================================================
    CLASSIFIER FREE GUIDANCE FUNCTIONS
    =========================================================================

    These functions are for training and sampling with classifier free guidance

    These functions assume the class labels are one hot encoded and concatenated to the input data, and that the indices of these class labels are specified in the DDPM_Dict with the key 'idx_class_OHE'
    
    '''
        
    @torch.no_grad()
    def gen_CFG_samples(self, num_samples, OHE_target, CFG_W = 1.0):
        '''
        Generates samples with classifier free guidance, 
        where CFG_W is the guidance weight. Higher values of CFG_W will result in samples that more closely follow the class labels, but
        may result in lower quality samples if too high. A value of 0.0 corresponds to no guidance

        class_OHE is a one hot encoded vectors of the class labels.
        '''
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)

        #print(x_gen.shape)

        class_target = np.zeros((len(x_gen[0]),), dtype='float32')
        class_target[self.idx_class_OHE] = OHE_target
        class_target = class_target.reshape(1,-1)
        class_target = self.data_norm.transform_Data(class_target) # transform class_target to normalized data
        class_target = torch.from_numpy(class_target.astype('float32')).to(self.device)
        #print('class target shape: ', class_target.shape)
        #rint('class target: ', class_target)

        null_class_target = np.zeros((len(x_gen[0]),), dtype='float32')
        null_class_target[self.idx_class_OHE] = np.zeros_like(OHE_target)
        null_class_target = null_class_target.reshape(1,-1)
        null_class_target = self.data_norm.transform_Data(null_class_target) # transform null_class_target to normalized data
        null_class_target = torch.from_numpy(null_class_target.astype('float32')).to(self.device)
        #print('null class target shape: ', null_class_target.shape)
        #print('null class target: ', null_class_target)
    
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):

            x_gen[:,self.idx_class_OHE] = class_target[:,self.idx_class_OHE]
            x_null = x_gen.clone().detach()
            
            x_null[:,self.idx_class_OHE] = null_class_target[:,self.idx_class_OHE]
            
            time = torch.full(
                (num_samples,),
                i,
                dtype=torch.int64,
                device=self.device,
            )
            eps_conditional = self.diffusion(x_gen, time)
            eps_null = self.diffusion(x_null, time)
            eps_guided = (
                (1.0 + CFG_W)*eps_conditional
                - CFG_W*eps_null
            )
            x_gen = self._reverse_from_epsilon(x_gen, i, eps_guided)
        
        x_gen[:,self.idx_class_OHE] = class_target[:,self.idx_class_OHE] # force class labels to be the intended class labels in the generated samples

        output = x_gen.cpu().detach().numpy()
            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)

    '''
    =========================================================================
    CLASSIFIER-GUIDED DIFFUSION FUNCTIONS
    =========================================================================
    
    These functions enable classifier-guided generation to steer sample generation
    toward constraint-satisfying designs.
    '''

    def train_classifier(
        self,
        constraint_data,
        class_labels,
        classifier_dict=None,
        batches_per_epoch=100,
        use_class_weighted_loss=False,
    ):
        """
        Train a constraint classifier on feasibility labels.
        
        Parameters
        ----------
        constraint_data : np.ndarray
            Design vectors for training (n_samples, xdim)
        class_labels : np.ndarray
            Binary feasibility labels (n_samples, 2) where [1,0]=feasible, [0,1]=infeasible
        classifier_dict : dict, optional
            Configuration for classifier. If None, uses default.
        batches_per_epoch : int
            Number of batches per training epoch
        use_class_weighted_loss : bool
            If True, weight binary cross-entropy by inverse class frequency.
        """
        # Set up classifier configuration
        if classifier_dict is None:
            classifier_dict = {
                'xdim': self.diffusion.xdim,
                'tdim': 128,
                'cdim': 2,
                'net': [256, 128],
                'learning_rate': 0.001,
                'Training_Epochs': 50,
            }
        else:
            classifier_dict = classifier_dict.copy()

        use_class_weighted_loss = bool(
            use_class_weighted_loss or classifier_dict.get('use_class_weighted_loss', False)
        )
        
        # Create classifier
        self.classifier = ConstraintClassifier(classifier_dict, device=self.device)
        self.classifier.to(self.device)
        
        # Normalize the training data using the same normalizer as the diffusion model
        constraint_data_norm = self.data_norm.transform_Data(constraint_data)
        
        # Convert data to tensors
        X_train = torch.from_numpy(constraint_data_norm.astype('float32')).to(self.device)
        Y_train = torch.from_numpy(class_labels.astype('float32')).to(self.device)

        class_weights = None
        if use_class_weighted_loss:
            class_counts = np.asarray(class_labels, dtype=np.float32).sum(axis=0)
            if np.any(class_counts <= 0):
                raise ValueError(
                    "Class-weighted classifier loss requires every class to have at least one sample; "
                    f"got class counts {class_counts.tolist()}"
                )
            total_samples = float(class_counts.sum())
            num_classes = float(len(class_counts))
            class_weights_np = total_samples / (num_classes * class_counts)
            class_weights = torch.from_numpy(class_weights_np.astype('float32')).to(self.device)
            print("Class-weighted classifier loss enabled.")
            print(f"  Class counts: {class_counts.astype(int).tolist()}")
            print(f"  Class weights: {class_weights_np.tolist()}")
        
        # Set up optimizer
        lr = classifier_dict.get('learning_rate', 0.001)
        self.classifier_optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        
        num_epochs = classifier_dict.get('Training_Epochs', 50)
        datalength = len(constraint_data)
        num_batches = datalength // self.batch_size
        batches_per_epoch = min(num_batches, batches_per_epoch)
        
        # Training loop
        print("Training Constraint Classifier...")
        self.classifier.train()
        
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0.0
            
            for batch_idx in range(batches_per_epoch):
                # Sample random batch
                indices = np.random.randint(0, datalength, self.batch_size)
                x_batch = X_train[indices]
                y_batch = Y_train[indices]
                
                # Forward pass
                self.classifier_optimizer.zero_grad()
                pred = self.classifier(x_batch)
                if class_weights is None:
                    loss = F.binary_cross_entropy(pred, y_batch)
                else:
                    per_output_loss = F.binary_cross_entropy(pred, y_batch, reduction='none')
                    per_sample_loss = per_output_loss.mean(dim=1)
                    sample_weights = class_weights[y_batch.argmax(dim=1)]
                    loss = (per_sample_loss * sample_weights).mean()
                
                # Backward pass
                loss.backward()
                self.classifier_optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % max(100, num_epochs // 1000) == 0:
                avg_loss = epoch_loss / batches_per_epoch
                # print scientific notation with 4 significant digits
                print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4e}")
        
        self.classifier.eval()
        self.classifier_trained = True
        print("Classifier Training Complete!")
    
    def _cond_fn(
        self,
        x,
        t,
        classifier,
        class_target=None,
        *,
        probability_threshold=None,
        relu_mode=None,
        return_diagnostics=False,
    ):
        """Compute the configured classifier guidance gradient."""

        classifier_time = t
        if classifier.time_conditioned and not torch.is_tensor(classifier_time):
            classifier_time = torch.full(
                (x.size(0),),
                int(classifier_time),
                dtype=torch.int64,
                device=x.device,
            )

        if self.classifier_guidance_mode in {'posterior_mean_hinge', 'cshipgen_noise_blend_hinge'}:
            return classifier_probability_hinge_gradient(
                classifier,
                x,
                classifier_time,
                class_target,
                probability_threshold=(
                    self.guidance_probability_threshold
                    if probability_threshold is None
                    else float(probability_threshold)
                ),
                relu_mode=(
                    self.guidance_relu_mode
                    if relu_mode is None
                    else str(relu_mode)
                ),
                probability_floor=self.guidance_probability_floor,
                gradient_floor=self.guidance_gradient_floor,
                probability_objective=self.guidance_probability_objective,
                return_diagnostics=return_diagnostics,
            )

        if self.classifier_guidance_mode != 'legacy_noise_blend':
            raise ValueError(
                'Unknown classifier_guidance_mode '
                f'{self.classifier_guidance_mode!r}.'
            )

        # Historical MSE guidance remains available for checkpoint/result
        # reproduction. New experiments use posterior_mean_hinge.
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            prediction = classifier(
                x_in,
                classifier_time if classifier.time_conditioned else None,
            )

            if class_target is None:
                target = torch.zeros_like(prediction, device=self.device)
                target[:, 0] = 1.0
            elif isinstance(class_target, (int, np.integer)):
                one_hot = torch.zeros(
                    (prediction.size(1),), device=self.device
                )
                one_hot[class_target] = 1.0
                target = one_hot.unsqueeze(0).expand(prediction.size(0), -1)
            else:
                target = torch.as_tensor(
                    class_target,
                    dtype=torch.float32,
                    device=self.device,
                )
                if target.dim() == 1:
                    target = target.unsqueeze(0).expand(prediction.size(0), -1)

            error = (target - prediction) ** 2.0
            gradient = -torch.autograd.grad(error.sum(), x_in)[0]

        if return_diagnostics:
            rms = torch.sqrt(torch.mean(gradient**2, dim=1))
            return gradient, {
                'active_output_fraction': 1.0,
                'active_sample_fraction': 1.0,
                'zero_gradient_fraction': float(
                    (rms <= self.guidance_gradient_floor)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                ),
                'mean_target_probability': float('nan'),
                'mean_gradient_rms': float(rms.mean().detach().cpu()),
            }
        return gradient

    def _p_sample_guided(
        self,
        x,
        t,
        classifier,
        alpha=1.0,
        class_target=None,
        *,
        probability_threshold=None,
        relu_mode=None,
        active_timestep_fraction=None,
        active_timestep_window=None,
        max_displacement_rms=None,
        max_displacement_to_noise_rms_ratio=None,
    ):
        """Perform one configured classifier-guided reverse transition."""

        time = torch.full(
            (x.size(dim=0),),
            t,
            dtype=torch.int64,
            device=self.device,
        )
        predicted_noise = self.diffusion(x, time)
        model_mean = self._reverse_mean_from_epsilon(
            x, t, predicted_noise
        )

        if t == 0:
            return model_mean

        posterior_variance_t = self.extract(
            self.posterior_variance, time, x.shape
        )
        noise = torch.randn_like(x, device=self.device)

        if self.classifier_guidance_mode == 'cshipgen_noise_blend_hinge':
            if float(alpha) == 0.0 or not self._guidance_timestep_is_active(t):
                return model_mean + torch.sqrt(posterior_variance_t) * noise
            gradient_result = self._cond_fn(
                x, time, classifier, class_target=class_target,
                return_diagnostics=self.guidance_collect_diagnostics,
            )
            if self.guidance_collect_diagnostics:
                cond_grad, diagnostics = gradient_result
                self._record_guidance_diagnostics(t, diagnostics)
            else:
                cond_grad = gradient_result
            if self.guidance_normalize_l2:
                cond_grad = normalize_classifier_gradient_l2(cond_grad)
            if self.guidance_step_cap:
                classifier_move = alpha * torch.sqrt(posterior_variance_t) * cond_grad
                classifier_move = cap_guidance_displacement(
                    classifier_move, self.guidance_step_l2_limits[t]
                )
                return (
                    model_mean
                    + torch.sqrt(posterior_variance_t) * (1.0 - alpha) * noise
                    + classifier_move
                )
            return model_mean + torch.sqrt(posterior_variance_t) * (
                (1.0 - alpha) * noise + alpha * cond_grad
            )

        if self.classifier_guidance_mode == 'legacy_noise_blend':
            cond_grad = self._cond_fn(
                model_mean,
                t,
                classifier,
                class_target=class_target,
            )
            noise_rms = torch.sqrt(
                torch.mean(noise**2, dim=1, keepdim=True)
            )
            grad_rms = torch.sqrt(
                torch.mean(cond_grad**2, dim=1, keepdim=True)
            )
            cond_grad_scaled = cond_grad * (
                noise_rms / (grad_rms + 1.0e-8)
            )
            guided_noise = (
                noise * (1.0 - alpha) + alpha * cond_grad_scaled
            )
            return (
                model_mean
                + torch.sqrt(posterior_variance_t) * guided_noise
            )

        if self.classifier_guidance_mode != 'posterior_mean_hinge':
            raise ValueError(
                'Unknown classifier_guidance_mode '
                f'{self.classifier_guidance_mode!r}.'
            )

        # Alpha zero is exactly the corresponding unguided transition under
        # identical random state: the classifier is skipped and noise is not
        # rescaled or blended.
        if float(alpha) == 0.0:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

        # Inactive steps are exactly the corresponding unguided transition.
        # The posterior innovation was already drawn above, preserving paired
        # random-number streams for controlled tuning comparisons.
        if not self._guidance_timestep_is_active(
            t,
            active_timestep_fraction=active_timestep_fraction,
            active_timestep_window=active_timestep_window,
        ):
            return model_mean + torch.sqrt(posterior_variance_t) * noise

        cond_grad, diagnostics = self._cond_fn(
            x,
            time,
            classifier,
            class_target=class_target,
            probability_threshold=probability_threshold,
            relu_mode=relu_mode,
            return_diagnostics=True,
        )
        self._record_guidance_diagnostics(t, diagnostics)
        return posterior_mean_guided_sample(
            model_mean,
            posterior_variance_t,
            noise,
            cond_grad,
            alpha,
            max_displacement_rms=(
                self.guidance_max_displacement_rms
                if max_displacement_rms is None
                else max_displacement_rms
            ),
            max_displacement_to_noise_rms_ratio=(
                self.guidance_max_displacement_to_noise_rms_ratio
                if max_displacement_to_noise_rms_ratio is None
                else max_displacement_to_noise_rms_ratio
            ),
        )

    @torch.no_grad()
    def generate_guided_samples(
        self,
        num_samples,
        guidance_model,
        alpha=0.5,
        *,
        probability_threshold=None,
        relu_mode=None,
        active_timestep_fraction=None,
        active_timestep_window=None,
        max_displacement_rms=None,
        max_displacement_to_noise_rms_ratio=None,
    ):
        """Generate samples using configured constraint classifier guidance."""

        if not isinstance(guidance_model, ConstraintClassifier):
            raise ValueError(
                "guidance_model must be a ConstraintClassifier instance"
            )

        x_gen = torch.randn(
            (num_samples, self.diffusion.xdim), device=self.device
        )

        self.diffusion.eval()
        guidance_model.eval()
        self.guidance_diagnostics = []

        print(
            f"Generating {num_samples} guided samples "
            f"(alpha={alpha}, mode={self.classifier_guidance_mode})..."
        )
        for i in tqdm(self.reverse_step_indices()):
            x_gen = self._p_sample_guided(
                x_gen,
                i,
                guidance_model,
                alpha=alpha,
                probability_threshold=probability_threshold,
                relu_mode=relu_mode,
                active_timestep_fraction=active_timestep_fraction,
                active_timestep_window=active_timestep_window,
                max_displacement_rms=max_displacement_rms,
                max_displacement_to_noise_rms_ratio=(
                    max_displacement_to_noise_rms_ratio
                ),
            )

        output = x_gen.cpu().detach().numpy()
        output_scaled = self.data_norm.scale_X(output)

        return SampleResult(output_scaled, x_gen)

    def load_trained_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.diffusion.load_state_dict(torch.load(PATH))
    
    def Load_DDPM_Dict(PATH):
        #returns the dictionary for the DDPM_Dictionary to rebuild the model
        #PATH is the path including file name and extension of the json file that stores it. 
        f = open(PATH)
        return json.loads(f)
        
    
    def Save_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.diffusion.state_dict(), PATH+name+'.pth')
        
        JSON = json.dumps(self.DDPM_Dict)
        f = open(PATH+name+'_DDPM_Dict.json', 'w')
        f.write(JSON)
        f.close()


class Conditional_DiffusionEnv(DiffusionEnv):

    def __init__(self, DDPM_Dict, X, Cond):
        
        self.DDPM_Dict = DDPM_Dict
        

        self.cdim = self.DDPM_Dict['cdim']
        self.device =torch.device(self.DDPM_Dict['device_name'])
        self.Cond_type = self.DDPM_Dict['Cond_type']
        self.context_training_mode = self.DDPM_Dict.get(
            'context_training_mode',
            'clean',
        )
        self.noise_condition_indices = tuple(
            self.DDPM_Dict.get('noise_condition_indices', ())
        )

        self.dataLength = len(X)
        self.batch_size = min(self.DDPM_Dict['batch_size'], self.dataLength)

        self.cond_norm = []
        self.Cond = []
        self.Cond_LL = self.DDPM_Dict['Cond_LL']
        self.Cond_UL = self.DDPM_Dict['Cond_UL']

        fitted_conditions = self.DDPM_Dict.get('pre_fitted_condition_normalizers')
        for i in range(len(self.cdim)):
            fitted = None if fitted_conditions is None else fitted_conditions[i]
            if fitted is None:
                normalizer = Data_Normalizer(
                    np.array(self.Cond_LL[i]), np.array(self.Cond_UL[i]),
                    self.dataLength, self.Cond_type[i],
                )
                Cond_scaled = normalizer.fit_Data(Cond[i])
            else:
                normalizer = fitted
                Cond_scaled = normalizer.transform_Data(Cond[i])
            self.cond_norm.append(normalizer)
            self.Cond.append(
                torch.from_numpy(Cond_scaled.astype('float32')).to(self.device)
            )

        self.idx_CFG_Cond = None
        if 'idx_CFG_Cond' in self.DDPM_Dict:
            self.idx_CFG_Cond = self.DDPM_Dict['idx_CFG_Cond'] # indices of the columns to use for classifier free guidance conditioning
            zeros = np.zeros((1,self.cdim[self.idx_CFG_Cond]))
            self.zeros_class_OHE = zeros # zero vector for the class label indices, used in run_CFG_step function
            #print('zeros ohe shape: ', self.zeros_class_OHE.shape)

            self.CFG_prob = self.DDPM_Dict.get('CFG_prob', 0.2)
            self.OG_CLS = np.asarray(Cond[self.idx_CFG_Cond]).copy()
            null_physical = np.zeros((1, self.cdim[self.idx_CFG_Cond]))
            null_normalized = self.cond_norm[
                self.idx_CFG_Cond
            ].transform_Data(null_physical)
            self.null_condition_normalized = torch.from_numpy(
                null_normalized.astype('float32')
            ).to(self.device)
        
        self.data_norm = self.DDPM_Dict.get('pre_fitted_data_normalizer')
        if self.data_norm is None:
            self.data_norm = Data_Normalizer(
                np.array(self.DDPM_Dict['X_LL']),
                np.array(self.DDPM_Dict['X_UL']),
                self.dataLength,
                self.DDPM_Dict.get('data_normalizer_type', 'quantile'),
            )
            self.X = self.data_norm.fit_Data(X)
        else:
            self.X = self.data_norm.transform_Data(X)
        self.X = torch.from_numpy(self.X.astype('float32'))
        self.X = self.X.to(self.device)

        # Use the same reverse-sampling controls as the unconditional model.
        self._configure_sampling_options()
        
        
        #Build the Diffusion Network
        self.diffusion = Cond_Denoise_ResNet_Model(self.DDPM_Dict,device=self.device)
        self.diffusion.to(self.device)

        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_epochs = self.DDPM_Dict['Training_Epochs']
        self.init_lr = self.DDPM_Dict['lr']
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.init_lr, weight_decay=self.DDPM_Dict['weight_decay'])
       
        
        self.log_every = 100
        if 'rebatch_every' in self.DDPM_Dict:
            self.rebatch_every = self.DDPM_Dict['rebatch_every']
        else:
            self.rebatch_every = 100
        self.print_every = 10000
        self.loss_history = []
        

        #Set up alpha terms
        self.betas = make_beta_schedule(self.DDPM_Dict, self.device)
        # The optional L2 limit is sqrt(beta_t) * sqrt(generated coordinates).
        self.guidance_step_l2_limits = (
            torch.sqrt(self.betas) * math.sqrt(self.DDPM_Dict['xdim'])
        )
        
        self.alphas = 1. - self.betas
        
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        
        self.log_1_min_alpha = log_1_min_a(self.log_alpha)
        self.log_1_min_cumprod_alpha = log_1_min_a(self.log_cumprod_alpha)
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 1)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
   
    """
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Start the training model functions
    """
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)
    
    def _anneal_lr(self, epoch_step):
        #Update the learning rate
        frac_done = epoch_step / self.num_epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    
    def p_loss(self,x_start, t, cond, noise=None,loss_type='l2'):
        '''
        from https://huggingface.co/blog/annotated-diffusion
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, cond, t)
        
        #predicted_noise = predicted_noise.clamp(-3,3)
        
        if loss_type == 'l1':
            loss1 = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss1 = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss1 = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
    
        return loss1
    
    def run_step(self, x, cond):
        self.optimizer.zero_grad()

        cond_train = cond
        if self.idx_CFG_Cond is not None and self.CFG_prob > 0.0:
            cond_train = [value.clone() for value in cond]
            dropped_rows = torch.rand(x.size(0), device=self.device) < self.CFG_prob
            if bool(dropped_rows.any()):
                cond_train[self.idx_CFG_Cond][dropped_rows] = (
                    self.null_condition_normalized
                )

        t = torch.randint(0,self.timesteps,(x.size(0),),device=self.device)

        # SCDDPM samples each block against evolving noisy complementary
        # blocks.  The production treatment exposes the denoiser to that same
        # context distribution during training.  HCDDPM leaves this disabled
        # because its predecessor blocks are complete, clean samples.
        if self.context_training_mode != 'clean':
            cond_train = [value.clone() for value in cond_train]
            for condition_index in self.noise_condition_indices:
                condition = cond_train[condition_index]
                if self.context_training_mode == 'time_matched_noisy':
                    context_t = t
                elif self.context_training_mode == 'independently_noisy':
                    context_t = torch.randint(
                        0,
                        self.timesteps,
                        (x.size(0),),
                        device=self.device,
                    )
                else:
                    raise ValueError(
                        'Unknown context_training_mode: '
                        f'{self.context_training_mode!r}'
                    )

                sqrt_alpha = self.extract(
                    self.sqrt_alphas_cumprod,
                    context_t,
                    condition.shape,
                )
                sqrt_one_minus_alpha = self.extract(
                    self.sqrt_one_minus_alphas_cumprod,
                    context_t,
                    condition.shape,
                )
                cond_train[condition_index] = (
                    sqrt_alpha*condition
                    + sqrt_one_minus_alpha*torch.randn_like(condition)
                )

        loss1 = self.p_loss(x,t, cond_train, loss_type='l2')
        
        loss = loss1 #+ loss2 #+ loss3
        loss.backward()
        self.optimizer.step()

        return loss
    

    def run_train_loop(self, batches_per_epoch=100):
        
        self.diffusion.train()

        # These batches are also sampled with replacement.  Honour the runner's
        # explicit optimizer-step budget so a multi-block model does not receive
        # an accidental fourfold compute advantage over the monolithic model.
        batches_per_epoch = max(1, int(batches_per_epoch))

        x_batches = torch.full(
            (batches_per_epoch, self.batch_size, self.diffusion.xdim),
            0.0,
            dtype=torch.float32,
            device=self.device,
        )
        cond_batches = []
        
        for i in tqdm(range(self.num_epochs)):

            if i % self.rebatch_every == 0: 
                cond_batches = []
                for j in range(batches_per_epoch):
                    A = torch.randint(0, len(self.X), (self.batch_size,), device=self.device)
                    x_batches[j] = self.X[A]
                    cond_batch = [cond[A] for cond in self.Cond]

                    for k, cond_k in enumerate(cond_batch):
                        expected_shape = (self.batch_size, self.cdim[k])
                        if tuple(cond_k.shape) != expected_shape:
                            raise ValueError(
                                f"Cond batch shape mismatch at cond[{k}]: got {tuple(cond_k.shape)}, expected {expected_shape}"
                            )

                    cond_batches.append(cond_batch)

            for j in range(batches_per_epoch):
                
                loss = self.run_step(x_batches[j], cond_batches[j])
                


            self._anneal_lr(i)

            try:
                self.loss_history.append(
                    [i + 1, float(loss.to('cpu').detach().numpy())]
                )
            except Exception:
                self.loss_history.append([i + 1, float('nan')])
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)

   
    
    @torch.no_grad()
    def p_sample(self, x, cond, t):
        time = torch.full(
            (x.size(dim=0),),
            t,
            dtype=torch.int64,
            device=self.device,
        )
        predicted_noise = self.diffusion(x, cond, time)
        return self._reverse_from_epsilon(x, t, predicted_noise)

    def _prepare_cfg_condition_tensors(self, cond, num_samples):
        cond = self._as_condition_list(cond)

        if len(cond) != len(self.cdim):
            raise ValueError(
                f"Expected {len(self.cdim)} conditioning groups, received {len(cond)}"
            )

        cond_target = []
        null_cond_target = []

        for i in range(len(self.cdim)):
            cond_i = self._prepare_condition_batch(cond[i], num_samples)
            cond_i = self.cond_norm[i].transform_Data(cond_i)
            cond_target.append(torch.from_numpy(cond_i.astype('float32')).to(self.device))

            if self.idx_CFG_Cond is not None and i == self.idx_CFG_Cond:
                null_cond_i = np.zeros_like(
                    self._prepare_condition_batch(cond[i], num_samples)
                )
                null_cond_i = self.cond_norm[i].transform_Data(null_cond_i)
                null_cond_target.append(
                    torch.from_numpy(null_cond_i.astype('float32')).to(self.device)
                )
            else:
                # Classifier-free guidance drops only the declared condition.
                # Structural context must remain identical in both branches.
                null_cond_target.append(cond_target[-1].clone())

        return cond_target, null_cond_target

    @torch.no_grad()
    def p_sample_CFG(self, x, cond_target, null_cond_target, t, CFG_W = 1.0):
        time = torch.full(
            (x.size(dim=0),),
            t,
            dtype=torch.int64,
            device=self.device,
        )
        eps_conditional = self.diffusion(x, cond_target, time)
        eps_null = self.diffusion(x, null_cond_target, time)
        eps_guided = (1.0 + CFG_W)*eps_conditional - CFG_W*eps_null
        return self._reverse_from_epsilon(x, t, eps_guided)
       
    @torch.no_grad()
    def gen_samples(self, num_samples, cond):

        # Preserve one conditioning row per generated sample.  This helper also
        # moves CUDA tensors to CPU before the fitted NumPy/scikit transforms,
        # then returns the normalized conditions to the model device.
        cond = self._as_condition_list(cond)
        if len(cond) != len(self.cdim):
            raise ValueError(
                f"Expected {len(self.cdim)} conditioning groups, received "
                f"{len(cond)}"
            )
        cond = [
            torch.from_numpy(
                self.cond_norm[i]
                .transform_Data(
                    self._prepare_condition_batch(cond[i], num_samples)
                )
                .astype('float32')
            ).to(self.device)
            for i in range(len(self.cdim))
        ]
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):
            
          
            x_gen = self.p_sample(x_gen, cond, i)
            #x_gen = x_gen.clamp(-1,1)
        
        output = x_gen.cpu().detach().numpy().astype('float64')
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)#leave unnormed output in tensor form 
    
    @torch.no_grad()
    def gen_targeted_samples(self, num_samples, cond=None, targets=None, class_target=None, mask=None, alpha=0.1, CFG_W=None, **kwargs):
        #targets is a numpy array with target values for certain parameters. all other parameters are nan.
        #mask is a numpy array with 1 where we want to target and 0 where we want regular sampling
        # Only targets values in x. 

        if targets is None and class_target is not None:
            if CFG_W is not None:
                return self.gen_Cond_CFG_samples(num_samples, cond, CFG_W)
            targets = self._class_target_to_targets(class_target)

        if targets is None:
            raise ValueError('targets or class_target must be provided')

        if cond is not None:
            cond = self._as_condition_list(cond)
            if len(cond) != len(self.cdim):
                raise ValueError(
                    f"Expected {len(self.cdim)} conditioning groups, received "
                    f"{len(cond)}"
                )
            cond = [
                torch.from_numpy(
                    self.cond_norm[i]
                    .transform_Data(
                        self._prepare_condition_batch(cond[i], num_samples)
                    )
                    .astype('float32')
                ).to(self.device)
                for i in range(len(self.cdim))
            ]

        # Assert that targets and mask have the same shape as the data
        #print(targets.shape)
        #print((self.diffusion.xdim,))
        assert targets.shape == (self.diffusion.xdim,)

        if mask is None:
            mask = np.zeros_like(targets)
            mask[~np.isnan(targets)] = 1
        else:
            assert mask.shape == (self.diffusion.xdim,)
        
        #Assign zero to any nan values in targets for the model to use, but we will ignore these in the loss and sampling
        targets = np.nan_to_num(targets, nan=0.0)


        TARG = self.data_norm.transform_Data(targets.reshape(1,-1)) # scale targets to same scale as data using the data normalizer
        

        TARG = torch.from_numpy(TARG.astype('float32')).to(self.device)
        MASK = torch.from_numpy(mask.astype('float32')).to(self.device)
        alpha = alpha
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):
            x_gen = self.p_sample(x_gen,cond,  i)
            x_gen = x_gen + alpha * (TARG - x_gen) * MASK 
            #x_gen[MASK==1] = COND[MASK==1]

        #print('final x_gen before forcing targ: ', x_gen[0])
        # Force TARG to x_gen
        x_gen = x_gen * (1-MASK) + TARG * MASK

        output = x_gen.cpu().detach().numpy()

            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)
        

    
    @torch.no_grad()
    def gen_Cond_CFG_samples(self, num_samples, cond, CFG_W = 1.0):
        '''
        Generates samples with classifier free guidance, 
        where CFG_W is the guidance weight. Higher values of CFG_W will result in samples that more closely follow the class labels, but
        may result in lower quality samples if too high. A value of 0.0 corresponds to no guidance

        class_OHE is a one hot encoded vectors of the class labels.
        '''
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)

    
        cond_target = []
        cond = self._as_condition_list(cond)

        if len(cond) != len(self.cdim):
            raise ValueError(
                f"Expected {len(self.cdim)} conditioning groups, received {len(cond)}"
            )

        for i in range(len(self.cdim)):
            cond_i = self._prepare_condition_batch(cond[i], num_samples)
            cond_i = self.cond_norm[i].transform_Data(cond_i) # transform cond_i to normalized data
            cond_target.append(torch.from_numpy(cond_i.astype('float32')).to(self.device))

        
        null_cond_target = []
        for i in range(len(self.cdim)):
            if self.idx_CFG_Cond is not None and i == self.idx_CFG_Cond:
                null_cond_i = np.zeros_like(
                    self._prepare_condition_batch(cond[i], num_samples)
                )
                null_cond_i = self.cond_norm[i].transform_Data(null_cond_i)
                null_cond_target.append(
                    torch.from_numpy(null_cond_i.astype('float32')).to(self.device)
                )
            else:
                null_cond_target.append(cond_target[i].clone())
        
        self.diffusion.eval()
   
        for i in tqdm(self.reverse_step_indices()):

            
            x_gen = self.p_sample_CFG(
                x_gen,
                cond_target,
                null_cond_target,
                i,
                CFG_W,
            )
        
        
        output = x_gen.cpu().detach().numpy()
            
        output_scaled = self.data_norm.scale_X(output)
        
        return SampleResult(output_scaled, x_gen)

    @torch.no_grad()
    def gen_conditional_guided_samples(
        self,
        num_samples,
        cond,
        guidance_model,
        alpha=0.5,
        class_target=None,
        guidance_context_indices=None,
        *,
        probability_threshold=None,
        relu_mode=None,
        active_timestep_fraction=None,
        active_timestep_window=None,
        max_displacement_rms=None,
        max_displacement_to_noise_rms_ratio=None,
    ):
        """Generate conditional samples with configured classifier guidance."""

        if not isinstance(guidance_model, ConstraintClassifier):
            raise ValueError(
                "guidance_model must be a ConstraintClassifier instance"
            )

        cond_target, _ = self._prepare_cfg_condition_tensors(
            cond, num_samples
        )
        x_gen = torch.randn(
            (num_samples, self.diffusion.xdim), device=self.device
        )

        self.diffusion.eval()
        guidance_model.eval()
        self.guidance_diagnostics = []

        print(
            f"Generating {num_samples} conditional guided samples "
            f"(alpha={alpha}, mode={self.classifier_guidance_mode})..."
        )
        for i in tqdm(self.reverse_step_indices()):
            time = torch.full(
                (x_gen.size(0),),
                i,
                dtype=torch.int64,
                device=self.device,
            )
            predicted_noise = self.diffusion(
                x_gen, cond_target, time
            )
            model_mean = self._reverse_mean_from_epsilon(
                x_gen, i, predicted_noise
            )

            if i == 0:
                x_gen = model_mean
                continue

            posterior_variance_t = self.extract(
                self.posterior_variance, time, x_gen.shape
            )
            noise = torch.randn_like(x_gen, device=self.device)

            if self.classifier_guidance_mode == 'cshipgen_noise_blend_hinge':
                if float(alpha) == 0.0 or not self._guidance_timestep_is_active(i):
                    x_gen = model_mean + torch.sqrt(posterior_variance_t) * noise
                    continue

                classifier_values = x_gen
                if guidance_context_indices:
                    classifier_values = torch.cat(
                        [x_gen]
                        + [cond_target[index] for index in guidance_context_indices],
                        dim=1,
                    )
                gradient_result = self._cond_fn(
                    classifier_values, time, guidance_model,
                    class_target=class_target,
                    return_diagnostics=self.guidance_collect_diagnostics,
                )
                if self.guidance_collect_diagnostics:
                    cond_grad, diagnostics = gradient_result
                    self._record_guidance_diagnostics(i, diagnostics)
                else:
                    cond_grad = gradient_result
                cond_grad = cond_grad[:, : x_gen.shape[1]]
                if self.guidance_normalize_l2:
                    cond_grad = normalize_classifier_gradient_l2(cond_grad)
                if self.guidance_step_cap:
                    classifier_move = alpha * torch.sqrt(posterior_variance_t) * cond_grad
                    classifier_move = cap_guidance_displacement(
                        classifier_move, self.guidance_step_l2_limits[i]
                    )
                    x_gen = (
                        model_mean
                        + torch.sqrt(posterior_variance_t) * (1.0 - alpha) * noise
                        + classifier_move
                    )
                else:
                    x_gen = model_mean + torch.sqrt(posterior_variance_t) * (
                        (1.0 - alpha) * noise + alpha * cond_grad
                    )
                continue

            if self.classifier_guidance_mode == 'legacy_noise_blend':
                cond_grad = self._cond_fn(
                    model_mean,
                    time,
                    guidance_model,
                    class_target=class_target,
                )
                noise_rms = torch.sqrt(
                    torch.mean(noise**2, dim=1, keepdim=True)
                )
                grad_rms = torch.sqrt(
                    torch.mean(cond_grad**2, dim=1, keepdim=True)
                )
                cond_grad_scaled = cond_grad * (
                    noise_rms / (grad_rms + 1.0e-8)
                )
                guided_noise = (
                    noise * (1.0 - alpha) + alpha * cond_grad_scaled
                )
                x_gen = (
                    model_mean
                    + torch.sqrt(posterior_variance_t) * guided_noise
                )
                continue

            if self.classifier_guidance_mode != 'posterior_mean_hinge':
                raise ValueError(
                    'Unknown classifier_guidance_mode '
                    f'{self.classifier_guidance_mode!r}.'
                )

            if float(alpha) == 0.0:
                x_gen = (
                    model_mean
                    + torch.sqrt(posterior_variance_t) * noise
                )
                continue

            if not self._guidance_timestep_is_active(
                i,
                active_timestep_fraction=active_timestep_fraction,
                active_timestep_window=active_timestep_window,
            ):
                x_gen = (
                    model_mean
                    + torch.sqrt(posterior_variance_t) * noise
                )
                continue

            # Contextual HCDDPM classifiers receive the current noisy owned
            # block followed by fixed normalized predecessor blocks. Only the
            # owned-block part of the resulting gradient moves this denoiser.
            classifier_values = x_gen
            if guidance_context_indices:
                classifier_values = torch.cat(
                    [x_gen]
                    + [cond_target[index] for index in guidance_context_indices],
                    dim=1,
                )
            cond_grad, diagnostics = self._cond_fn(
                classifier_values,
                time,
                guidance_model,
                class_target=class_target,
                probability_threshold=probability_threshold,
                relu_mode=relu_mode,
                return_diagnostics=True,
            )
            cond_grad = cond_grad[:, : x_gen.shape[1]]
            self._record_guidance_diagnostics(i, diagnostics)
            x_gen = posterior_mean_guided_sample(
                model_mean,
                posterior_variance_t,
                noise,
                cond_grad,
                alpha,
                max_displacement_rms=(
                    self.guidance_max_displacement_rms
                    if max_displacement_rms is None
                    else max_displacement_rms
                ),
                max_displacement_to_noise_rms_ratio=(
                    self.guidance_max_displacement_to_noise_rms_ratio
                    if max_displacement_to_noise_rms_ratio is None
                    else max_displacement_to_noise_rms_ratio
                ),
            )

        output = x_gen.cpu().detach().numpy()
        output_scaled = self.data_norm.scale_X(output)
        return SampleResult(output_scaled, x_gen)


