from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler, DEISMultistepScheduler
import torch
import yaml
import math
import tqdm
import time

from diffusers.utils.torch_utils import randn_tensor


def masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor


def nan_masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = float('nan')
    return tensor


def length_to_mask(length, device: torch.device = None):
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    if isinstance(max_len, torch.Tensor):
        max_len = int(max_len.item())

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def compute_log_likelihood(x, mu, sigma):
    var = sigma ** 2 + 1e-8  # Ensure variance is > 0
    log_prob = -0.5 * (torch.log(2 * torch.pi * var) + ((x - mu) ** 2) / var)
    return log_prob


def get_parameters(
        scheduler,
        timestep,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,

):
    set_none = False
    if scheduler.step_index is None:
        set_none = True
        scheduler._init_step_index(timestep)

    # Improve numerical stability for small number of steps
    model_output = scheduler.convert_model_output(model_output, sample=sample)
    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    sigma_t, sigma_s = scheduler.sigmas[scheduler.step_index + 1], scheduler.sigmas[scheduler.step_index]
    alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s

    if scheduler.config.algorithm_type == "dpmsolver++":

        mean = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        sigma = 0

    elif scheduler.config.algorithm_type == "dpmsolver":
        mean = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        sigma = 0

    elif scheduler.config.algorithm_type == "sde-dpmsolver++":
        mean = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output

        )

        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))
    elif scheduler.config.algorithm_type == "sde-dpmsolver":

        mean = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output

        )
        sigma = sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0)
    sigma = torch.max(sigma, torch.tensor(1e-1, device=sigma.device))
    if set_none:
        scheduler._step_index = None
    return mean, sigma


class DiffusePipeline(object):

    def __init__(self, opt, model, diffuser_name, num_inference_steps, device, torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.diffuser_name = diffuser_name
        self.num_inference_steps = num_inference_steps
        if self.torch_dtype == torch.float16:
            model = model.half()
        self.model = model.to(device)
        self.opt = opt

        # Load parameters from YAML file
        with open('config/diffuser_params.yaml', 'r') as yaml_file:
            diffuser_params = yaml.safe_load(yaml_file)

        # Select diffusion'parameters based on diffuser_name
        if diffuser_name in diffuser_params:
            params = diffuser_params[diffuser_name]
            scheduler_class_name = params['scheduler_class']
            additional_params = params['additional_params']

            # align training parameters
            additional_params['num_train_timesteps'] = opt.diffusion_steps
            additional_params['beta_schedule'] = opt.beta_schedule
            additional_params['prediction_type'] = opt.prediction_type

            try:
                scheduler_class = globals()[scheduler_class_name]
            except KeyError:
                raise ValueError(f"Class '{scheduler_class_name}' not found.")

            self.scheduler = scheduler_class(**additional_params)
        else:
            raise ValueError(f"Unsupported diffuser_name: {diffuser_name}")

    def generate_batch(self, caption, m_lens):
        B = len(caption)
        T = m_lens.max()
        shape = (B, T, self.model.input_feats)

        # random sampling noise x_T
        sample = torch.randn(shape, device=self.device, dtype=self.torch_dtype)

        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [torch.tensor([t] * B, device=self.device).long() for t in self.scheduler.timesteps]

        # cache text_embedded 
        enc_text = self.model.encode_text(caption, self.device)

        for i, t in enumerate(timesteps):
            # 1. model predict 
            with torch.no_grad():
                if getattr(self.model, 'cond_mask_prob', 0) > 0:
                    predict = self.model.forward_with_cfg(sample, t, enc_text=enc_text)
                else:

                    predict = self.model(sample, t, enc_text=enc_text)

            # 2. compute less noisy motion and set x_t -> x_t-1
            sample = self.scheduler.step(predict, t[0], sample).prev_sample

        return sample

    def generate_batch_rl(self, caption, m_lens):
        B = len(caption)
        T = m_lens.max()
        shape = (B, T, self.model.input_feats)

        results = {}

        # random sampling noise x_T
        sample = torch.randn(shape, device=self.device, dtype=self.torch_dtype)

        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = [torch.tensor([t] * B, device=self.device).long() for t in self.scheduler.timesteps]

        # cache text_embedded
        enc_text = self.model.encode_text(caption, self.device)

        mask = length_to_mask(m_lens.to(self.device), device=self.device)

        for i, t in enumerate(timesteps):
            # 1. model predict
            with torch.no_grad():
                if getattr(self.model, 'cond_mask_prob', 0) > 0:
                    predict = self.model.forward_with_cfg(sample, t, enc_text=enc_text)
                else:
                    predict = self.model(sample, t, enc_text=enc_text)
            # mean, std = get_parameters(self.scheduler, t[0], predict, sample)
            old_sample = sample.clone()

            # 2. compute less noisy motion and set x_t -> x_t-1
            sus = self.scheduler.step(predict, t[0], sample)

            sample = sus.prev_sample

            # log_likelihood = compute_log_likelihood(sample, mean, std)
            # log_likelihood = nan_masked(log_likelihood, mask)
            # log_prob = log_likelihood.nanmean(dim=[1, 2])

            log_likelihood = sus.log_prob
            log_likelihood = nan_masked(log_likelihood, mask)
            log_prob = log_likelihood.nanmean(dim=[1, 2]).unsqueeze(-1)

            results[i] = {

                "t": t,  # begin the t
                "xt_old": old_sample.detach().cpu(),  # begin the xt
                "xt_new": sample.clone().detach().cpu(),  # begin the A when train PPO
                "log_prob": log_prob.detach().cpu(),

                "length": m_lens.detach().cpu(),
                "mask": mask.detach().cpu(),
                "enc_text": enc_text,

            }

        return sample, results

    # def get_loglike(self, enc_text, m_lens, actions, mask, state):
    #     B = enc_text.shape[0]
    #     T = m_lens.max()
    #     shape = (B, T, self.model.input_feats)
    #
    #     # set timesteps
    #     self.scheduler.set_timesteps(self.num_inference_steps, self.device)
    #     timesteps = [torch.tensor([t] * B, device=self.device).long() for t in self.scheduler.timesteps]
    #     log_prob = []
    #     for i, t in enumerate(timesteps):
    #         sample = state[:, i]
    #         # 1. model predict
    #         if getattr(self.model, 'cond_mask_prob', 0) > 0:
    #             predict = self.model.forward_with_cfg(sample, t, enc_text=enc_text)
    #         else:
    #             predict = self.model(sample, t, enc_text=enc_text)
    #
    #         old_sample = sample.clone()
    #
    #         # 2. compute less noisy motion and set x_t -> x_t-1
    #         sus = self.scheduler.step(predict, t[0], sample, action=actions[:, i])
    #
    #         sample = sus.prev_sample
    #
    #         # log_likelihood = compute_log_likelihood(sample, mean, std)
    #         log_likelihood = nan_masked(sus.log_prob, mask)
    #         log_prob_ = log_likelihood.nanmean(dim=[1, 2])
    #         log_prob.append(log_prob_.unsqueeze(-1))
    #         # log_prob.append(log_likelihood.nanmean(dim=[1, 2]).unsqueeze(-1))
    #
    #     return torch.cat(log_prob, dim=1)

    def get_loglike_aa(self, enc_text, m_lens, actions, mask, state):
        B = enc_text.shape[0]

        T = m_lens.max()

        # set timesteps for all steps at once
        self.scheduler.set_timesteps(self.num_inference_steps, self.device)
        timesteps = torch.tensor(self.scheduler.timesteps, device=self.device).long()  # shape (self.num_inference_steps,)
        timesteps = timesteps.unsqueeze(0).repeat(B, 1)  # shape (B, self.num_inference_steps)

        enc_text = enc_text.unsqueeze(1).repeat(1, self.num_inference_steps, 1, 1)

        timesteps = timesteps.reshape(B * self.num_inference_steps)
        enc_text = enc_text.reshape(B * self.num_inference_steps, enc_text.shape[-2], enc_text.shape[-1])
        state = state.reshape(B * self.num_inference_steps, T, state.shape[-1])

        # model forward for all timesteps at once
        if getattr(self.model, 'cond_mask_prob', 0) > 0:
            predict = self.model.forward_with_cfg(state, timesteps, enc_text=enc_text)
        else:
            predict = self.model(state, timesteps, enc_text=enc_text)

        timesteps = timesteps.reshape(B, self.num_inference_steps)
        state = state.reshape(B, self.num_inference_steps, T, state.shape[-1])
        predict = predict.reshape(B, self.num_inference_steps, T, state.shape[-1])

        log_prob = []
        for t in range(self.num_inference_steps):

            # scheduler step for all timesteps at once, assuming it supports batch mode
            sus = self.scheduler.step(predict[:, t], timesteps[0, t], state[:, t], action=actions[:, t])

            log_likelihood = sus.log_prob
            log_likelihood = nan_masked(log_likelihood, mask)
            log_prob.append(log_likelihood.nanmean(dim=[1, 2]).unsqueeze(-1))

        return torch.cat(log_prob, dim=1)

    def generate(self, caption, m_lens, batch_size=32):
        N = len(caption)
        infer_mode = ''
        if getattr(self.model, 'cond_mask_prob', 0) > 0:
            infer_mode = 'classifier-free-guidance'
        print(f'\nUsing {self.diffuser_name} diffusion scheduler to {infer_mode} generate {N} motions, sampling {self.num_inference_steps} steps.')
        self.model.eval()

        all_output = []
        t_sum = 0
        cur_idx = 0
        for bacth_idx in tqdm.tqdm(range(math.ceil(N / batch_size))):
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            torch.cuda.synchronize()
            start_time = time.time()
            output = self.generate_batch(batch_caption, batch_m_lens)
            torch.cuda.synchronize()
            now_time = time.time()

            # The average inference time is calculated after GPU warm-up in the first 50 steps.
            if (bacth_idx + 1) * self.num_inference_steps >= 50:
                t_sum += now_time - start_time

            # Crop motion with gt/predicted motion length
            B = output.shape[0]
            for i in range(B):
                all_output.append(output[i, :batch_m_lens[i]])

            cur_idx += batch_size

        # calcalate average inference time
        t_eval = t_sum / (bacth_idx - 1)
        print('The average generation time of a batch motion (bs=%d) is %f seconds' % (batch_size, t_eval))
        return all_output, t_eval
