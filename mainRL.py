import itertools
import os
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from matplotlib.pyplot import title
from omegaconf import DictConfig, OmegaConf
from torch import nn

from RL.reward_model import all_metrics, reward_model, get_motion_guofeats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate.utils import set_seed
from os.path import join as pjoin

from models import build_models
from models.gaussian_diffusion import DiffusePipeline, length_to_mask, masked
from motion_loader import get_dataset_loader
from options.evaluate_options import TestOptions
from options.generate_options import GenerateOptions
from src.config import read_config
# from src.model.text_encoder import TextToEmb
import wandb
from peft import LoraModel, LoraConfig, TaskType, get_peft_model

from RL.utils import render, get_embeddings, get_embeddings_2, freeze_normalization_layers, create_folder_results
from src.tools.extract_joints import extract_joints
from utils.model_load import load_model_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


class RLDataset(Dataset):
    def __init__(self, dataset_dict):
        """
        dataset_dict: a dictionary where each value is a tensor of shape (N, ...)
        All tensors should have the same first dimension size N.
        """
        self.dataset = dataset_dict
        # Verify all tensors have the same first dimension
        lengths = [v.size(0) for v in self.dataset.values()]
        assert all(length == lengths[0] for length in lengths), "All tensors must have the same first dimension size"
        self.length = lengths[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dict of all tensors for this index
        return {key: tensor[idx] for key, tensor in self.dataset.items()}


def render(x_starts, infos, smplh, joints_renderer, smpl_renderer, texts, file_path, ty_log, out_formats, video_log=False):
    # out_formats = ['txt', 'smpl', 'joints', 'txt', 'smpl', 'videojoints', 'videosmpl']
    tmp = file_path

    for idx, (x_start, length, text) in enumerate(zip(x_starts, infos["all_lengths"], texts)):

        if idx == 16:
            break #todo codice brutto

        x_start = x_start[:length]

        extracted_output = extract_joints(
            torch.tensor(x_start),
            "guoh3dfeats",
        )

        file_path = tmp + "/"
        os.makedirs(file_path, exist_ok=True)

        if "videojoints" in out_formats:
            video_path = file_path + str(idx) + "_joints.gif"
            render_text = text
            joints_renderer(extracted_output["joints"], title=render_text, output=video_path, canonicalize=False)
            if video_log:
                wandb.log({ty_log: {"Video-joints": wandb.Video(video_path, format="gif", caption=text)}})



def preload_tmr_text(dataloader):
    all_embeddings = []
    for batch_idx, batch in enumerate(dataloader):
        all_embeddings.append(batch["tmr_text"])
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


@torch.no_grad()
def generate(model, train_dataloader, iteration, c, device, infos, text_model, smplh,
             train_embedding_tmr):  # , generation_iter
    model.model.unet.train()
    model.model.textTransEncoder.train()
    model.model.embed_text.train()

    dataset = {
        "r": [],
        "xt_1": [],
        "xt": [],
        "t": [],
        "log_like": [],

        "mask": [],
        "length": [],
        "enc_text": [],
    }

    generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(train_dataloader), 1)),
                        desc=f"Iteration {iteration + 1}/{c.iterations} [Generate new dataset]",
                        total=1, leave=False)

    for batch_idx, batch in generate_bar:
        texts, _, motion_lens = batch

        motion_lens = torch.LongTensor([int(x) for x in motion_lens])

        texts = texts * c.num_gen_per_prompt
        motion_lens = motion_lens.repeat(c.num_gen_per_prompt)
        infos["all_lengths"] = motion_lens
        mask = length_to_mask(motion_lens.to(device), device=device)

        with torch.no_grad():
            animations, results_by_timestep = model.generate_batch_rl(texts, torch.LongTensor([int(x) for x in motion_lens]))  # TODO Check feet bs

        animations = masked(animations, mask)

        infos["all_lengths"] = motion_lens
        metrics_reward = reward_model(animations, infos, smplh, texts, c)

        has_nan = (
                any(torch.isnan(t.cpu()).any() for t in metrics_reward.values())
                or torch.isnan(animations.cpu()).any()
        )

        if has_nan:
            print("Found NaN in masked_tmr")
            save_dir = "NaN_folder"
            os.makedirs(save_dir, exist_ok=True)
            # Do something if there is at least one NaN
            try:
                masked_tmr_np = metrics_reward["tmr"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_with_nan.npy"), masked_tmr_np)

            except:
                masked_tmr_plus_np = metrics_reward["tmr++"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_plus_with_nan.npy"), masked_tmr_plus_np)
            # Save to .npy file
            np.save(os.path.join(save_dir, "sequences.npy"), animations.cpu().numpy())
            exit()

        timesteps = sorted(results_by_timestep.keys(), reverse=False)
        diff_step = len(timesteps)

        batch_size = animations.shape[0]
        seq_len = results_by_timestep[0]["xt_new"].shape[1]

        # Store text embeddings just once, with repeat handling during concatenation
        all_rewards = []

        all_xt_new = []
        all_xt_old = []
        all_t = []
        all_log_probs = []

        # y
        all_mask = []
        all_lengths = []
        all_enc_text = []

        for t in timesteps:
            experiment = results_by_timestep[t]
            experiment = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in experiment.items()}

            if t == timesteps[-1]:
                all_rewards.append(metrics_reward["reward"].cpu())
            else:
                all_rewards.append(torch.zeros_like(metrics_reward["reward"]).cpu())

            all_xt_new.append(experiment["xt_new"])
            all_xt_old.append(experiment["xt_old"])
            all_t.append(torch.full((batch_size,), t, device=metrics_reward["reward"].device).cpu())
            all_log_probs.append(experiment["log_prob"])

            # y
            all_mask.append(experiment["mask"])
            all_lengths.append(experiment["length"])
            all_enc_text.append(experiment["enc_text"])

        # Concatenate all the results for this batch
        dataset["r"].append(torch.cat(all_rewards, dim=0).view(diff_step, batch_size).T.clone())
        dataset["xt_1"].append(
            torch.cat(all_xt_new, dim=0).view(diff_step, batch_size, seq_len, 263).permute(1, 0, 2, 3))
        dataset["xt"].append(
            torch.cat(all_xt_old, dim=0).view(diff_step, batch_size, seq_len, 263).permute(1, 0, 2, 3))
        dataset["t"].append(torch.cat(all_t, dim=0).view(diff_step, batch_size).T)
        dataset["log_like"].append(torch.cat(all_log_probs, dim=0).view(diff_step, batch_size).T)

        # y
        dataset["mask"].append(torch.cat(all_mask, dim=0).view(diff_step, batch_size, seq_len).permute(1, 0, 2))
        dataset["length"].append(torch.cat(all_lengths, dim=0).view(diff_step, batch_size).T)
        dataset["enc_text"].append(torch.cat(all_enc_text, dim=0).view(diff_step, batch_size, 77, 256).permute(1, 0, 2, 3))

    for key in dataset:
        dataset[key] = torch.cat(dataset[key], dim=0)

    return dataset


# def get_batch(dataset, i, minibatch_size, infos, diff_step, device):
#     enc_text = dataset["enc_text"][i: i + minibatch_size].to(device)
#     mask = dataset["mask"][i: i + minibatch_size].to(device)
#     lengths = dataset["length"][i: i + minibatch_size].to(device)
#     r = dataset["r"][i: i + minibatch_size].to(device)
#     xt_1 = dataset["xt_1"][i: i + minibatch_size].to(device)
#     xt = dataset["xt"][i: i + minibatch_size].to(device)
#     t = dataset["t"][i: i + minibatch_size].to(device)
#     log_like = dataset["log_like"][i: i + minibatch_size].to(device)
#
#     return enc_text, mask, lengths, r, xt_1, xt, t, log_like
#

def prepare_dataset(dataset):
    dataset_size = dataset["r"].shape[0]
    shuffle_indices = torch.randperm(dataset_size)

    for key in dataset:
        dataset[key] = dataset[key][shuffle_indices]

    return dataset


def train(model, optimizer, dataset, iteration, c, infos, device, accelerator, old_model=None):
    model.model.unet.train()
    model.model.textTransEncoder.train()
    model.model.embed_text.train()

    delta = 1e-7
    mask = dataset["r"] != 0
    mean_r = torch.mean(dataset["r"][mask], dim=0)
    std_r = torch.std(dataset["r"][mask], dim=0)

    wandb.log({"Train": {"Mean Reward": mean_r.item(), "Std Reward": std_r.item(), "iterations": iteration}})

    dataset["advantage"] = torch.zeros_like(dataset["r"])
    dataset["advantage"][mask] = (dataset["r"][mask] - mean_r) / (std_r + delta)
    dataset["advantage"] = (dataset["r"] - mean_r) / (std_r + delta)

    num_minibatches = (dataset["r"].shape[0] + c.train_batch_size - 1) // c.train_batch_size

    diff_step = dataset["xt_1"][0].shape[0]

    my_dataset = RLDataset(dataset)
    dataloader = DataLoader(my_dataset, batch_size=c.train_batch_size, shuffle=True, drop_last=False)


    model, optimizer, training_dataloader, _ = accelerator.prepare(
        model, optimizer, dataloader, None)

    train_bar = tqdm(range(c.train_epochs), desc=f"Iteration {iteration + 1}/{c.iterations} [Train]", leave=False)
    for e in train_bar:
        tot_loss = 0
        tot_kl = 0
        tot_policy_loss = 0
        epoch_clipped_elements = 0
        epoch_total_elements = 0

        minibatch_bar = tqdm(dataloader, leave=False, desc="Minibatch")
        dataset = prepare_dataset(dataset)

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(minibatch_bar):
            optimizer.zero_grad()


            advantage = batch["advantage"] #.to(device)
            enc_text = batch["enc_text"] #.to(device)
            mask = batch["mask"]  #.to(device)
            lengths = batch["length"] #.to(device)
            xt_1 = batch["xt_1"]  #.to(device)
            xt = batch["xt"]  #.to(device)
            log_like = batch["log_like"]  #.to(device)
            with accelerator.autocast():
                new_log_like = model.get_loglike_aa(enc_text[:, 0], lengths, xt_1, mask[:, 0], xt)  # TODO Check feet bs

                ratio = torch.exp(new_log_like - log_like)
                # torch.set_printoptions(precision=4)

                real_adv = advantage[:, -1:]  # r[:,-1:]/10

                # Count how many elements need clipping
                lower_bound = 1.0 - c.advantage_clip_epsilon
                upper_bound = 1.0 + c.advantage_clip_epsilon

                too_small = (ratio < lower_bound).sum().item()
                too_large = (ratio > upper_bound).sum().item()
                current_clipped = too_small + too_large
                epoch_clipped_elements += current_clipped
                current_total = ratio.numel()
                epoch_total_elements += current_total

                clip_adv = torch.clamp(ratio, lower_bound, upper_bound) * real_adv
                policy_loss = -torch.min(ratio * real_adv, clip_adv).sum(1).mean()

                combined_loss = c.alphaL * policy_loss

            # combined_loss.backward()
            accelerator.backward(combined_loss)
            tot_loss += combined_loss.item()
            tot_policy_loss += policy_loss.item()


            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in model.model.parameters() if p.grad is not None))
            wandb.log({"Train": {"Gradient Norm": grad_norm.item(),
                                 "real_step": (iteration * c.train_epochs + e) * num_minibatches + (
                                         batch_idx // c.train_batch_size)}})

            torch.nn.utils.clip_grad_norm_(model.model.parameters(), c.grad_clip)
            optimizer.step()


            minibatch_bar.set_postfix(batch_loss=f"{combined_loss.item():.4f}")

        epoch_loss = tot_loss / num_minibatches
        epoch_policy_loss = tot_policy_loss / num_minibatches
        clipping_percentage = 100 * epoch_clipped_elements / epoch_total_elements

        train_bar.set_postfix(epoch_loss=f"{epoch_loss:.4f}")

        if c.betaL > 0:
            epoch_kl = tot_kl / num_minibatches
            wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "kl_loss": epoch_kl,
                                 "trigger-clip": clipping_percentage}})
        else:
            wandb.log({"Train": {"loss": epoch_loss, "epochs": iteration * c.train_epochs + e,
                                 "policy_loss": epoch_policy_loss, "trigger-clip": clipping_percentage}})


def test(model, dataloader, device, infos, text_model, smplh, joints_renderer, smpl_renderer, c, all_embedding_tmr,
         path):
    os.makedirs(path, exist_ok=True)
    out_formats = ['txt', 'smpl', 'joints', 'videojoints']

    ty_log = ""
    if "VAL" in path:
        ty_log = "Validation"
    else:
        ty_log = "Test"

    model.model.eval()

    if c.val_num_batch == 0:
        generate_bar = tqdm(enumerate(dataloader), leave=False, desc=f"[Validation/Test Generations]",
                            total=len(dataloader))
    else:
        generate_bar = tqdm(enumerate(itertools.islice(itertools.cycle(dataloader), c.val_num_batch)),
                            total=c.val_num_batch, leave=False, desc=f"[Validation/Test Generations]")

    total_reward, total_tmr, total_tmr_plus_plus, total_tmr_guo = 0, 0, 0, 0
    batch_count_reward, batch_count_tmr, batch_count_tmr_plus_plus, batch_count_guo = 0, 0, 0, 0

    for batch_idx, batch in generate_bar:
        tmp_path = path + "batch_" + str(batch_idx) + "/"
        os.makedirs(tmp_path, exist_ok=True)

        texts, _, motion_lens = batch

        motion_lens = torch.LongTensor([int(x) for x in motion_lens])
        mask = length_to_mask(motion_lens.to(device), device=device)

        with torch.no_grad():
            animations = model.generate_batch(texts, motion_lens)  # TODO Check feet bs
        animations = masked(animations, mask)

        infos["all_lengths"] = motion_lens

        metrics_reward = reward_model(animations, infos, smplh, texts, c)
        metrics = all_metrics(animations, infos, smplh, texts, c)
        if batch_idx == 0:
            norm_animations = get_motion_guofeats(animations, infos)
            render(norm_animations, infos, smplh, joints_renderer, smpl_renderer, texts, tmp_path, ty_log, out_formats, video_log=True)

        has_nan = (
                any(torch.isnan(t.cpu()).any() for t in metrics_reward.values())
                or torch.isnan(animations.cpu()).any()
        )
        if has_nan:
            print("Found NaN in masked_tmr")
            save_dir = "NaN_folder"
            os.makedirs(save_dir, exist_ok=True)
            # Do something if there is at least one NaN
            try:
                masked_tmr_np = metrics_reward["tmr"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_with_nan.npy"), masked_tmr_np)

            except:
                masked_tmr_plus_np = metrics_reward["tmr++"].cpu().numpy()
                np.save(os.path.join(save_dir, "masked_tmr_plus_with_nan.npy"), masked_tmr_plus_np)
            # Save to .npy file
            np.save(os.path.join(save_dir, "sequences.npy"), animations.cpu().numpy())
            import json
            with open(os.path.join(save_dir, "infos.json"), "w") as f:
                infos["all_lengths"] = infos["all_lengths"].tolist()
                json.dump(infos, f, indent=4)

        total_reward += metrics_reward["reward"].sum().item()
        batch_count_reward += metrics_reward["reward"].shape[0]

        total_tmr += metrics["tmr"].sum().item()
        batch_count_tmr += metrics["tmr"].shape[0]

        total_tmr_plus_plus += metrics["tmr++"].sum().item()
        batch_count_tmr_plus_plus += metrics["tmr++"].shape[0]

        total_tmr_guo += metrics["guo"].sum().item()
        batch_count_guo += metrics["guo"].shape[0]

    avg_reward = total_reward / batch_count_reward
    avg_tmr = total_tmr / batch_count_tmr
    avg_tmr_plus_plus = total_tmr_plus_plus / batch_count_tmr_plus_plus
    avg_guo = total_tmr_guo / batch_count_guo

    print(avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo)

    return avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo


@hydra.main(config_path="config", config_name="TrainRL", version_base="1.3")
def main(c: DictConfig):
    config_dict = OmegaConf.to_container(c, resolve=True)
    wandb.login(key="686f740320175b422861147930c51baba0e47fe6")

    wandb.init(
        project="TM-BM",
        name="ROBA A CASO",
        config=config_dict,
        group="StableMoFusionRL"
    )

    create_folder_results("ResultRL")
    create_folder_results("RL_Model")

    accelerator = Accelerator()

    device = accelerator.device

    # ckpt_path = os.path.join(c.run_dir, c.ckpt_name)
    # print("Loading the checkpoint")
    # ckpt = torch.load(str(ckpt_path), map_location=device)
    #
    # joints_renderer = instantiate(c.joints_renderer)
    # smpl_renderer = instantiate(c.smpl_renderer)

    parser = GenerateOptions()
    # parser = TestOptions()
    opt = parser.parse()

    set_seed(opt.seed)
    opt.device = device

    print("Loading the models")
    model = build_models(opt)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')
    niter = load_model_weights(model, ckpt_path, use_ema=False, device=device)

    # Create a pipeline for generation in diffusion model framework
    diffusion_rl = DiffusePipeline(
        opt=opt,
        model=model,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float32,
    )

    diffusion_rl.model.cond_mask_prob = 0.0  # todo remvoe
    #
    # for p in diffusion_rl.model.parameters():
    #     p.requires_grad = False

    freeze_normalization_layers(diffusion_rl)

    # lora_config = LoraConfig(
    #     # task_type=TaskType.UNET,  # we’re adapting a diffusion UNet
    #     inference_mode=False,  # fine-tuning, not just inference
    #     r=8,  # LoRA rank
    #     lora_alpha=16,  # LoRA scaling
    #     lora_dropout=0.0,  # dropout on LoRA layers
    #     bias="none",  # no bias adapters
    #     # target all the cross-attention Q/K/V projections
    #     target_modules=["query", "key", "value", "embed_text"
    #                     "textTransEncoder.layers.0.self_attn.out_proj",
    #                     "textTransEncoder.layers.1.self_attn.out_proj",
    #                     "textTransEncoder.layers.2.self_attn.out_proj",
    #                     "textTransEncoder.layers.3.self_attn.out_proj",
    #                     ],
    # )
    #
    # diffusion_rl.model.unet = get_peft_model(diffusion_rl.model.unet, lora_config)

    total_trainable_params = sum(p.numel() for p in diffusion_rl.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_trainable_params:,}")

    total_trainable_params = sum(p.numel() for p in diffusion_rl.model.parameters())
    print(f"Total parameters: {total_trainable_params:,}")

    trainable_params = [name for name, param in diffusion_rl.model.named_parameters() if param.requires_grad]
    print("Trainable layer:", trainable_params)

    joints_renderer = instantiate(c.joints_renderer)

    # text_model = TextToEmb(
    #     modelpath=cfg.data.text_encoder.modelname, mean_pooling=cfg.data.text_encoder.mean_pooling, device=device
    # )
    text_model = diffusion_rl.model.encode_text

    smplh = None  # we do need conversion for thsi model, it works in guofeats

    train_dataloader = get_dataset_loader(opt, opt.batch_size, mode='hml_gt', split='easy_val')
    val_dataloader = get_dataset_loader(opt, opt.batch_size, mode='hml_gt', split='easy_test', shuffle=False)

    infos = {
        "featsname": None,  # cfg.motion_features,
        "fps": c.fps,
        "guidance_weight": c.guidance_weight
    }

    if c.sequence_fixed:
        infos["all_lengths"] = torch.tensor(np.full(2048, int(c.time * c.fps))).to(device)

    file_path = "../ResultRL/VAL/"
    os.makedirs(file_path, exist_ok=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_rl.model.parameters()), lr=c.lr,
                                  betas=(c.beta1, c.beta2), eps=c.eps,
                                  weight_decay=c.weight_decay)

    avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, val_dataloader, device, infos, text_model,
                                                           smplh, joints_renderer, None, c, None,
                                                           path="ResultRL/VAL/OLD/")
    wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo,
                              "iterations": 0}})

    iter_bar = tqdm(range(c.iterations), desc="Iterations", total=c.iterations)

    # torch.save(diffusion_rl.model.state_dict(), 'RL_Model/checkpoint_' + str(0 + 1) + '.pth')
    for iteration in iter_bar:

        train_datasets_rl = generate(diffusion_rl, train_dataloader, iteration, c, device, infos, text_model, smplh,
                                     None)  # , generation_iter
        train(diffusion_rl, optimizer, train_datasets_rl, iteration, c, infos, device, old_model=None)

        if (iteration + 1) % c.val_iter == 0:
            avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, val_dataloader, device, infos,
                                                                   text_model, smplh, joints_renderer,
                                                                   None, c, None,
                                                                   path="ResultRL/VAL/" + str(iteration + 1) + "/")
            wandb.log({"Validation": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo,
                                      "iterations": iteration + 1}})
            # torch.save(diffusion_rl.model.state_dict(), 'RL_Model/checkpoint_' + str(iteration + 1) + '.pth')
            iter_bar.set_postfix(val_tmr=f"{avg_tmr:.4f}")

    # file_path = "ResultRL/TEST/"
    # os.makedirs(file_path, exist_ok=True)
    # avg_reward, avg_tmr, avg_tmr_plus_plus, avg_guo = test(diffusion_rl, test_dataloader, device, infos, text_model,
    #                                                       smplh, joints_renderer,
    #                                                       smpl_renderer, c, test_embedding_tmr, path="ResultRL/TEST/")
    # wandb.log({"Test": {"Reward": avg_reward, "TMR": avg_tmr, "TMR++": avg_tmr_plus_plus, "Guo": avg_guo}})

    torch.save(diffusion_rl.model.state_dict(), 'RL_Model/model_final.pth')


if __name__ == "__main__":
    main()
    wandb.finish()
