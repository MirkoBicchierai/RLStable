import numpy as np
import torch
from colorama import Fore, Style, init
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.model.tmr import get_sim_matrix
from TMR.mtt.load_tmr_model import load_tmr_model_easy, easy_forward, load_tmr_model_complete
from TM2T.load_tm2t_model import load_tm2t_model_easy
from src.tools.guofeats.motion_representation import guofeats_to_joints
from MotionCritic.MotionCritic.lib.model.load_critic import load_critic

from MotionCritic.MotionCritic.parsedata import into_critic
# todo tenere il modello di reward su gpu (se serve solo mentre lo calcolo)
tmr_forward_plus_plus_complete = load_tmr_model_complete(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")
# tmr_forward_plus_plus = load_tmr_model_easy(device="cpu", dataset="tmr_humanml3d_kitml_guoh3dfeats")
# tmr_forward = load_tmr_model_easy(device="cpu", dataset="humanml3d")
tmr_forward_complete = load_tmr_model_complete(device="cpu", dataset="humanml3d")
guo_forward = load_tm2t_model_easy(device="cpu", dataset="humanml3d") # humanml3d OR humanml3d_kitml_augmented_and_hn OR tmr_humanml3d_kitml_guoh3dfeats

mean_norm = np.load("./checkpoints/t2m/t2m_condunet1d_batch64/meta/" + 'mean.npy')
std_norm = np.load("./checkpoints/t2m/t2m_condunet1d_batch64/meta/" + 'std.npy')
critic_model = load_critic("./MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth", "cpu")


def calc_eval_stats(x_guofeats, forward):
    x_latents = forward(x_guofeats)# tensor(N, 256)
    return x_latents


def is_list_of_strings(var):
    return isinstance(var, list) and all(isinstance(item, str) for item in var)


def print_matrix_nicely(matrix: np.ndarray, mmax=True):
    init(autoreset=True)
    for row in matrix:
        if mmax:
            max_val = np.max(row)
        else:
            max_val = np.min(row)
        line = ""
        for val in row:
            truncated = int(val * 1000) / 1000
            formatted = f"{truncated:.3f}"
            if val == max_val:
                line += f"{Fore.GREEN}{formatted}{Style.RESET_ALL}  "
            else:
                line += f"{formatted}  "
        print(line)

#
# def tmr_metrics(motions_guofeats,real_texts, c):
#     texts = tmr_forward(real_texts)
#     x_latents = calc_eval_stats(motions_guofeats, tmr_forward)
#     sim_matrix = get_sim_matrix(x_latents, texts.detach().cpu().type(x_latents.dtype)).numpy()
#     sim_matrix = torch.tensor(sim_matrix)
#     sim_matrix = (sim_matrix + 1) / 2
#     tmr = sim_matrix.diagonal()
#
#     reward = tmr * c.reward_scale
#
#     return tmr, reward
#
# def tmr_plus_plus_metrics(motions_guofeats,real_texts, c):
#     texts_plus_plus = tmr_forward_plus_plus(real_texts)
#     x_latents_plus_plus = calc_eval_stats(motions_guofeats, tmr_forward_plus_plus)
#
#     sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus,texts_plus_plus.detach().cpu().type(texts_plus_plus.dtype)).numpy()
#
#     sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
#     sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
#     tmr_plus_plus = sim_matrix_plus_plus.diagonal()
#
#     reward = tmr_plus_plus * c.reward_scale
#
#     return tmr_plus_plus, reward

def metric_fast(model, motions_guofeats,real_texts, c):
    texts_plus_plus = easy_forward(*model, motions_or_texts=real_texts, device="cuda:0")
    x_latents_plus_plus = easy_forward(*model, motions_or_texts=motions_guofeats, device="cuda:0")

    sim_matrix_plus_plus = get_sim_matrix(x_latents_plus_plus,texts_plus_plus.detach().cpu().type(texts_plus_plus.dtype)).numpy()

    sim_matrix_plus_plus = torch.tensor(sim_matrix_plus_plus)
    sim_matrix_plus_plus = (sim_matrix_plus_plus + 1) / 2
    tmr_plus_plus = sim_matrix_plus_plus.diagonal()

    reward = tmr_plus_plus * c.reward_scale

    return tmr_plus_plus, reward

def guo_metrics(motions_guofeats, real_texts, c):
    motions_latents, texts_latents = guo_forward(motions=motions_guofeats, texts=real_texts)
    sim_matrix = euclidean_distance_matrix(motions_latents.detach().cpu().numpy(), texts_latents.detach().cpu().numpy())

    sim_matrix = torch.tensor(sim_matrix)
    guo_et_al = sim_matrix.diagonal()
    #reward = (1 / (guo_et_al + 1)) * c.reward_scale
    reward = -guo_et_al
    return guo_et_al, reward

def get_motion_guofeats(sequences, infos, smplh=None):
    motions = []
    for idx in range(sequences.shape[0]):
        x_start = sequences[idx]
        length = infos["all_lengths"][idx].item()
        x_start = x_start[:length]
        norm =  x_start.detach().cpu().numpy() * std_norm + mean_norm
        motions.append(norm)

    # motions_guofeats = smpl_to_guofeats(motions, smplh=smplh)

    return motions


def reward_model(sequences, infos, smplh, real_texts, c):
    metrics = {}

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    # reward = stillness_reward(motions_guofeats, infos, None)
    # metrics = {
    #         "tmr": reward,
    #         "reward": reward
    #     }
    # return metrics

    if c.reward == "TMR":
        tmr, reward = metric_fast(tmr_forward_complete,motions_guofeats,real_texts, c)
        metrics = {
            "tmr": tmr,
            "reward": reward
        }


    if c.reward == "TMR++":
        tmr_plus_plus, reward = metric_fast(tmr_forward_plus_plus_complete,motions_guofeats,real_texts, c)
        metrics = {
            "tmr++": tmr_plus_plus,
            "reward" : reward
        }

    if c.reward == "GUO":
        guo_et_al, reward = guo_metrics(motions_guofeats,real_texts, c)
        metrics = {
            "guo": guo_et_al,
            "reward": reward
        }


    if c.reward == "motionCritic":

        pass
        # joints = guofeats_to_rot6d(motions_guofeats[0])
        # joints = into_critic(joints)
        # reward = critic_model.module.batch_critic(joints)
        # metrics = {
        #     "reward": reward
        # }

    return metrics

def all_metrics(sequences, infos, smplh, real_texts, c):

    motions_guofeats = get_motion_guofeats(sequences, infos, smplh)

    guo_et_al, _  = guo_metrics(motions_guofeats, real_texts, c)
    tmr_plus_plus, _ = metric_fast(tmr_forward_plus_plus_complete,motions_guofeats,real_texts, c)
    tmr, _ = metric_fast(tmr_forward_complete,motions_guofeats,real_texts, c)

    metrics = {
        "tmr": tmr,
        "tmr++": tmr_plus_plus,
        "guo": guo_et_al,
    }

    return metrics

"""
    TMR SPECIAL

    sim_matrix_tmp = get_sim_matrix(x_latents, all_embedding_tmr.detach().cpu().type(x_latents.dtype)).numpy()
    # print_matrix_nicely(sim_matrix_tmp)

    sim_matrix_tmp = (sim_matrix_tmp + 1) / 2
    diagonal_values = sim_matrix.diagonal()

    # Calculate similarity between texts and all_embedding_tmr and find the most similar embedding in all_embedding_tmr
    text_to_all_sim = torch.matmul(texts.detach().cpu(), all_embedding_tmr.transpose(0, 1))

    matching_indices = torch.argmax(text_to_all_sim, dim=1)

    special = []
    for i in range(sim_matrix_tmp.shape[0]):
        # Get the index to exclude for this row
        exclude_idx = matching_indices[i].item()
        # Make a copy of the row and set the element to exclude to NaN
        row_copy = sim_matrix_tmp[i].copy()
        row_copy[exclude_idx] = np.nan
        row_copy[row_copy > c.masking_ratio] = np.nan

        # Calculate mean without the excluded element
        row_mean = np.nanmean(row_copy)
        # Calculate special value for this row (real - mean of row of all emb)
        special_value = diagonal_values[i] - row_mean
        special.append(special_value)

    special = torch.tensor(special)

return special * c.reward_scale
"""

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def collate_tensor_with_padding(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



def stillness_reward(sequences, infos, smplh):
    joint_positions = []
    for idx in range(len(sequences)):
        x_start = sequences[idx]
        # the mask here should already be done here, right?
        joints = guofeats_to_joints(torch.tensor(x_start))
        joint_positions.append(joints)

    joints = collate_tensor_with_padding(joint_positions)
    dt = 1.0 / 200

    velocities = torch.diff(joints, dim=1) / dt
    velocity_loss = torch.mean(velocities.pow(2), dim=(1, 2, 3))

    reward = velocity_loss
    return - reward
