import torch
import torch.nn as nn
from core.config import config


def dense_collate_fn(batch):
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_sentence_mask = [b['sentence_mask'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]

    # word embedding for sentence
    # batch_word_vectors (b, 8, seq, 300)
    seq = 0
    for b in batch:
        seq = max(b['word_vectors'].shape[1], seq)
    batch_word_vectors = torch.zeros(len(batch), 8, seq, 300)
    for i, b in enumerate(batch):
        b_data = b['word_vectors']
        n_s = min(b_data.shape[0], 8)
        seq_s = b_data.shape[1]
        batch_word_vectors[i, :n_s, :seq_s] = b_data[:n_s]

    # txt_mask
    # padded_batch_txt_mask (b, 8, seq, 1)
    padded_batch_txt_mask = torch.zeros(len(batch), 8, seq, 1)
    for i, txt_mask in enumerate(batch_txt_mask):
        n_s = min(txt_mask.shape[0], 8)
        seq_s = txt_mask.shape[1]
        padded_batch_txt_mask[i, :n_s, :seq_s] = txt_mask[:n_s]

    # ground truth iou
    # batch_map_gt, padded_batch_map_gt (b, 8, 1, 64, 64)
    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch), 8, 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        n_s = min(map_gt.shape[0], 8)
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][:n_s, 0, :num_clips, :num_clips] = map_gt[:n_s]

    # sentence_mask
    padded_batch_sentence_mask = torch.zeros(len(batch), 8, 1)
    for i, sentence_mask in enumerate(batch_sentence_mask):
        n_s = min(sentence_mask.shape[0], 8)
        padded_batch_sentence_mask[i][:n_s] = sentence_mask[:n_s]
    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_duration': batch_duration,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_word_vectors': batch_word_vectors,
        'batch_map_gt': padded_batch_map_gt,
        'batch_sentence_mask': padded_batch_sentence_mask,
        'batch_txt_mask': padded_batch_txt_mask
    }
    return batch_data

def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][0,:num_clips,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
    }

    return batch_data

def average_to_fixed_length(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

# from datasets.activitynet import ActivityNet
from datasets.dense_activitynet import DenseActivityNet
# from datasets.dense_activitynet_for_test import TestDenseActivityNet
# from datasets.charades import Charades
# from datasets.tacos import TACoS
# from datasets.dense_tacos import DenseTACoS

