""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import random
from IPython import embed

class DenseActivityNet(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../../data/glove')
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(DenseActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        if split == 'test':
            self.annotations = np.load(os.path.join('../../data/ActivityNet/' '{}.npy'.format(split)),  allow_pickle=True)
        else:
            with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
                annotations = json.load(f)
            anno_pairs = []
            for vid, video_anno in annotations.items():
                duration = video_anno['duration']
                flag = True
                for timestamp in video_anno['timestamps']:
                    if timestamp[0] >= timestamp[1]:
                        flag = False
                if not flag:
                    continue
                anno_pairs.append(
                    {
                        'video': vid,
                        'duration': duration,
                        'sentences': video_anno['sentences'],
                        'timestamps': video_anno['timestamps']
                    }
                )
            self.annotations = anno_pairs

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        duration = self.annotations[index]['duration']
        tot_sentence = len(self.annotations[index]['sentences'])

        P = min(9, tot_sentence+1)
        num_sentence = np.random.randint(1, P)
        if num_sentence > tot_sentence:
            num_sentence = tot_sentence
        # id_sentence = np.random.choice(tot_sentence, num_sentence)
        idx_sample = random.sample(range(tot_sentence), num_sentence)
        idx_sample.sort()
        if self.split == 'train':
            sentence_sample = [self.annotations[index]['sentences'][idx] for idx in idx_sample]
            timestamps_sample = [self.annotations[index]['timestamps'][idx] for idx in idx_sample]
        else:
            sentence_sample = self.annotations[index]['sentences']
            timestamps_sample = self.annotations[index]['timestamps']
        word_vectors_list = []
        txt_mask_list = []

        # for sentence in self.annotations[index]['sentences']:
        for sentence in sentence_sample:
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()],
                                     dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs) # word_vectors (seq, 300)
            word_vectors_list.append(word_vectors)
            txt_mask_list.append(torch.ones(word_vectors.shape[0], 1))
        word_vectors_list = pad_sequence(word_vectors_list, batch_first=True) # word_vectors_list (k, seq, 300)
        txt_mask_list = pad_sequence(txt_mask_list, batch_first=True) # txt_mask_list (k, seq, 1)

        visual_input, visual_mask = self.get_video_features(video_id)

        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE

            overlaps_list = []
            # for gt_s_time, gt_e_time in self.annotations[index]['timestamps']:
            for gt_s_time, gt_e_time in timestamps_sample:
                s_times = torch.arange(0, num_clips).float() * duration / num_clips
                e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
                overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                            e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                               torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
                #overlaps (64, 64)
                overlaps_list.append(torch.from_numpy(overlaps))
            overlaps_list = pad_sequence(overlaps_list, batch_first=True) #overlaps_list (k, 64, 64)
        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
        if self.split == 'train':
            item = {
                'visual_input': visual_input,
                'vis_mask': visual_mask,
                'anno_idx': index,
                'word_vectors': word_vectors_list, # new for dense
                'txt_mask': txt_mask_list, # new for dense
                # 'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
                'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
                'duration': duration,
                'map_gt': overlaps_list, # new for dense
            }
        else:
            item = {
                'visual_input': visual_input,
                'vis_mask': visual_mask,
                'anno_idx': index,
                'word_vectors': word_vectors_list, # new for dense
                'txt_mask': txt_mask_list, # new for dense
                'sentence_mask': torch.ones(len(self.annotations[index]['sentences']), 1), # sentence_mask (k,1) # new for dense
                # 'sentence_mask': torch.ones(len(idx_sample), 1), # sentence_mask (k,1) # new for dense
                'duration': duration,
                'map_gt': overlaps_list, # new for dense
            }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask