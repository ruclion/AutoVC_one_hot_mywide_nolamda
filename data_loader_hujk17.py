import os
import torch
import numpy as np
import pickle as pkl
from torch.utils import data

 

speaker_id_dict_path = '/ceph/home/hujk17/AutoVC_hujk17/full_106_spmel_nosli/speaker_seen_unseen.txt'


def text2list(file):
    f = open(file, 'r').readlines()
    file_list = [i.strip() for i in f]
    return file_list


def text2dict(file):
    speaker_id_dict = {}
    f = open(file, 'r').readlines()
    for i, name in enumerate(f):
        name = name.strip().split('|')[0]
        speaker_id_dict[name] = i
    # print(speaker_id_dict)
    return speaker_id_dict


def get_mel_data(fpath):
    # print('mel-path:', fpath)
    mel = np.load(fpath)
    return mel



class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, meta_path, max_len):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.max_len = max_len
        self.file_list = text2list(file=meta_path)
        self.speaker_id_dict = text2dict(speaker_id_dict_path)
        
        
    def __getitem__(self, index):
        now = self.file_list[index].split('|')
        speaker_id = self.speaker_id_dict[now[1]]
        mel = get_mel_data(os.path.join(self.root_dir, now[0]))


        if mel.shape[0] < self.max_len:
            len_pad = self.max_len - mel.shape[0]
            mel_fix = np.pad(mel, ((0,len_pad),(0,0)), 'constant')
        elif mel.shape[0] > self.max_len:
            left = np.random.randint(mel.shape[0]-self.max_len + 1)
            assert left + self.max_len <= mel.shape[0]
            mel_fix = mel[left:left+self.max_len, :]
        else:
            mel_fix = mel
        
        return mel_fix, speaker_id
    

    def __len__(self):
        return len(self.file_list)
    
    
    

def get_loader(root_dir, meta_path, batch_size=16, max_len=128, shuffle=True, drop_last = False, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(root_dir, meta_path, max_len)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last)
    return data_loader






