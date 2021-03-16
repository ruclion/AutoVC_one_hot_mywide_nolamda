import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from audio_hujk2_GL_AutoVC import hparams as audio_hparams
from audio_hujk2_GL_AutoVC import write_wav, normalized_db_mel2wav



# 超参数个数：17
hparams = {
    'write_sample_rate': 22050,
    'sample_rate': 16000,
    'preemphasis': None,
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'num_mels': 80,
    'window': 'hann',
    'fmin': 90.,
    'fmax': 7600.,
    'ref_db': 16,  
    'min_db': -100.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'center': True, # 不知道为什么要是True
}

assert hparams == audio_hparams



device = 'cuda:0'
# ckpt_path_ckpt = 'logs_dir/autovc_one_hot146000.ckpt'
ckpt_path_dir = 'logs_dir'
log_ckpt_every = 3


conversion_list_path = 'conversion_list.txt'
data_dir = '../AutoVC_hujk17/full_106_spmel_nosli'
speaker_id_dict_path = '../AutoVC_hujk17/full_106_spmel_nosli/speaker_seen_unseen.txt'

dim_neck = 256
dim_emb = 256
dim_pre = 512
freq = 8
# look up table用, 102个人, 用128作为上限
speaker_num =128



def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        print(dir_list)
        return dir_list



def pad_seq(x, base=freq):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def text2dict(file):
    speaker_id_dict = {}
    f = open(file, 'r').readlines()
    for i, name in enumerate(f):
        name = name.strip().split('|')[0]
        speaker_id_dict[name] = i
    # print(speaker_id_dict)
    return speaker_id_dict

def load_ckpt_arch():
    G = Generator(dim_neck=dim_neck, dim_emb=dim_emb, dim_pre=dim_pre, freq=freq, speaker_num=speaker_num).eval().to(device)
    return G

def log_ckpt_wav(ckpt_path, G):
    # ckpt_path: autovc_one_hot146000.ckpt
    # log_dir = autovc_one_hot146000-log-wav
    log_dir = ckpt_path.split('.')[0] + '-log-wav'
    os.makedirs(log_dir, exist_ok=True)


    # init model
    # device = 'cuda:0'
    # G = Generator(dim_neck=dim_neck, dim_emb=dim_emb, dim_pre=dim_pre, freq=freq, speaker_num=speaker_num).eval().to(device)
    g_checkpoint = torch.load(os.path.join(ckpt_path_dir, ckpt_path))
    G.load_state_dict(g_checkpoint['model'])

    # init speaker name -> id
    speaker_id_dict = text2dict(speaker_id_dict_path)


    # p228/p228_077.npy|p228|p227
    f = open(conversion_list_path, 'r').readlines()
    tasks = [i.strip() for i in f]


    # spect_vc = []
    for now_i, task in enumerate(tasks): 
        task = task.split('|')
        assert len(task) == 3
        mel_path = task[0]
        s_name = task[1]
        t_name = task[2]

        # process from string -> data: mel, s, t
        mel = np.load(os.path.join(data_dir, mel_path))
        mel, len_pad = pad_seq(mel)
        s_id = speaker_id_dict[s_name]
        t_id = speaker_id_dict[t_name]

        # process from data -> batch tensor: mel, s, t
        mel = torch.from_numpy(mel[np.newaxis, :, :]).to(device)
        s_id = torch.from_numpy(np.asarray([s_id])).to(device)
        t_id = torch.from_numpy(np.asarray([t_id])).to(device)
        print('speaker model out----------', s_id.size())


        with torch.no_grad():
            _, x_identic_psnt, _ = G(mel, s_id, t_id)
            print('mel size:', x_identic_psnt.size())
        
        if len_pad == 0:
            # uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            x_identic_psnt = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            # uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            x_identic_psnt = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
        
        spec_auto_vc = x_identic_psnt
        wav_auto_vc_rec_path = os.path.join(log_dir, str(now_i) + '_GL.wav')

        wav_arr_rec_auto_vc = normalized_db_mel2wav(spec_auto_vc)
        write_wav(wav_auto_vc_rec_path, wav_arr_rec_auto_vc)

            
    # with open('results.pkl', 'wb') as handle:
    #     pickle.dump(spect_vc, handle)          

if __name__ == "__main__":
    G = load_ckpt_arch()
    ckpt_list = get_file_list(ckpt_path_dir)

    for i, ckpt_path in enumerate(ckpt_list): 
        if i % log_ckpt_every == 0:
            log_ckpt_wav(ckpt_path=ckpt_path, G=G)
        # break