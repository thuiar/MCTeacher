import numpy as np

import pickle
import sys
sys.path.append('task')
from tools import get_len, noise_mimic, TestTask
from tqdm import tqdm
class EvalTask:

    @staticmethod
    def modify_commandline_options(parser):
        # n_shots
        # parser.add_argument('--n_shots', type=int, default=1284, help='Number of shots.')
        # model learning parameters.
        
        return parser

    def __init__(self, opt) -> None:

        self.task_cache = []
        # train_state ='trian'
        train_state ='train_mix'
        # Load data from pkl.
        path = opt.dataPath
        # path = 'dataset/mosi.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.tr_text, self.tr_audio, self.tr_vision = data[train_state]['text_bert'], data[train_state]['audio'], data[train_state]['vision']
        self.train_id,self.raw_text = data[train_state]['id'], data[train_state]['raw_text']
        self.train_mask =  [0 for i in data[train_state]['id']]

        self.tr_t_len, self.tr_a_len, self.tr_v_len = get_len(self.tr_text[:, 1, :]), \
             data[train_state]['audio_lengths'], data[train_state]['vision_lengths']

        self.tr_label = data[train_state]['regression_labels']

        for n_t in ['block', 'rand']:
            for n_r in np.arange(0.1, opt.noisy_rate , 0.1):
                n = {'text': (n_t, n_r), 'audio': (n_t, n_r), 'vision': (n_t, n_r)}
                train_x = noise_mimic({
                    'text': self.tr_text, 'audio': self.tr_audio, 'vision': self.tr_vision,
                    'text_lengths': self.tr_t_len, 'audio_lengths': self.tr_a_len, 'vision_lengths': self.tr_v_len
                }, noise=n)
                self.task_cache.append(TestTask(
                    train_x=train_x, train_y=self.tr_label
                ))
