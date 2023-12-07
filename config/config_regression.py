from multiprocessing import set_forkserver_preload
import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # multi-task
            'v1_semi': self.__V1_Semi,
            'mosei':self._MOSEI,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # self.args=args
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = 'dataset'
        tmp = {
            'sims3l':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir,'simsv2.pkl'),
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 6529,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
            'mosei':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir,'mosei20%.pkl'),
                    "seq_lens": [50, 500, 375],
                    "feature_dims": [768, 74, 35],
                    "train_samples": 16326,
                    "num_classes": 3,
                    "language": "en",
                    "KeyEval": "Loss"
                },
            },
        }
        return tmp
    def _MOSEI(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_sampling': False,
                'need_sampling_fix': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                # 改成true
                'early_stop':8
            },
            # dataset
            'datasetParas':{
                'mosei':{
                    'batch_size':128,
                    'learning_rate_bert':2e-06,
                    'learning_rate_audio':0.001,
                    'learning_rate_video':0.0001,
                    'learning_rate_other':0.0001,
                    'weight_decay_bert':0.0001,
                    'learning_rate_filter':0.01,
                    'weight_decay_audio':0,
                    'weight_decay_video':0.0001,
                    'weight_decay_other':0,
                    'weight_decay_filter':0.0001,
                    'hidden_dims':(64, 32, 64),
                    'post_fusion_dim':32,
                    'post_text_dim':16,
                    'post_audio_dim':16,
                    'post_video_dim':32,
                    'dropouts':(0.1, 0.1, 0.1),
                    'post_dropouts':(0.3, 0.3, 0.3, 0.3),
                    'M':0.4,
                    'T':0.8,
                    'A':0.6,
                    'V':0.2,
                    'Consisi_A':0.2,
                    'Consisi_T':0.6,
                    'Consisi_V':1,
                    'initvalue':20,
                }
            },
        }
        return tmp

    def __V1_Semi(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_sampling': False,
                'need_sampling_fix': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'batch_size':64,
                    'learning_rate_bert':5e-05,
                    'learning_rate_audio':0.0001,
                    'learning_rate_video':0.0001,
                    'learning_rate_other':0.0001,
                    'learning_rate_filter':0.05,
                    'weight_decay_bert':0,
                    'weight_decay_audio':0,
                    'weight_decay_video':0.0001,
                    'weight_decay_other':0.0001,
                    'weight_decay_filter':0,
                    'hidden_dims':(64, 32, 64),
                    'post_fusion_dim':64,
                    'post_text_dim':8,
                    'post_audio_dim':8,
                    'post_video_dim':32,
                    'dropouts':(0.2, 0.2, 0.2),
                    'post_dropouts':(0.1, 0.1, 0.1, 0.1),
                    'M':0.8,
                    'T':1,
                    'A':1,
                    'V':0.8,
                    'Consisi_A':0.8,
                    'Consisi_T':0.2,
                    'Consisi_V':1,
                    'initvalue':80,
                }
            },
        }
        return tmp
    def get_config(self):
        return self.args