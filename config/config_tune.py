import os
import random
import argparse

from utils.functions import Storage


class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'v1_semi': self.__V1_Semi,
            'mosei':self._MOSEI,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams(str.lower(args.datasetName))
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
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self,datasetName):
        if datasetName == 'mosei':
            root_dataset_dir = 'dataset'
        else:
            root_dataset_dir = 'dataset'
        tmp = {
            'sims3':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir,'SimsLargeV1.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (41, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
            'sims3l':{
                'unaligned': {
                    # 'dataPath': os.path.join('simsv3_unsup.pkl'),
                    'dataPath': os.path.join(root_dataset_dir,'simsv2_30.pkl'),
                    # (batch_size, seq_lens, feature_dim)
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
                    'dataPath': os.path.join(root_dataset_dir,'unaligned_50_semi_1.pkl'),
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
                'early_stop': 8
            },
            # dataset
             'debugParas':{
                'd_paras': ['initvalue','learning_rate_filter','weight_decay_filter','hidden_dims','post_fusion_dim','post_text_dim','post_audio_dim','post_video_dim','dropouts','post_dropouts','batch_size','M', 'T', 'A', 'V','Consisi_A','Consisi_T','Consisi_V','learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other','weight_decay_bert','weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                        'batch_size': random.choice([32,64,128]),
                        'learning_rate_bert': random.choice([2e-6,5e-5]),
                        'learning_rate_audio': random.choice([1e-4, 1e-3 ,5e-3]),
                        'learning_rate_video': random.choice([1e-4, 1e-3,5e-3]),
                        'learning_rate_other': random.choice([1e-4, 1e-3,5e-3]),
                        'learning_rate_filter':random.choice([1e-2,1e-3,5e-3,5e-2,1e-4]), 
                        'weight_decay_bert': random.choice([0,1e-4]),
                        'weight_decay_audio': random.choice([0,1e-4]),
                        'weight_decay_video': random.choice([0,1e-4]),
                        'weight_decay_other': random.choice([0,1e-4]),
                        'weight_decay_filter':random.choice([0,1e-4]),  
                        'hidden_dims': (64, 32, 64),
                        'post_fusion_dim': random.choice([16,32,64]),
                        'post_text_dim': random.choice([8,16,32,64]),
                        'post_audio_dim': random.choice([8,16,32]),
                        'post_video_dim': random.choice([8,16,32,64]),
                        'dropouts': random.choice([(0.1,0.1,0.1),(0.2,0.2,0.2),(0,0,0),(0.3,0.3,0.3)]),
                        'post_dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.1,0.1,0.1,0.1),(0,0,0,0)]),
                        'M':random.choice([0.2,0.4,0.6,0.8,1]),
                        'T':random.choice([0.2,0.4,0.6,0.8,1]),
                        'A':random.choice([0.2,0.4,0.6,0.8,1]),
                        'V':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_A':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_T':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_V':random.choice([0.2,0.4,0.6,0.8,1]),
                        'initvalue':random.choice([10,20,40,80]),
            }
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
            'debugParas':{
                 'd_paras': ['initvalue','learning_rate_filter','weight_decay_filter','hidden_dims','post_fusion_dim','post_text_dim','post_audio_dim','post_video_dim','dropouts','post_dropouts','batch_size','M', 'T', 'A', 'V','Consisi_A','Consisi_T','Consisi_V','learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other','weight_decay_bert','weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                        'batch_size': random.choice([32,64,128]),
                        'learning_rate_bert': random.choice([2e-6,5e-5]),
                        'learning_rate_audio': random.choice([1e-4, 1e-3 ,5e-3]),
                        'learning_rate_video': random.choice([1e-4, 1e-3,5e-3]),
                        'learning_rate_other': random.choice([1e-4, 1e-3,5e-3]),
                        'learning_rate_filter':random.choice([1e-2,1e-3,5e-3,5e-2,1e-4]), 
                        'weight_decay_bert': random.choice([0,1e-4]),
                        'weight_decay_audio': random.choice([0,1e-4]),
                        'weight_decay_video': random.choice([0,1e-4]),
                        'weight_decay_other': random.choice([0,1e-4]),
                        'weight_decay_filter':random.choice([0,1e-4]),  
                        'hidden_dims': (64, 32, 64),
                        'post_fusion_dim': random.choice([16,32,64]),
                        'post_text_dim': random.choice([8,16,32,64]),
                        'post_audio_dim': random.choice([8,16,32]),
                        'post_video_dim': random.choice([8,16,32,64]),
                        'dropouts': random.choice([(0.1,0.1,0.1),(0.2,0.2,0.2),(0,0,0),(0.3,0.3,0.3)]),
                        'post_dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.1,0.1,0.1,0.1),(0,0,0,0)]),
                        'M':random.choice([0.2,0.4,0.6,0.8,1]),
                        'T':random.choice([0.2,0.4,0.6,0.8,1]),
                        'A':random.choice([0.2,0.4,0.6,0.8,1]),
                        'V':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_A':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_T':random.choice([0.2,0.4,0.6,0.8,1]),
                        'Consisi_V':random.choice([0.2,0.4,0.6,0.8,1]),
                        'initvalue':random.choice([10,20,40,80]),
            }
        }
        return tmp

    def get_config(self):
        return self.args
