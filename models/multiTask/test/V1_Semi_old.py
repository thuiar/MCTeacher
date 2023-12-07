from __future__ import print_function
from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal
from models.subNets.BertTextEncoder import BertTextEncoder
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.subNets.transformers_encoder.multihead_attention import MultiheadAttention
__all__ = ['V1_Semi']
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3

class AVsubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, bidirectional):
        super(AVsubNet, self).__init__()
        # define the pre-fusion subnetworks
        self.liner = nn.Linear(in_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn1 = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)
        self.rnn2 = nn.LSTM(2*hidden_size, hidden_size, bidirectional=bidirectional)
        self.layer_norm = nn.LayerNorm((2*hidden_size,))

    def forward(self, sequence, lengths):
        lengths = lengths.squeeze().int().detach().cpu().view(-1)
        batch_size = sequence.shape[0]
        sequence = self.dropout(self.liner(sequence))	
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = self.rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = self.layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = self.rnn2(packed_normed_h1)
        utterance = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return utterance

class Reconsitution(nn.Module):
    """效仿ARGF模型"""
    def __init__(self, args, input_dim, output_dim):
        super(Reconsitution, self).__init__()
        self.rec_dropout = nn.Dropout(args.rec_dropout)
        self.post_layer_1_rec = nn.Linear(input_dim, input_dim)
        self.post_layer_2_rec = nn.Linear(input_dim, output_dim)
        # self.tanh = nn.Tanh()

    def forward(self, input_feature):
        input_feature = self.rec_dropout(input_feature)
        input_feature1 = F.relu(self.post_layer_1_rec(input_feature))
        input_feature2 = self.post_layer_2_rec(input_feature1)
        return input_feature2

class V1_Semi(nn.Module):
    def __init__(self, args):
        super(V1_Semi, self).__init__()
        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts
        self.m_label,self.m1_label,self.m2_label = args.Final_metric
        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim

        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune)
        self.tliner = nn.Linear(self.text_in, self.text_hidden)
        self.audio_model = AVsubNet(self.audio_in, self.audio_hidden, self.audio_prob, bidirectional=True)
        self.video_model = AVsubNet(self.video_in, self.video_hidden, self.video_prob, bidirectional=True)

        
        # define the classify layer for text
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)


        # define the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(4 * self.audio_hidden, self.post_audio_dim)


        # define the classify layer for video
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(4 * self.video_hidden, self.post_video_dim)

        # transformer fuison
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.post_text_dim + self.post_audio_dim + self.post_video_dim, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)


        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        # t2a
        self.fusion_cross_t2a = nn.Linear(self.post_text_dim + self.post_audio_dim, self.post_fusion_dim)
        # self.fusion_cross_t2a_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)/
        self.fusion_pre_t2a = nn.Linear(self.post_fusion_dim, 7)

        #t2v
        self.fusion_cross_t2v = nn.Linear(self.post_text_dim + self.post_video_dim, self.post_fusion_dim)
        # self.fusion_cross_t2v_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.fusion_pre_t2v = nn.Linear(self.post_fusion_dim, 7)

        #a2v
        self.fusion_cross_a2v = nn.Linear(self.post_video_dim + self.post_audio_dim,self.post_fusion_dim)
        # self.fusion_cross_a2v_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.fusion_pre_a2v = nn.Linear(self.post_fusion_dim, 7)
        
        #mlp
        self.mlp_av = nn.Sequential(
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            # nn.Dropout(final_dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(final_h),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        )
        self.mlp_tv = nn.Sequential(
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            # nn.Dropout(final_dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(final_h),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        )
        self.mlp_ta = nn.Sequential(
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
            # nn.Dropout(final_dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(final_h),
            nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        )

        # t2a+a2v
        self.fusion_aatv_1 = nn.Linear(2*self.post_fusion_dim,self.post_fusion_dim) 
        self.fusion_aatv_2 = nn.Linear(self.post_fusion_dim,self.post_fusion_dim)  
        self.fusion_aatv_3 = nn.Linear(self.post_fusion_dim,1)

        # t2v+a2v
        self.fusion_vvta_1 = nn.Linear(2*self.post_fusion_dim,self.post_fusion_dim) 
        self.fusion_vvta_2 = nn.Linear(self.post_fusion_dim,self.post_fusion_dim) 
        self.fusion_vvta_3 = nn.Linear(self.post_fusion_dim,1)
        self.softmax = nn.Softmax(dim=1)

    def extract_features_eazy(self, audio, audio_lengths, vision, vision_lengths):
        vision_temp = []
        audio_temp = []
        for vi in range(len(vision_lengths)):
            vision_temp.append(torch.mean(vision[vi][:vision_lengths[vi]], axis=0))
        for ai in range(len(audio_lengths)):
            audio_temp.append(torch.mean(audio[ai][:audio_lengths[ai]], axis=0))
        vision_utt = torch.stack(vision_temp)
        audio_utt = torch.stack(audio_temp)
        return audio_utt, vision_utt

    def forward(self, text_x, audio_x, video_x):
        text_x , flag = text_x
        batch_size = text_x.shape[0]
        # utterance_audio_raw, utterance_video_raw = self.extract_features_eazy(audio_x, a_len, video_x, v_len)
        global text_h
        global audio_h
        global video_h
        if flag == 'train':
            # data_pre
            audio_x, a_len = audio_x
            video_x, v_len = video_x
            text_x = self.text_model(text_x)[:,0,:]
            text_h = self.tliner(text_x)
            audio_h = self.audio_model(audio_x, a_len)
            video_h = self.video_model(video_x, v_len)
            # audio_h = self.audio_model(audio_x.squeeze(1))
            # video_h = self.video_model(video_x.squeeze(1))

        if flag == 'mix_train':
            text_h = text_x
            audio_h = audio_x
            video_h = video_x

        # text
        x_t1 = self.post_text_dropout(text_h)
        # x_t1 = text_h
        x_t2 = F.relu(self.post_text_layer_1(x_t1), inplace=True)
        # audio
        x_a1 = self.post_audio_dropout(audio_h)
        x_a2 = F.relu(self.post_audio_layer_1(x_a1), inplace=True)
        # video
        x_v1 = self.post_video_dropout(video_h)
        x_v2 = F.relu(self.post_video_layer_1(x_v1), inplace=True)
        # fusion
        fusion_t_with_v = torch.cat([x_t2,x_v2], dim =1)
        fusion_t_with_v = F.relu(self.fusion_cross_t2v(fusion_t_with_v), inplace=True)
        fusion_t_with_v_semi = self.mlp_tv(fusion_t_with_v)
        # fusion_t_with_v_semi = fusion_t_with_v
        #  100% 有监督数据的时候，把softmax去掉
        pre_t2v_semi = self.softmax(self.fusion_pre_t2v(fusion_t_with_v_semi))

        fusion_t_with_a = torch.cat([x_t2,x_a2], dim =1)
        fusion_t_with_a = F.relu(self.fusion_cross_t2a(fusion_t_with_a), inplace=True)
        fusion_t_with_a_semi = self.mlp_ta(fusion_t_with_a)
        # fusion_t_with_a_semi = fusion_t_with_a
        pre_t2a_semi = self.softmax(self.fusion_pre_t2a(fusion_t_with_a_semi))
        

        fusion_v_with_a = torch.cat([x_v2,x_a2], dim =1)
        fusion_v_with_a = F.relu(self.fusion_cross_a2v(fusion_v_with_a), inplace=True)
        fusion_v_with_a_semi = self.mlp_av(fusion_v_with_a)
        pre_a2v_semi = self.softmax(self.fusion_pre_a2v(fusion_v_with_a_semi))
        # 融合1
        drop_fusion_data_1 = torch.cat([fusion_v_with_a_semi,fusion_t_with_v_semi],dim=1)
        drop_fusion_data_1 = self.post_fusion_dropout(drop_fusion_data_1)
        fusion_data_1 = self.fusion_vvta_1(drop_fusion_data_1)
        fusion_data_1= self.fusion_vvta_2(fusion_data_1)
        M_1 = self.fusion_vvta_3(fusion_data_1)
        M_1 = torch.sigmoid(M_1)
        # 融合2
        drop_fusion_data_2 = torch.cat([fusion_v_with_a_semi,fusion_t_with_a_semi],dim=1)
        drop_fusion_data_2 = self.post_fusion_dropout(drop_fusion_data_2)
        fusion_data_2 = self.fusion_aatv_1(drop_fusion_data_2)
        fusion_data_2= self.fusion_aatv_2(fusion_data_2)
        M_2 = self.fusion_aatv_3(fusion_data_2)
        M_2 = torch.sigmoid(M_2)
        #
        fusion_data = torch.cat([x_t2, x_a2, x_v2], dim =1)
        fusion_data = self.post_fusion_dropout(fusion_data)
        fusion_data = self.post_fusion_layer_1(fusion_data)
        fusion_data = self.post_fusion_layer_2(fusion_data)
        fusion_data_m = fusion_data
        fusion_data = self.post_fusion_layer_3(fusion_data)
        output_fusion = torch.sigmoid(fusion_data)
  
        final_M = (self.m_label*output_fusion+self.m1_label*M_1+self.m2_label*M_2)* self.output_range+ self.output_shift
        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'a':x_a2,
            't':x_t2,
            'v':x_v2,
            'pre_t2a':pre_t2a_semi,
            'pre_t2v':pre_t2v_semi,
            'pre_a2v':pre_a2v_semi,
            'fusion_t_with_v_semi':fusion_t_with_v_semi,
            'fusion_t_with_a_semi':fusion_t_with_a_semi,
            'fusion_v_with_a_semi':fusion_v_with_a_semi,
            'fusion_t_with_v':fusion_t_with_v,
            'fusion_t_with_a':fusion_t_with_a,
            'fusion_v_with_a':fusion_v_with_a,
            'M': final_M,
            'f':fusion_data_m,
        }

        return res