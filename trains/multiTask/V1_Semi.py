import json
from operator import length_hint
import os
from shutil import which
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import copy
from loss.angular import AngularPenaltySMLoss
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from torch.autograd import Variable
import math
class MyDataset(Dataset):
    def __init__(self, memory_bank_T, memory_bank_S):
        self.memory_bank_T = memory_bank_T
        self.memory_bank_S = memory_bank_S

    def __len__(self):
        return len(self.memory_bank_T)

    def __getitem__(self, index):
        memory_t = self.memory_bank_T[index]
        memory_s = self.memory_bank_S[index]
        return memory_t, memory_s


logger = logging.getLogger('MSA')

class V1_Semi():
    def __init__(self, args):
        assert args.datasetName == 'sims3l' or args.datasetName == 'mosei'
        self.args = args
        self.args.tasks = "M"
        self.eval = self.args.tasks
        self.eval=['M','P','Filter_P']
        # ,'P','Filter_P'
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.cls = nn.CrossEntropyLoss()
        self.recloss = nn.MSELoss(reduce=False)
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)
        self.cosface = AngularPenaltySMLoss(args.post_fusion_dim,4,loss_type='cosface')
        self.cosface.Specific_fc.cuda(self.args.gpu_ids[0])
        self.global_step = 0
        self.train_step = 0
        self.pre_var = 0
        self.weight = torch.tensor(1)
        self.memory_bank_T  = None
        self.memory_bank_S  = None
        self.post_fusion_dropout = nn.Dropout(p=0.2)
        self.max_len = 15000
    def update_tensor_queue(self,tensor_queue, new_tensor, max_length=15000):
        tensor_queue = torch.cat([tensor_queue, new_tensor], dim=0)
        if tensor_queue.size(0) > max_length:
            tensor_queue = tensor_queue[-max_length:]
        return tensor_queue
    def update_bank(self,feature_T,feature_S):
        if self.memory_bank_T is None:
            self.memory_bank_T = feature_T
            self.memory_bank_S = feature_S
        else:
            self.memory_bank_T = self.update_tensor_queue(self.memory_bank_T,feature_T,max_length = self.max_len)
            self.memory_bank_S = self.update_tensor_queue(self.memory_bank_S,feature_S,max_length = self.max_len)
    def get_bank(self):
        dataset = MyDataset(self.memory_bank_T, self.memory_bank_S)
        batch_size = 128
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def update_ema_variables(self,model, ema_model, alpha, global_step):
        alpha = min(1-1/np.power((global_step + 1),float(self.weight.cpu())), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    def softmax_mse_loss(self,input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        num_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

    def softmax_kl_loss(self,input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
        if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        return F.kl_div(input_log_softmax, target_softmax, size_average=False)

    def get_opt_para(self,model):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())
        thernet_params = [p for n,p in list(model.Model.named_parameters()) if  'thernet' in n]
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and 'audio_model' not in n and 'video_model' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
            {'params': thernet_params,'weight_decay': self.args.weight_decay_filter,'lr': self.args.learning_rate_filter},
        ]
        return optimizer_grouped_parameters
    def generate_mask(self, tensor1, tensor2, sd):
        assert tensor1.shape == tensor2.shape, "error"
        if self.args.datasetName == 'sims3l':
            diff = torch.abs(tensor1 -tensor2)
        else:
            diff = torch.abs(torch.round(tensor1) - torch.round(tensor2))
        # 生成 mask
        mask = torch.where((diff <= sd) & (tensor1 * tensor2 > 0), torch.tensor(1).cuda(self.args.gpu_ids[0]), torch.tensor(0).cuda(self.args.gpu_ids[0]))
        return mask
    def get_sd(self):
        if self.args.datasetName == 'sims3l':
            sd = min((eval('self.args.initvalue')/(self.train_step+1))*0.8 + 0.2,1)
        else:
            sd = min((eval('self.args.initvalue')/(self.train_step+1))*2 + 1,3)
        return sd
    def do_train(self, model, dataloader):
        with torch.no_grad():
            flag = 'train'
            crossmodal = {
                'pre_t2a':[],
                'pre_a2v':[],
                'pre_t2v':[],
            }
            model_t = copy.deepcopy(model)
            masklist = []
            with tqdm(dataloader['train_mix']) as td:
                for batch_data in td:
                    masklist.append(batch_data['mask'])
        value = torch.cat(masklist)
        self.max_len = int(value.shape[0] - sum(value))
        optimizer_grouped_parameters = self.get_opt_para(model)
        optimizer = optim.Adam(optimizer_grouped_parameters[:5])
        optimizer_thernet = optim.Adam([optimizer_grouped_parameters[5]])
        # initilize results
        epochs, best_epoch = 0, 0 
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            self.global_step = 0
            # train
            y_pred = {'M': [], 'P': [], 'Filter_P': []}
            y_true = {'M': [], 'P': [], 'Filter_P': []}
            crossmodal = {'pre_t2a':[],'pre_a2v':[],'pre_t2v':[]}
            model.train()
            indices_all = []
            model_t.train()
            train_loss = 0.0
            filter_loss_ = 0.0
            with tqdm(dataloader['train_mix']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    mask = batch_data['mask']
                    labels = batch_data['labels']
                    # clear gradient
                    optimizer.zero_grad()
                    optimizer_thernet.zero_grad()
                    flag = 'train'
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    outputs_t = model_t((text, flag), (audio, audio_lengths), (vision,vision_lengths))
                    # compute loss
                    loss = 0
                    # 1. Supvised Loss
                    labels_true = {}
                    outputs_true = {}
                    for k in self.args.tasks:
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)
                        mask_index = torch.where(mask==1)
                        labels_true[k] = labels[k][mask_index]
                        outputs_true[k] = outputs[k][mask_index]
                        outputs_true['pre_a2v'] = outputs['pre_a2v'][mask_index]
                        outputs_true['pre_t2a'] = outputs['pre_t2a'][mask_index]
                        outputs_true['pre_t2v'] = outputs['pre_t2v'][mask_index]
                    if mask.sum()>0:
                        loss += eval('self.args.M') * self.criterion(outputs_true['M'], labels_true['M'])+eval('self.args.T')*self.criterion(outputs_true['pre_t2v'], labels_true['M'])+eval('self.args.A')*self.criterion(outputs_true['pre_t2a'], labels_true['M'])+eval('self.args.V')*self.criterion(outputs_true['pre_a2v'], labels_true['M'])
                        if not math.isnan(loss):
                            train_loss += loss.item()
                    filter_feature = torch.cat([outputs_t['fusion_t_with_v'][mask_index],outputs_t['fusion_t_with_a'][mask_index],outputs_t['fusion_v_with_a'][mask_index],outputs_t['feature_m'][mask_index]],dim=1).detach()
                    filter_feature = F.relu(model.Model.thernet_1(filter_feature), inplace=True)
                    predictions = F.softmax(model.Model.thernet_2(filter_feature),dim=1)
                    # sd = 3
                    sd = self.get_sd()
                    label  = self.generate_mask(outputs_t[k][mask_index].view(-1),labels_true['M'].view(-1),sd)
                    if mask.sum()>0:
                        filter_loss = self.cls(predictions, label)  
                        filter_loss_ += filter_loss
                        filter_loss.backward()
                        optimizer_thernet.step()
                    # 2. unSupvised Loss
                    for k in self.args.tasks:
                        mask_index = torch.where(mask==0)
                        if mask_index[0].shape[0] == 0:
                            continue
                        with torch.no_grad():
                            filter_feature = torch.cat([outputs_t['fusion_t_with_v'][mask_index],outputs_t['fusion_t_with_a'][mask_index],outputs_t['fusion_v_with_a'][mask_index],outputs_t['feature_m'][mask_index]],dim=1).detach()
                            filter_feature_ = F.relu(model.Model.thernet_1(filter_feature), inplace=True)
                            predictions = F.softmax(model.Model.thernet_2(filter_feature_),dim=1)
                            max_indices = torch.argmax(predictions, dim=1)
                            indices_all.append(max_indices)
                            un_mask = (max_indices == 1).reshape(max_indices.shape[0])
                            filter_feature_S = torch.cat([outputs['fusion_t_with_v'][mask_index],outputs['fusion_t_with_a'][mask_index],outputs['fusion_v_with_a'][mask_index],outputs['feature_m'][mask_index]],dim=1).detach()
                            self.update_bank(filter_feature,filter_feature_S)
                        # # # # # 
                        labels_true['P'] = labels['M'][mask_index]
                        outputs_true['P'] = outputs_t['M'][mask_index]
                        labels_true['Filter_P'] = labels['M'][mask_index][un_mask]
                        outputs_true['Filter_P'] = outputs_t['M'][mask_index][un_mask]
                        # # # # #
                        output_label= outputs['M'][mask_index][un_mask]
                        output_t_label= outputs_t['M'][mask_index][un_mask]
                        fusion_t_with_v = outputs['fusion_t_with_v'][mask_index][un_mask]
                        fusion_t_with_a = outputs['fusion_t_with_a'][mask_index][un_mask]
                        fusion_v_with_a = outputs['fusion_v_with_a'][mask_index][un_mask]
                        fusion_t_with_v_t = outputs_t['fusion_t_with_v'][mask_index][un_mask]
                        fusion_t_with_a_t = outputs_t['fusion_t_with_a'][mask_index][un_mask]
                        fusion_v_with_a_t = outputs_t['fusion_v_with_a'][mask_index][un_mask] #
                        if self.args.datasetName == 'sims3l':
                            loss_semi = eval('self.args.M')*self.criterion(output_label, output_t_label) + eval('self.args.Consisi_T')*self.softmax_kl_loss(fusion_t_with_v,fusion_t_with_v_t)+eval('self.args.Consisi_A')*self.softmax_kl_loss(fusion_t_with_a,fusion_t_with_a_t)+eval('self.args.Consisi_V')*self.softmax_kl_loss(fusion_v_with_a,fusion_v_with_a_t)+1e-6
                        else:
                            loss_semi = eval('self.args.M')*self.criterion(output_label, torch.round(output_t_label))+eval('self.args.Consisi_T')*self.softmax_kl_loss(fusion_t_with_v,fusion_t_with_v_t)+eval('self.args.Consisi_A')*self.softmax_kl_loss(fusion_t_with_a,fusion_t_with_a_t)+eval('self.args.Consisi_V')*self.softmax_kl_loss(fusion_v_with_a,fusion_v_with_a_t)+1e-6
                    loss += loss_semi
                    if not isinstance(loss,int):
                        loss.backward()
                    else:
                        continue
                    if not math.isnan(loss_semi):
                        train_loss += loss_semi.item()
                    # update
                    optimizer.step()
                    self.global_step = self.global_step+1
                    self.train_step = self.train_step+1
                    self.update_ema_variables(model, model_t, 0.97, self.global_step)
                    for m in self.eval:
                        y_pred[m].append(outputs_true[m].cpu())
                        y_true[m].append(labels_true[m].cpu())
                    for key in crossmodal:
                        crossmodal[key].append(outputs_t[key])
            list_ = []
            for key in crossmodal:
                list_.append(torch.cat(crossmodal[key],dim=0))
            var_ = torch.var(torch.cat(list_,dim=1),dim=1)
            var_ = torch.mean(var_)
            if epochs > 1:
                self.weight = torch.div(self.pre_var,var_)
            self.pre_var = var_

            with tqdm(dataloader['train']) as td:
                for index,batch_data in enumerate(td):
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    flag = 'train'
                    # forward
                    outputs_true = {}
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += self.criterion(outputs[m], labels[m])
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels[m].cpu())
            loss_u = 0.0
            if self.memory_bank_T is not None:
                    optimizer_thernet.zero_grad()
                    memory_dataloader = self.get_bank() 
                    with tqdm(memory_dataloader) as td:
                        for feature_t, feature_s in td:
                            feature_T = F.relu(model.Model.thernet_1(feature_t), inplace=True)
                            feature_T = model.Model.thernet_2(feature_T)
                            feature_s = self.post_fusion_dropout(feature_s)
                            feature_S = F.relu(model.Model.thernet_1(feature_s), inplace=True)
                            feature_S = model.Model.thernet_2(feature_S)
                            Lu = self.softmax_kl_loss(feature_T,feature_S)
                            Lu.backward()
                            optimizer_thernet.step()
                            loss_u += Lu
            train_loss = train_loss / len(dataloader['train_mix'])
            filter_loss_ = filter_loss_ / len(dataloader['train_mix'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))
            logger.info("filter_loss >> %.4f " % ( filter_loss_ ) )
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            logger.info("weight >> %.4f " % ( self.weight ) )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path )
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return
 

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': []}
        y_true = {'M': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    flag = 'train'
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())

        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' %(m) + dict_to_str(results))
            eval_results[m] = results

        eval_results = eval_results[self.eval[0]]
        eval_results['Loss'] = eval_loss
        return eval_results