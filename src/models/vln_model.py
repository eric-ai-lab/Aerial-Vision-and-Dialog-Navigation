from math import gamma
import numpy as np
import math
# import dsntnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel, BertTokenizerFast

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None): # context will be weighted and concat with h
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax 
            attn.data.masked_fill_(mask, -float('inf'))              
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        lang_embeds = torch.cat((weighted_context, h), 1)

        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class MulAttention(nn.Module):

    def __init__(self, dim):
        '''Initialize layer.'''
        super(MulAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim)
        self.tanh = nn.Tanh()

    def forward(self, h, context): # context will be weighted and concat with h
        '''Propagate h through the network.

        h: batch x dim
        context: batch  x dim
        '''
        target = self.linear_in(h) # batch x dim
        # Get attention
        attn = torch.mul(context, target)  # batch x dim
        attn = self.sm(attn)

        weighted_context = torch.mul(attn, context)  # batch x dim
        lang_embeds = torch.cat((weighted_context, h), 1)

        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn
        
class pre_direction(nn.Module):
    def __init__(self):
        super(pre_direction, self).__init__()
        self.embedding = nn.Linear(2, 32)

        self.direction_prediction = nn.Linear(512,2)
        self.linears = nn.Sequential(nn.Linear(768+32, 512),
            # nn.BatchNorm1d(512, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        
        # self.sm = nn.Softmax(dim=1)
        
    def forward(self, h, current_direct):
        # direct_embeds = self.embedding(current_direct).squeeze(1)
        direct_embeds = torch.concat((torch.sin(current_direct/180*3.14159),torch.cos(current_direct/180*3.14159)),axis = 1)
        # print(direct_embeds.shape)
        # print(h.shape)
        h = torch.cat((self.embedding(direct_embeds), h), 1)
        h = self.linears(h)
        return self.direction_prediction(h)
    

class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        # freeze_network(self.bert)
        # for child in self.bert.children():
        #     # ct += 1
        #     # if ct < 7:
        #         for param in child.parameters():
        #             param.requires_grad = False
        
        ### New layers:
        self.linears = nn.Sequential(nn.Linear(768, 64),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 49),
            # nn.BatchNorm1d(768, eps=1e-12),
            nn.ReLU())

    def forward(self, ids,  mask ):
        bert_output= self.bert(ids, attention_mask=mask)
        cls_hidden = bert_output['pooler_output']
        sequence_output = bert_output['last_hidden_state']# sequence_output[0]: batch_size*seq_lenth*768
        # print(cls_hidden)
        # cls_hidden = sequence_output[0][:,0,:].view(-1,768)
        linear_output = self.linears(cls_hidden) ## extract the 1st token's embeddings



        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        return sequence_output, linear_output, cls_hidden



class ViT_LSTM(nn.Module):
    def __init__(self, args, vit_model, hidden_size=768, 
                      dropout_ratio = 0.5, im_channel_size=512, im_feature_size = 49, embedding_size = 32):
        super().__init__()
        print('\nInitalizing the CLIP_LSTM model ...')
        
        self.args = args
        # self.direction_action_embedding = nn.Embedding(output_action_size_0, int(embedding_size))
        self.direction_embedding = nn.Linear(2, embedding_size)
        
        self.pos_embedding = nn.Linear(2, embedding_size)
        # self.conv1_traj_img = nn.Conv2d(8,1,kernel_size = 1)
        
        self.vision_model = vit_model
        # self.vision_conv = nn.Conv2d(650,64, kernel_size=1)
        # self.hh_img_linears = nn.Sequential(nn.Linear(768, im_feature_size),
        #     # nn.BatchNorm1d(64, eps=1e-12),
        #     nn.Tanh())
        self.attention_layer_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision = SoftDotAttention(im_feature_size)
        self.vision_lstm = nn.LSTMCell(im_feature_size, 576)

        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, 192)
        
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            # nn.BatchNorm1d(768, eps=1e-12),
            )
        
        
        # self.fc = nn.Linear(im_feature_size+512*49, 64)
        self.fc = nn.Sequential(nn.Linear(im_feature_size, 128),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(768, eps=1e-12),
            nn.ReLU())


    def forward(self, current_direct, im_input, pos_input, cls_hidden, lang_feature, h_0=None, c_0=None, hh_0=None, cc_0=None):
        
        
        im_feature = self.vision_model(im_input)
        # image_embeds = self.vision_conv(im_embeds)
        # Size([8, 512, 7,7])

        im_feature=im_feature.view(im_feature.size(0), im_feature.size(1), -1)
        input_lstm_0, beta = self.attention_layer_vision(cls_hidden, im_feature)
        drop = self.drop(input_lstm_0)
        
        if hh_0 is None or cc_0 is None:
            hh_1,cc_1 = self.vision_lstm(drop)
        else:
            hh_1,cc_1 = self.vision_lstm(drop, (hh_0,cc_0))

        direction = torch.concat((torch.sin(current_direct/180*3.14159),torch.cos(current_direct/180*3.14159)),axis = 1)
        direction_embeds = self.direction_embedding(direction)   # (batch, embedding_size)

        # batch_size * 37888
        if h_0 is None or c_0 is None:
            h_1,c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1,c_1 = self.direct_lstm(direction_embeds, (h_0,c_0))

        action_module_input, alpha = self.attention_layer_lang(torch.cat((h_1,hh_1), 1),
                                                        lang_feature) 
        # lang_embeds torch.Size([6, 768])
        # h_sali = self.decoder_to_sali(im_embeds).squeeze(2)
        # # batch_size * 197

        h_sali = self.fc(input_lstm_0).view(-1,1,8,8)

        pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # lang_embeds = self.fc(lang_embeds)
        output = self.decoder_2_action_full(action_module_input)

        return h_1, c_1, hh_1, cc_1, output, pred_saliency 




class ViT_LSTM_vision_only(nn.Module):
    def __init__(self, args, vit_model, hidden_size=768, 
                      dropout_ratio = 0.5, im_channel_size=512, im_feature_size = 49, embedding_size = 32):
        super().__init__()
        print('\nInitalizing the CLIP_LSTM model ...')
        
        self.args = args
        # self.direction_action_embedding = nn.Embedding(output_action_size_0, int(embedding_size))
        self.direction_embedding = nn.Linear(2, embedding_size)
        
        self.pos_embedding = nn.Linear(2, embedding_size)
        # self.conv1_traj_img = nn.Conv2d(8,1,kernel_size = 1)
        
        self.vision_model = vit_model
        # self.vision_conv = nn.Conv2d(650,64, kernel_size=1)
        # self.hh_img_linears = nn.Sequential(nn.Linear(768, im_feature_size),
        #     # nn.BatchNorm1d(64, eps=1e-12),
        #     nn.Tanh())
        self.attention_layer_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision = SoftDotAttention(im_feature_size)
        self.vision_lstm = nn.LSTMCell(im_feature_size, 576)

        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, 192)
        
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            # nn.BatchNorm1d(768, eps=1e-12),
            )
        
        
        # self.fc = nn.Linear(im_feature_size+512*49, 64)
        self.fc = nn.Sequential(nn.Linear(im_feature_size, 128),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(768, eps=1e-12),
            nn.ReLU())
        self.h_fc = nn.Sequential(nn.Linear(hidden_size, im_feature_size),
            nn.ReLU())


    def forward(self, current_direct, im_input, pos_input, h_0=None, c_0=None, hh_0=None, cc_0=None):
        
        
        im_feature = self.vision_model(im_input)
        # image_embeds = self.vision_conv(im_embeds)
        # Size([8, 512, 7,7])

        im_feature=im_feature.view(im_feature.size(0), im_feature.size(1), -1)
        input_lstm_0, beta = self.attention_layer_vision(self.h_fc(torch.cat((h_0,hh_0),1)), im_feature)
        drop = self.drop(input_lstm_0)
        
        if hh_0 is None or cc_0 is None:
            hh_1,cc_1 = self.vision_lstm(drop)
        else:
            hh_1,cc_1 = self.vision_lstm(drop, (hh_0,cc_0))

        direction = torch.concat((torch.sin(current_direct/180*3.14159),torch.cos(current_direct/180*3.14159)),axis = 1)
        direction_embeds = self.direction_embedding(direction)   # (batch, embedding_size)

        # batch_size * 37888
        if h_0 is None or c_0 is None:
            h_1,c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1,c_1 = self.direct_lstm(direction_embeds, (h_0,c_0))

        action_module_input = torch.cat((h_1,hh_1), 1)
        # lang_embeds torch.Size([6, 768])
        # h_sali = self.decoder_to_sali(im_embeds).squeeze(2)
        # # batch_size * 197

        h_sali = self.fc(input_lstm_0).view(-1,1,8,8)

        pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # lang_embeds = self.fc(lang_embeds)
        output = self.decoder_2_action_full(action_module_input)

        return h_1, c_1, hh_1, cc_1, output, pred_saliency 





class ViT_LSTM_lang_only(nn.Module):
    def __init__(self, args, vit_model, hidden_size=768, 
                      dropout_ratio = 0.5, im_channel_size=512, im_feature_size = 49, embedding_size = 32):
        super().__init__()
        print('\nInitalizing the CLIP_LSTM model ...')
        
        self.args = args
        # self.direction_action_embedding = nn.Embedding(output_action_size_0, int(embedding_size))
        self.direction_embedding = nn.Linear(2, embedding_size)
        
        self.pos_embedding = nn.Linear(2, embedding_size)
        # self.conv1_traj_img = nn.Conv2d(8,1,kernel_size = 1)
        
        
        self.attention_layer_lang = SoftDotAttention(hidden_size)

        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, hidden_size)
        
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            # nn.BatchNorm1d(256, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(32, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            )
        
        
        # self.fc = nn.Linear(im_feature_size+512*49, 64)
        self.fc = nn.Sequential(nn.Linear(im_feature_size, 128),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(768, eps=1e-12),
            nn.ReLU())



    def forward(self, current_direct, pos_input, lang_feature, h_0=None, c_0=None):
        

        direction = torch.concat((torch.sin(current_direct/180*3.14159),torch.cos(current_direct/180*3.14159)),axis = 1)
        direction_embeds = self.direction_embedding(direction)   # (batch, embedding_size)

        # batch_size * 37888
        if h_0 is None or c_0 is None:
            h_1,c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1,c_1 = self.direct_lstm(direction_embeds, (h_0,c_0))
   
        concat_h = h_1 # (batch, embedding_size+feature_size)
        lang_embeds, alpha = self.attention_layer_lang(concat_h, lang_feature ) 
        # lang_embeds torch.Size([6, 768])
        # h_sali = self.decoder_to_sali(im_embeds).squeeze(2)
        # # batch_size * 197

        output = self.decoder_2_action_full(lang_embeds)

        return h_1, c_1, output 

