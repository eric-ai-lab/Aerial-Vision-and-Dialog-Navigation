import torch
from models.enc_visual import FeatureFlat
from models.enc_vl import EncoderVL
from models.encodings import DatasetLearnedEncoding
from models import model_util
from torch import nn
from torch.nn import functional as F

import numpy as np








        # output = self.decoder_2_action_full(action_module_input)







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
        
        self.c = nn.Sequential(
        nn.Linear(768, 256),
        # nn.BatchNorm1d(64, eps=1e-12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 32),
        # nn.BatchNorm1d(64, eps=1e-12),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 4),
        # nn.BatchNorm1d(768, eps=1e-12),
        nn.ReLU())

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


class ET(nn.Module):
    def __init__(self, args):
        """
        transformer agent
        """
        super().__init__()
        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # # feature embeddings
        # self.vis_feat = FeatureFlat(input_shape=self.visual_tensor_shape, output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None
        
        # self.vis_feat = FeatureFlat(input_shape=(650,7,7), output_size=args.demb)

        self.args = args
        # decoder parts
        encoder_output_size = args.demb
        self.dec_action = nn.Linear(encoder_output_size, args.demb)

        # XVIEW
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(768, 256),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
            )
        self.attention_layer_vision = SoftDotAttention(49)
        self.fc = nn.Sequential(nn.Linear(768, 64),
            # nn.BatchNorm1d(64, eps=1e-12),
            nn.Dropout(0.2),
            nn.ReLU())
        self.direction_embedding = nn.Linear(2, 768)

        self.fc2 = nn.Linear(49, 768)
        
        # final touch
        self.init_weights()

    def forward(self, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # embed language
        output = {}
        emb_lang = inputs["lang"]

        # # embed frames and direiction (650,49) --> 768
        # im_feature = inputs["frames"]
        # embed_frame, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:,-1, :, :])        
        # h_sali = self.fc(embed_frame).view(-1,1,8,8)
        # pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)
        # frames_pad_emb = self.vis_feat(im_feature.view(-1, 650,7,7)).view(*im_feature.shape[:2], -1)
        

        # embed frames and direiction (1,49) --> 768
        im_feature = inputs["frames"]
        att_frame_feature = torch.zeros((im_feature.shape[0],0,49)).cuda()
        for i in range(im_feature.shape[1]):
            att_single_frame_feature, beta = self.attention_layer_vision(inputs["lang_cls"], im_feature[:,i,:,:])       
            att_frame_feature = torch.concat((att_frame_feature, att_single_frame_feature.unsqueeze(1)), axis = 1)

        emb_frames = self.fc2(att_frame_feature.view(-1,49)).view(*im_feature.shape[:2], -1)


        emb_directions = self.direction_embedding(inputs["directions"].view(-1,2)).view(im_feature.shape[0], -1, 768)   # (batch, embedding_size)

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_directions,
            inputs['lenths']
        )
        # use outputs corresponding to last visual frames for prediction only
        encoder_out_visual = encoder_out[:, emb_lang.shape[1] + np.max(inputs['lenths'])-1]
        encoder_out_direction = encoder_out[:, emb_lang.shape[1] + 2* np.max(inputs['lenths'])-1]
        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_decoder_input = encoder_out_direction.reshape(-1, self.args.demb)

        # decoder_input = emb_directions[:,-1].reshape(-1, self.args.demb)
        output = self.decoder_2_action_full(action_decoder_input)
        
        h_sali = self.fc(decoder_input).view(-1,1,8,8)
        pred_saliency = nn.functional.interpolate(h_sali,size=(224,224),mode='bilinear',align_corners=False)


        # # get the output objects
        # emb_object_flat = emb_object.view(-1, self.args.demb)
        # decoder_input = decoder_input + emb_object_flat
        # object_flat = self.dec_object(decoder_input)
        # objects = object_flat.view(*encoder_out_visual.shape[:2], *object_flat.shape[1:])
        # output.update({"action": action, "object": objects})

        # (optionally) get progress monitor predictions
        # if self.args.progress_aux_loss_wt > 0:
        #     progress = torch.sigmoid(self.dec_progress(encoder_out_visual))
        #     output["progress"] = progress
        # if self.args.subgoal_aux_loss_wt > 0:
        #     subgoal = torch.sigmoid(self.dec_subgoal(encoder_out_visual))
        #     output["subgoal"] = subgoal
        return output,pred_saliency

    # def embed_lang(self, lang_pad, vocab):
    #     """
    #     take a list of annotation tokens and extract embeddings with EncoderLang
    #     """
    #     assert lang_pad.max().item() < len(vocab)
    #     embedder_lang = self.embs_ann[vocab.name]
    #     emb_lang, lengths_lang = self.encoder_lang(lang_pad, embedder_lang, vocab, self.pad)
    #     if self.args.detach_lang_emb:
    #         emb_lang = emb_lang.clone().detach()
    #     return emb_lang, lengths_lang

    # def embed_frames(self, frames_pad):
    #     """
    #     take a list of frames tensors, pad it, apply dropout and extract embeddings
    #     """
    #     self.dropout_vis(frames_pad)
    #     frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
    #     frames_pad_emb = self.vis_feat(frames_4d).view(*frames_pad.shape[:2], -1)
    #     frames_pad_emb_skip = self.object_feat(frames_4d).view(*frames_pad.shape[:2], -1)
    #     return frames_pad_emb, frames_pad_emb_skip

    # def embed_actions(self, actions):
    #     """
    #     embed previous actions
    #     """
    #     emb_actions = self.emb_action(actions)
    #     emb_actions = self.dropout_action(emb_actions)
    #     return emb_actions



    def compute_batch_loss(self, model_out, gt_dict):
        """
        loss function for Seq2Seq agent
        """
        losses = dict()

        # action loss
        action_pred = model_out["action"].view(-1, model_out["action"].shape[-1])
        action_gt = gt_dict["action"].view(-1)
        pad_mask = action_gt != self.pad

        # Calculate loss only over future actions
        action_pred_mask = gt_dict["driver_actions_pred_mask"].view(-1)

        action_loss = F.cross_entropy(action_pred, action_gt, reduction="none")
        action_loss *= pad_mask.float()
        if not self.args.compute_train_loss_over_history:
            action_loss *= action_pred_mask.float()
        action_loss = action_loss.mean()
        losses["action"] = action_loss * self.args.action_loss_wt

        # object classes loss
        if len(gt_dict["object"]) > 0:
            object_pred = model_out["object"]
            object_gt = torch.cat(gt_dict["object"], dim=0)

            if self.args.compute_train_loss_over_history:
                interact_idxs = gt_dict["obj_interaction_action"].view(-1).nonzero(as_tuple=False).view(-1)
            else:
                interact_idxs = (
                    (gt_dict["driver_actions_pred_mask"] * gt_dict["obj_interaction_action"])
                    .view(-1)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
            if interact_idxs.nelement() > 0:
                object_pred = object_pred.view(object_pred.shape[0] * object_pred.shape[1], *object_pred.shape[2:])
                object_loss = model_util.obj_classes_loss(object_pred, object_gt, interact_idxs)
                losses["object"] = object_loss * self.args.object_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            subgoal_pred = model_out["subgoal"].squeeze(2)
            subgoal_gt = gt_dict["subgoals_completed"]
            subgoal_loss = F.mse_loss(subgoal_pred, subgoal_gt, reduction="none")
            subgoal_loss = subgoal_loss.view(-1) * pad_mask.float()
            subgoal_loss = subgoal_loss.mean()
            losses["subgoal_aux"] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.progress_aux_loss_wt > 0:
            progress_pred = model_out["progress"].squeeze(2)
            progress_gt = gt_dict["goal_progress"]
            progress_loss = F.mse_loss(progress_pred, progress_gt, reduction="none")
            progress_loss = progress_loss.view(-1) * pad_mask.float()
            progress_loss = progress_loss.mean()
            losses["progress_aux"] = self.args.progress_aux_loss_wt * progress_loss

        # maximize entropy of the policy if asked
        if self.args.entropy_wt > 0.0:
            policy_entropy = -F.softmax(action_pred, dim=1) * F.log_softmax(action_pred, dim=1)
            policy_entropy = policy_entropy.mean(dim=1)
            policy_entropy *= pad_mask.float()
            losses["entropy"] = -policy_entropy.mean() * self.args.entropy_wt

        return losses

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, compute_train_loss_over_history):
        """
        compute exact matching and f1 score for action predictions
        """
        preds = model_util.extract_action_preds(model_out, self.pad, self.vocab_out, lang_only=True)
        stop_token = self.vocab_out.word2index("Stop")
        gt_actions = model_util.tokens_to_lang(gt_dict["action"], self.vocab_out, {self.pad, stop_token})
        model_util.compute_f1_and_exact(metrics_dict, [p["action"] for p in preds], gt_actions, "action")
        model_util.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out["object"], compute_train_loss_over_history
        )
    
    def compute_loss(self, model_outs, gt_dicts):
        """
        compute the loss function for several batches
        """
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(model_outs[dataset_key], gt_dicts[dataset_key])
        return losses