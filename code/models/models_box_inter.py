import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .gnn_encoder import StarEBase
from utils.utils_gcn import get_param
from utils.euclidean import givens_rotations


class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

    def volumes(self, dim=-1):
        return F.softplus(self.delta_embed).prod(dim, keepdim=True).clamp(1e-5,1e5)

class ShrinkE(StarEBase):
    """Baseline for Transformer decoder only model w/o starE encoder"""

    def __init__(self, config: dict):
        super().__init__(config)

        self.emb_dim = config['EMBEDDING_DIM']
        self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.BCELoss()

        self.entities = get_param((self.num_ent, self.emb_dim))
        self.relations = get_param((2*self.num_rel, self.emb_dim))
        self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
        self.relation_mins = get_param((2*self.num_rel, self.emb_dim))
        self.relation_maxs = get_param((2*self.num_rel, self.emb_dim))

        self.model_name = 'ShrinkE'
        self.device = config['DEVICE']

        self.min_fc = torch.nn.Linear(6*self.emb_dim, self.emb_dim)
        self.max_fc = torch.nn.Linear(6*self.emb_dim, self.emb_dim)

        # self.min_fc_1 = torch.nn.Linear(5*self.emb_dim, self.emb_dim)
        # self.max_fc_1 = torch.nn.Linear(5*self.emb_dim, self.emb_dim)

        # self.min_fc_2 = torch.nn.Linear(2*self.emb_dim, self.emb_dim)
        # self.max_fc_2 = torch.nn.Linear(2*self.emb_dim, self.emb_dim)

        self.bh = get_param((self.num_ent, 1))
        self.bt = get_param((self.num_ent, 1))

        self.alpha = 0.2

    def loss(self, pred, true_label):
        return self.bcelogitloss(pred, true_label)

    def forward(self, sub, rel, quals):
        # print(self.num_rel, rel.max())
        sub_emb = torch.index_select(self.entities, 0, sub)
        rel_emb = torch.index_select(self.relations, 0, rel)
        diag_emb = torch.index_select(self.relation_diags, 0, rel)
        min_emb = torch.index_select(self.relation_mins, 0, rel)
        max_emb = torch.index_select(self.relation_maxs, 0, rel)

        quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
        quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
        qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
        qual_rel_emb = torch.index_select(self.relations, 0, quals_rels)
        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
        qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        # qual_min_emb = torch.index_select(self.relation_mins, 0, quals_rels)
        # qual_max_emb = torch.index_select(self.relation_maxs, 0, quals_rels)
        # qual_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)

        # qual_min_emb = qual_min_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
        # qual_max_emb = qual_max_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
        # qual_diag_emb = qual_diag_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        # trans_emb = self.translation(sub_emb, rel_emb)

        trans_emb = self.rot_trans(sub_emb, diag_emb, rel_emb)
        query_boxes = Box(trans_emb - F.softplus(min_emb), trans_emb + F.softplus(max_emb))

        query_boxes = self.shrinking(query_boxes, rel_emb, diag_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb)
        # query_boxes = self.bilevel_shrinking(query_boxes, rel_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb)
        # query_boxes = self.bilevel_shrinking(query_boxes, rel_emb, diag_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb)
        
        bh = torch.index_select(self.bh, 0, sub) # bsz*1
        bt = self.bt.t() # b=num_ent*1
        
        neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
        return torch.add(torch.add(neg_dist, bh), bt) 

    def translation(self, sub_emb, rel_emb):
        return sub_emb + rel_emb

    def rot_trans(self, sub_emb, diag_emb, rel_emb):
        return givens_rotations(diag_emb, sub_emb) + rel_emb

    # def rot_trans_inverse(self, sub_emb, diag_emb, rel_emb):
    #     return givens_rotations(-diag_emb, sub_emb-rel_emb)

    def shrinking(self, boxes, rel_emb, diag_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb):
        rel_embedded = rel_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        min_embedded = min_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        max_embedded = max_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

        rel_key_value_emb = torch.cat((rel_embedded, diag_embedded, min_embedded, max_embedded, qual_rel_emb, qual_obj_emb), -1) 

        box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_widths = box_maxs - box_mins

        # shrinking_min = F.sigmoid(self.min_fc(rel_key_value_emb))*box_widths
        # shrinking_max = F.sigmoid(self.max_fc(rel_key_value_emb))*box_widths

        shrinking_min = torch.abs(F.sigmoid( self.min_fc(rel_key_value_emb) )*box_widths)
        shrinking_max = torch.abs(F.sigmoid( self.max_fc(rel_key_value_emb) )*box_widths)

        box_mins = box_mins + shrinking_min
        box_maxs = box_maxs - shrinking_max
        
        boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
        return boxes

    # def bilevel_shrinking(self, boxes, rel_emb, diag_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb):
    #     rel_embedded = rel_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     min_embedded = min_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     max_embedded = max_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     rel_key_emb = torch.cat((rel_embedded, diag_embedded, min_embedded, max_embedded, qual_rel_emb), -1) 
    #     key_value_emb = torch.cat((qual_rel_emb, qual_obj_emb), -1) 
 
    #     box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     box_widths = box_maxs - box_mins

    #     # shrinking_min = torch.abs( (F.sigmoid( self.min_fc_1(rel_key_emb) ) * F.sigmoid(self.min_fc_2(key_value_emb))) *box_widths)
    #     # shrinking_max = torch.abs( (F.sigmoid( self.max_fc_1(rel_key_emb)) * F.sigmoid(self.max_fc_2(key_value_emb))) *box_widths)

    #     shrinking_min = torch.abs(F.sigmoid(self.min_fc_1(rel_key_emb) + self.min_fc_2(key_value_emb) )*box_widths)
    #     shrinking_max = torch.abs(F.sigmoid(self.max_fc_1(rel_key_emb) + self.max_fc_2(key_value_emb) )*box_widths)

    #     box_mins = box_mins + shrinking_min
    #     box_maxs = box_maxs - shrinking_max
        
    #     boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
    #     return boxes

    # def bilevel_shrinking(self, boxes, rel_emb, diag_emb, min_emb, max_emb, qual_rel_emb, qual_obj_emb):
    #     rel_embedded = rel_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     min_embedded = min_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     max_embedded = max_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     rel_key_emb = torch.cat((rel_embedded, diag_embedded, min_embedded, max_embedded, qual_rel_emb), -1) 
    #     key_value_emb = torch.cat((qual_rel_emb, qual_obj_emb), -1) 

    #     box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
    #     box_widths = box_maxs - box_mins

    #     shrinking_min = torch.abs(F.sigmoid(self.min_fc_1(rel_key_emb))*box_widths)
    #     shrinking_max = torch.abs(F.sigmoid(self.max_fc_1(rel_key_emb))*box_widths)
    #     box_mins = box_mins + shrinking_min
    #     box_maxs = box_maxs - shrinking_max
    #     box_widths = box_maxs - box_mins
        
    #     shrinking_min = torch.abs(F.sigmoid(self.min_fc_2(key_value_emb))*box_widths)
    #     shrinking_max = torch.abs(F.sigmoid(self.max_fc_2(key_value_emb))*box_widths)
    #     box_mins = box_mins + shrinking_min
    #     box_maxs = box_maxs - shrinking_max
    #     boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
    #     return boxes

    def point2box_distance(self, points, boxes, alpha=0.2):  
        # points = num_ent*dim, boxes.min_embed = bsz*dim, boxes.max_embed = bsz*dim
        centres = 0.5 * (boxes.min_embed + boxes.max_embed)
        # width = F.softplus(boxes.max_embed - boxes.min_embed)
        boxes_min = boxes.min_embed
        boxes_max = boxes.max_embed

        dist_c = torch.cdist(centres, points, p=1)
        dist_m = torch.cdist(boxes_min, points, p=1)
        dist_M = torch.cdist(boxes_max, points, p=1)
        dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

        dist_inside = dist_c/dist_mM
        dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
        dist = dist_inside + dist_outside
        return dist 

# class IntersectionE(StarEBase):
#     """Baseline for Transformer decoder only model w/o starE encoder"""

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relations = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_mins = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_maxs = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'IntersectionE'
#         self.device = config['DEVICE']

#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#         self.alpha = 0.2

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         # print(self.num_rel, rel.max())
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         rel_emb = torch.index_select(self.relations, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         min_emb = torch.index_select(self.relation_mins, 0, rel)
#         max_emb = torch.index_select(self.relation_maxs, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_rel_emb = torch.index_select(self.relations, 0, quals_rels)
#         qual_min_emb = torch.index_select(self.relation_mins, 0, quals_rels)
#         qual_max_emb = torch.index_select(self.relation_maxs, 0, quals_rels)

#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
#         qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
#         qual_min_emb = qual_min_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
#         qual_max_emb = qual_max_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

#         trans_emb = self.translation(sub_emb, rel_emb)
#         # trans_emb = self.rot_trans(sub_emb, diag_emb, rel_emb)
#         query_boxes = Box(trans_emb - F.softplus(min_emb), trans_emb + F.softplus(max_emb))
        
#         query_boxes = self.intersection(query_boxes, qual_rel_emb, qual_min_emb, qual_max_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def translation(self, sub_emb, rel_emb):
#         return sub_emb + rel_emb

#     def rot_trans(self, sub_emb, diag_emb, rel_emb):
#         return givens_rotations(diag_emb, sub_emb) + rel_emb

#     def intersection(self, boxes, qual_rel_emb, qual_min_emb, qual_max_emb, qual_obj_emb):
#         trans_emb = self.translation(qual_obj_emb, -qual_rel_emb)
#         box_mins = trans_emb -  F.softplus(qual_min_emb)
#         box_maxs = trans_emb +  F.softplus(qual_max_emb)
#         qual_boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         boxes = Box(torch.max(boxes.min_embed, qual_boxes.min_embed), torch.min(boxes.max_embed, qual_boxes.max_embed) )
#         return boxes

#     def point2box_distance(self, points, boxes, alpha=0.2):  
#         # points = num_ent*dim, boxes.min_embed = bsz*dim, boxes.max_embed = bsz*dim
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         # width = F.softplus(boxes.max_embed - boxes.min_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(F.softplus(boxes_max - boxes_min),p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 


