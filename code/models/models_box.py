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

        self.entities = get_param((self.num_ent, self.emb_dim))
        self.relations = get_param((2*self.num_rel, self.emb_dim))
        self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
        self.relation_mins = get_param((2*self.num_rel, self.emb_dim))
        self.relation_maxs = get_param((2*self.num_rel, self.emb_dim))

        self.model_name = 'ShrinkE'
        self.device = config['DEVICE']

        self.min_fc = torch.nn.Linear(6*self.emb_dim, self.emb_dim)
        self.max_fc = torch.nn.Linear(6*self.emb_dim, self.emb_dim)

        self.bh = get_param((self.num_ent, 1))
        self.bt = get_param((self.num_ent, 1))

    def loss(self, pred, true_label):
        return self.bcelogitloss(pred, true_label)

    def forward(self, sub, rel, quals):
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

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.head_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.head_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.tail_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.tail_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)

#         head_trans_emb = torch.index_select(self.head_trans, 0, rel)
#         head_offset_emb = torch.index_select(self.head_offsets, 0, rel)

#         tail_trans_emb = torch.index_select(self.tail_trans, 0, rel)
#         tail_offset_emb = torch.index_select(self.tail_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, offset_emb, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, sub_emb, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((sub_emb, trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_r_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_r_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
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

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(3*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(3*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.offset_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )

#         self.center_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def intersection(self, boxes1, boxes2):
#         intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
#         intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)
#         intersection_box = Box(intersections_min, intersections_max)
#         return intersection_box

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         # diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.trans(sub_emb, trans_emb)

#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1

#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 
#         # return neg_dist

#     def trans(self, sub_emb, trans_emb):
#         return sub_emb + trans_emb
#         # return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb):
#         # rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb), -1) 
#         qual_obj_trans_emb = self.trans(qual_obj_emb, -qual_rel_trans_emb) 
#         box_mins = qual_obj_trans_emb - F.softplus(qual_rel_offset_emb)
#         box_maxs = qual_obj_trans_emb + F.softplus(qual_rel_offset_emb)
        
#         centers = (box_mins + box_maxs)/2 # 128*6*200
#         exp_centers = torch.exp(self.center_mlp(centers)) # 128*6*200
#         sum_exp_centers = torch.exp(self.center_mlp(centers)).sum(1,keepdim=True) # 128*1*200

#         attention = exp_centers/sum_exp_centers # 128*6*200
#         att_centers = attention*centers #128*6*200
#         att_centers = att_centers.sum(1) # 128*200
        
#         offsets = (box_maxs - box_mins) # 128*6*200
#         min_offsets = offsets.min(1)[0] # 128*200

#         deepset_offset = F.sigmoid(self.offset_mlp(self.offset_mlp(offsets).mean(1)))
#         offset = min_offsets * deepset_offset

#         qual_boxes = Box(att_centers-offset/2, att_centers+offset/2) 
#         boxes = self.intersection(boxes,qual_boxes)
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + 10*dist_outside
#         return dist 

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         # self.relation_off = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(2*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(2*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.offset_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )

#         self.center_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         # qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         # qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))
#         query_boxes = self.shrinking(query_boxes, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb)
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def trans(self, sub_emb, diag_emb, trans_emb):
#         return sub_emb + trans_emb
#         # return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb):
        

#         rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb), -1) 
#         # qual_obj_trans_emb = qual_obj_emb - qual_rel_trans_emb

#         # qual_boxes_mins = qual_obj_trans_emb - F.softplus(qual_rel_offset_emb)
#         # qual_boxes_maxs = qual_obj_trans_emb + F.softplus(qual_rel_offset_emb)

#         # quali_boxes_min_max = torch.cat((qual_boxes_mins, qual_boxes_maxs), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_center = F.relu(self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_offset = F.relu(self.max_fc(rel_key_value_emb))

#         centers = box_mins + shrinking_center

#         shrinking_box_maxs = centers + (box_maxs - centers)*shrinking_offset
#         shrinking_box_mins = centers - (centers - box_mins)*shrinking_offset
        
#         box_mins = shrinking_box_mins
#         box_maxs = shrinking_box_maxs
        
#         centers = (box_mins + box_maxs)/2 # 128*6*200

#         exp_centers = torch.exp(self.center_mlp(centers)) # 128*6*200
#         sum_exp_centers = torch.exp(self.center_mlp(centers)).sum(1,keepdim=True) # 128*1*200

#         attention = exp_centers/sum_exp_centers # 128*6*200
#         att_centers = attention*centers #128*6*200
#         att_centers = att_centers.sum(1) # 128*200
        
#         offsets = (box_maxs - box_mins) # 128*6*200
#         min_offsets = offsets.min(1)[0] # 128*200
#         deepset_offset = F.sigmoid(self.offset_mlp(self.offset_mlp(offsets).mean(1)))
#         offset = min_offsets * deepset_offset
#         boxes = Box(att_centers-offset/2, att_centers+offset/2) 
#         return boxes


#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + 10*dist_outside
#         return dist 


# trans shrinking
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, offset_emb, qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, sub_emb, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((sub_emb, trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_r_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_r_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
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

# shrinke no empty
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max
#         box_offset = F.softplus(box_maxs - box_mins)/2
#         centers = (box_mins + box_maxs)/2
#         box_mins = centers - box_offset
#         box_maxs = centers + box_offset
#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 

# Shrinking with only relation trans
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(5*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(5*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 

# qualifier shrinking + compatibility shrinking
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(4*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(4*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.min_r_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_r_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         # qualifier level shrinking
#         query_boxes = self.qualifier_shrinking(query_boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
#         query_boxes = self.compatibility_shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb
#     # sub_emb, trans_emb, diag_emb, offset_emb, 

#     def compatibility_shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_r_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_r_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def qualifier_shrinking(self, boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  

#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 


# shrinking by only qualifier
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(3*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(3*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.offset_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )

#         self.center_mlp = nn.Sequential(
#             torch.nn.Linear(self.emb_dim, self.emb_dim),
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         # return sub_emb + trans_emb
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     # def shrinking(self, boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#     #     rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#     #     box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#     #     box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#     #     box_widths = box_maxs - box_mins

#     #     shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#     #     box_mins = box_mins + shrinking_min
#     #     # box_widths = box_maxs - box_mins
#     #     shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)
#     #     box_maxs = box_maxs - shrinking_max
#     #     boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#     #     return boxes

#     def shrinking(self, boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)
#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         centers = (box_mins + box_maxs)/2 # 128*6*200

#         exp_centers = torch.exp(self.center_mlp(centers)) # 128*6*200
#         sum_exp_centers = torch.exp(self.center_mlp(centers)).sum(1,keepdim=True) # 128*1*200

#         attention = exp_centers/sum_exp_centers # 128*6*200
#         att_centers = attention*centers #128*6*200
#         att_centers = att_centers.sum(1) # 128*200
        
#         offsets = (box_maxs - box_mins) # 128*6*200
#         min_offsets = offsets.min(1)[0] # 128*200
#         deepset_offset = F.sigmoid(self.offset_mlp(self.offset_mlp(offsets).mean(1)))

#         offset = min_offsets * deepset_offset
 
#         boxes = Box(att_centers-offset/2, att_centers+offset/2) 
#         return boxes


#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 

# shrinking by no sub
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 

# shrinking by all
# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(8*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, sub_emb, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, sub_emb, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((sub_embedded, trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside

#         empty = (boxes_max - boxes_min).min(1)[0].unsqueeze(1)
#         empty = F.relu(-empty)**2
#         print(dist.shape, empty.shape)
#         dist = dist*(1+empty)
        
#         return dist 

# ----new_shrinking_with_sub

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(4*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(4*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.min_r_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_r_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.qualifier_shrinking(query_boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
#         query_boxes = self.compatibility_shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb
#     # sub_emb, trans_emb, diag_emb, offset_emb, 

#     def compatibility_shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_r_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_r_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def qualifier_shrinking(self, boxes, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         # sub_embedded = sub_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         # offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max

#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  

#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
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


## two-layer encoder

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
        
#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = torch.nn.Linear(7*self.emb_dim, self.emb_dim)
#         self.max_fc = torch.nn.Linear(7*self.emb_dim, self.emb_dim)
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def translation(self, sub_emb, trans_emb):
#         return sub_emb + trans_emb

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu(F.sigmoid( self.min_fc(rel_key_value_emb) )*box_widths)
#         shrinking_max = F.relu(F.sigmoid( self.max_fc(rel_key_value_emb) )*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max
        
#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
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

# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
#         # self.bceloss = torch.nn.BCELoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relations = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_mins = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_maxs = get_param((2*self.num_rel, self.emb_dim))

#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         self.min_fc = torch.nn.Linear(5*self.emb_dim, self.emb_dim)
#         self.max_fc = torch.nn.Linear(5*self.emb_dim, self.emb_dim)

#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         # print(self.num_rel, rel.max())
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         rel_emb = torch.index_select(self.relations, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)


#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)

#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_rel_emb = torch.index_select(self.relations, 0, quals_rels)
#         qual_off_emb = torch.index_select(self.relation_offsets, 0, quals_rels)

#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
#         qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
#         qual_off_emb = qual_off_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])

        
#         trans_emb = self.rot_trans(sub_emb, diag_emb,rel_emb)
#         query_boxes = Box(trans_emb - 0.5*F.softplus(offset_emb), trans_emb + 0.5*F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, diag_emb, rel_emb, offset_emb, qual_rel_emb, qual_off_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def translation(self, sub_emb, rel_emb):
#         return sub_emb + rel_emb

#     def rot_trans(self, sub_emb, diag_emb, rel_emb):
#         return givens_rotations(diag_emb, sub_emb) + rel_emb

#     def shrinking(self, boxes, rel_emb, diag_emb, offset_emb, qual_rel_emb, qual_off_emb, qual_obj_emb):
#         rel_embedded = rel_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        
#         rel_key_value_emb = torch.cat((rel_embedded, offset_embedded, qual_rel_emb, qual_off_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = torch.abs(F.sigmoid( self.min_fc(rel_key_value_emb) )*box_widths)
#         shrinking_max = torch.abs(F.sigmoid( self.max_fc(rel_key_value_emb) )*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max
        
#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
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


# class ShrinkE(StarEBase):

#     def __init__(self, config: dict):
#         super().__init__(config)

#         self.emb_dim = config['EMBEDDING_DIM']
#         self.bcelogitloss = torch.nn.BCEWithLogitsLoss()

#         self.entities = get_param((self.num_ent, self.emb_dim))
#         self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
#         self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

#         self.model_name = 'ShrinkE'
#         self.device = config['DEVICE']

#         # hidden_dim = 3*self.emb_dim
#         self.min_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )

#         self.max_fc = nn.Sequential(
#             torch.nn.Linear(7*self.emb_dim, self.emb_dim),
#             torch.nn.Sigmoid()
#            )
        
#         self.bh = get_param((self.num_ent, 1))
#         self.bt = get_param((self.num_ent, 1))

#     def loss(self, pred, true_label):
#         return self.bcelogitloss(pred, true_label)

#     def forward(self, sub, rel, quals):
#         sub_emb = torch.index_select(self.entities, 0, sub)
#         trans_emb = torch.index_select(self.relation_trans, 0, rel)
#         diag_emb = torch.index_select(self.relation_diags, 0, rel)
#         offset_emb = torch.index_select(self.relation_offsets, 0, rel)

#         quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
#         quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
#         qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
#         qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

#         qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
#         qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

#         qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
#         qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
#         qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
#         sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
#         query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

#         query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
#         bh = torch.index_select(self.bh, 0, sub) # bsz*1
#         bt = self.bt.t() # b=num_ent*1
        
#         neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
#         return torch.add(torch.add(neg_dist, bh), bt) 

#     def rot_trans(self, sub_emb, diag_emb, trans_emb):
#         return givens_rotations(diag_emb, sub_emb) + trans_emb

#     def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
#         trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

#         rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

#         box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
#         box_widths = box_maxs - box_mins

#         shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
#         shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

#         box_mins = box_mins + shrinking_min
#         box_maxs = box_maxs - shrinking_max
#         box_offset = F.softplus(box_maxs - box_mins)/2
#         centers = (box_mins + box_maxs)/2
#         box_mins = centers - box_offset
#         box_maxs = centers + box_offset
#         boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
#         return boxes

#     def point2box_distance(self, points, boxes):  
#         centres = 0.5 * (boxes.min_embed + boxes.max_embed)
#         boxes_min = boxes.min_embed
#         boxes_max = boxes.max_embed

#         dist_c = torch.cdist(centres, points, p=1)
#         dist_m = torch.cdist(boxes_min, points, p=1)
#         dist_M = torch.cdist(boxes_max, points, p=1)
#         dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

#         dist_inside = dist_c/dist_mM
#         dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
#         dist = dist_inside + dist_outside
#         return dist 