import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .gnn_encoder import StarEBase
from utils.utils_gcn import get_param
from utils.euclidean import givens_rotations

#
class ShrinkE(StarEBase):

    def __init__(self, config: dict):
        super().__init__(config)

        self.emb_dim = config['EMBEDDING_DIM']
        self.bcelogitloss = torch.nn.BCEWithLogitsLoss()

        self.entities = get_param((self.num_ent, self.emb_dim))
        self.relation_trans = get_param((2*self.num_rel, self.emb_dim))
        self.relation_diags = get_param((2*self.num_rel, self.emb_dim))
        self.relation_offsets = get_param((2*self.num_rel, self.emb_dim))

        self.model_name = 'ShrinkE'
        self.device = config['DEVICE']

        # hidden_dim = 3*self.emb_dim
        self.min_fc = nn.Sequential(
            torch.nn.Linear(7*self.emb_dim, self.emb_dim),
            torch.nn.Sigmoid()
           )

        self.max_fc = nn.Sequential(
            torch.nn.Linear(7*self.emb_dim, self.emb_dim),
            torch.nn.Sigmoid()
           )
        
        self.bh = get_param((self.num_ent, 1))
        self.bt = get_param((self.num_ent, 1))

    def loss(self, pred, true_label):
        return self.bcelogitloss(pred, true_label)

    def forward(self, sub, rel, quals):
        sub_emb = torch.index_select(self.entities, 0, sub)
        trans_emb = torch.index_select(self.relation_trans, 0, rel)
        diag_emb = torch.index_select(self.relation_diags, 0, rel)
        offset_emb = torch.index_select(self.relation_offsets, 0, rel)

        quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
        quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
        qual_obj_emb = torch.index_select(self.entities, 0, quals_ents)
        qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])

        qual_rel_trans_emb = torch.index_select(self.relation_trans, 0, quals_rels)
        qual_rel_trans_emb = qual_rel_trans_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])

        qual_rel_diag_emb = torch.index_select(self.relation_diags, 0, quals_rels)
        qual_rel_diag_emb = qual_rel_diag_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
        qual_rel_offset_emb = torch.index_select(self.relation_offsets, 0, quals_rels)
        qual_rel_offset_emb = qual_rel_offset_emb.view(trans_emb.shape[0], -1, trans_emb.shape[1])
        
        sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
        query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

        query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
        
        bh = torch.index_select(self.bh, 0, sub) # bsz*1
        bt = self.bt.t() # b=num_ent*1
        
        neg_dist = - self.point2box_distance(self.entities, query_boxes) # bsz*num_ent
        return torch.add(torch.add(neg_dist, bh), bt) 

    def rot_trans(self, sub_emb, diag_emb, trans_emb):
        return givens_rotations(diag_emb, sub_emb) + trans_emb

    def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
        trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

        rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

        box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_widths = box_maxs - box_mins

        shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
        shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

        box_mins = box_mins + shrinking_min
        box_maxs = box_maxs - shrinking_max
        box_offset = F.softplus(box_maxs - box_mins)/2
        centers = (box_mins + box_maxs)/2
        box_mins = centers - box_offset
        box_maxs = centers + box_offset
        boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
        return boxes

    def point2box_distance(self, points, boxes):  
        centres = 0.5 * (boxes.min_embed + boxes.max_embed)
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

