import model.backbone.resnet as resnet
from model.backbone.xception import xception
from scipy.spatial.distance import cosine
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.uniform import Uniform
import torch.distributed as dist

class Protetype(nn.Module):
    def __init__(self):
        super(Protetype, self).__init__()

        memory_bank = nn.Parameter(torch.zeros(21, 1, 256, dtype=torch.float), requires_grad=False)
        self.memory_bank = memory_bank.cuda()

    def cals_cosine_similarity(self,feature1, feature2):
        # 将特征展平为一维数组
        # fet1 = feature1.clone()
        # fet2 = feature2.clone()
        # feature1_flat = fet1.flatten().detach().cpu().numpy()
        # feature2_flat = fet2.flatten().detach().cpu().numpy()
        # 计算余弦相似度
        cos_diss = nn.CosineSimilarity(dim=0, eps=1e-6)
        similarity = cos_diss(feature1, feature2)
        return similarity
    
    def calculate_loss(self,ls_featrure_per_class_weak,ls_featrure_per_class_strong):
        
        loss_prote = 0
        count = 0
        for i in range(0,len(ls_featrure_per_class_weak)):
            feature_weak = ls_featrure_per_class_weak[i] # 第i个类的特征
            feature_stro = ls_featrure_per_class_strong[i]
            feature_weak_normalized = torch.nn.functional.normalize(feature_weak.detach(), p=2, dim=1)
            feature_strong_normalized = torch.nn.functional.normalize(feature_stro, p=2, dim=1)

            cosine_similarity = torch.mm(feature_weak_normalized, feature_strong_normalized.transpose(0, 1))
            max_similarities = torch.max(cosine_similarity, dim=1).values
            similarity_avg = torch.sum(max_similarities)
            loss_prote = loss_prote + similarity_avg
            count = count + feature_stro.size()[0]
        
        if count ==0 :
            return torch.tensor(0)
        
        loss_prote  = 1 - loss_prote/count 
        return loss_prote



        # 返回10%的与原型相似的特征
    def select_feature(self,feature_weak,pred_weak,ignore_mask,probablity_weak):
        batch_size, num_channels, h, w = feature_weak.size()
        ignore_mask = F.interpolate(ignore_mask.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=True).squeeze(0)
        segmentation = pred_weak.long()
        ignore_mask = ignore_mask.long()
        features = feature_weak.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        probablity_weak = probablity_weak.view(-1)
        ans_sim = []
        ans = 0
        cos_diss2 = nn.CosineSimilarity(dim=1, eps=1e-6)
        for clsid in clsids:
            if clsid == 255: continue
            seg_cls = segmentation.view(-1) 
            ignore_mask_cls = ignore_mask.view(-1)
            # print(seg_cls.size(),ignore_mask_cls.size())
            mask =  (seg_cls == clsid) & (ignore_mask_cls!=255) & (probablity_weak > 0.95)
            feats_cls = features[ mask ]  # 筛选合适的类别特征

            if feats_cls.size()[0] == 0: continue
            ls_sim = []

            proty = self.memory_bank[clsid][0].repeat(feats_cls.size()[0],1)
           
            similarities = cos_diss2(proty,feats_cls)
            _, indices = torch.sort(similarities) # 从小到大的顺序

            first_ten = feats_cls[indices][-1].unsqueeze(0) # 这是相似度最大的特征
            # print(first_ten.size())
            select_nums = min(16,feats_cls.size()[0] )
            for i in range(1,select_nums):
                first_ten = torch.cat((first_ten,feats_cls[indices][-1 * i].unsqueeze(0)),dim=0)
            ans_sim.append(first_ten)
            
        return ans_sim

    
    def select_feature_low(self,feature_strong,pred_weak,ignore_mask):
        batch_size, num_channels, h, w = feature_strong.size()
        ignore_mask = F.interpolate(ignore_mask.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=True).squeeze(0)
        segmentation = pred_weak.long()
        ignore_mask = ignore_mask.long()
        features = feature_strong.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        ans_sim=[]
        cos_diss3 = nn.CosineSimilarity(dim=1, eps=1e-6)
        for clsid in clsids:
            if clsid == 255: continue
            seg_cls = segmentation.view(-1)
            ignore_mask_cls = ignore_mask.view(-1)
            mask =  (seg_cls == clsid) & (ignore_mask_cls!=255)
            feats_cls = features[ mask ]  # 筛选合适的类别特征
            if feats_cls.size()[0] == 0: continue
            ls_sim = []
            
            proty = self.memory_bank[clsid][0].repeat(feats_cls.size()[0],1)
           
            similarities = cos_diss3(proty,feats_cls)
            _, indices = torch.sort(similarities) # 从小到大的顺序

            first_ten = feats_cls[indices][0].unsqueeze(0) # 这是相似度最小的特征
            # print(first_ten.size())
            select_nums = min(256,feats_cls.size()[0])
            for i in range(1,select_nums):
                first_ten = torch.cat((first_ten,feats_cls[indices][i].unsqueeze(0)),dim=0)
            ans_sim.append(first_ten)
        return ans_sim
    
    def update_memory(self,features, segmentation,probablity_weak,ignore_mask,ignore_index=255):
        batch_size, num_channels, h, w = features.size()
        ignore_mask = F.interpolate(ignore_mask.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=True).squeeze(0)
        momentum = 0.99
        segmentation = segmentation.long()
        ignore_mask = ignore_mask.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        num_feats_per_cls = 1
        probablity_weak = probablity_weak.view(-1)

        need_update = True
        for clsid in clsids:
            if clsid == ignore_index: continue
            seg_cls = segmentation.view(-1) 
            ignore_mask_cls = ignore_mask.view(-1)
            mask =  (seg_cls == clsid) & (ignore_mask_cls!=255) & (probablity_weak > 0.95)
            feats_cls = features[mask]  # 筛选合适的类别特征
            if feats_cls.size()[0] == 0:
                continue
            for idx in range(num_feats_per_cls):
                if (self.memory_bank[clsid][idx] == 0).sum() == 256:
                    self.memory_bank[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue

            # similarity = F.cosine_similarity(feats_cls, self.memory_bank[clsid].data.expand_as(feats_cls))
            # weight = (1 - similarity) / (1 - similarity).sum()
            # feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
            feats_cls = feats_cls.mean(0)
            feats_cls = momentum * self.memory_bank[clsid].data + (1 - momentum) * feats_cls.unsqueeze(0)
            self.memory_bank[clsid].data.copy_(feats_cls)

        if dist.is_available() and dist.is_initialized():
            self.memory = self.memory_bank.data.clone()
            dist.all_reduce(self.memory.div_(dist.get_world_size()))
            self.memory_bank = nn.Parameter(self.memory, requires_grad=False)