import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from scipy.spatial.distance import cosine
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.contrast_loss import StudentSegContrast
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from util.consistency import consistency_weight
import numpy as np
import torch.distributed as dist
from prote import Protetype

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default="/home/user/HFPL/configs/pascal.yaml")
parser.add_argument('--labeled-id-path', type=str, default="/home/user/HFPL/splits/pascal/1464/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="/home/user/HFPL/splits/pascal/1464/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="/home/user/HFPL/log_1464")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=16107, type=int)



# 初始化内存库
# memory_bank = nn.Parameter(torch.zeros(21, 1, 256, dtype=torch.float), requires_grad=False)
# memory_bank = memory_bank.cuda()

def upsample(features,h,w):
    return F.interpolate(features, size=(h, w), mode="bilinear", align_corners=True)

def calculate_region(feature,label,cfg,ignore_mask,logger):
    batch_size,c_channel,h,w = feature.size()
    num_classes = label.size(1)
    feats_sl = torch.zeros(batch_size, h*w, c_channel).type_as(feature)
    max_logit = label.softmax(dim=1).max(dim=1)[0] # 取最大值
    index = (max_logit >= cfg['conf_thresh']) # B H W
    feature_temp = feature.clone()
    feature_temp = feature_temp.reshape(batch_size, h*w, c_channel)
    # logger.info("大于阈值的像素个数所占比例为: %s %s"%(index.sum()/(batch_size*h*w), index.sum()))
    for batch_idx in range(batch_size):
        ##### (C, H, W), (num_classes, H, W) --> (H*W, C), (H*W, num_classes)
        feats_iter, preds_iter = feature[batch_idx], label[batch_idx]
        feats_iter, preds_iter = feats_iter.reshape(c_channel, -1), preds_iter.reshape(num_classes, -1)
        feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
        index_batch = index[batch_idx] # H * W
        index_batch = index_batch.reshape(h*w)
        batch_ignore_index = ignore_mask[batch_idx] # H * W
        batch_ig = batch_ignore_index.reshape(h*w)
        argmax = preds_iter.argmax(1) # 直接 返回下标索引
        feature_temp_index = feature_temp[batch_idx]
        for clsid in range(num_classes):
            maskk = (argmax == clsid) # 为该类的掩码 H * 1 
            mask = (maskk == True) & (index_batch == True) & (batch_ig != 255) # 符合条件的像素个数
            if mask.sum() == 0: continue  # 没有这个类就放弃
            #logger.info("属于类%s的像素个数为:%s"%(clsid,maskk.sum()))
            #logger.info("满足条件的像素个数为:%s"%(mask.sum()))
            feats_iter_cls = feats_iter[mask] # (h*w) × C
            preds_iter_cls = preds_iter[:, clsid][mask] # 抽取特定类的最大值出来
            weight = F.softmax(preds_iter_cls, dim=0) # 进行softmax 
            feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1) # 加权乘积等到一个类的区域特征
            feats_iter_cls = feats_iter_cls.sum(0)
            feats_sl[batch_idx][maskk] = feats_iter_cls # 得到对应的区域特征
        feats_sl[batch_idx][batch_ig == 255] =  feature_temp_index[batch_ig == 255]


    feats_sl = feats_sl.reshape(batch_size, h, w, c_channel)
    feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous() # 返回B C H W


    return feats_sl

def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=0, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size() # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold) # 筛选出大于阈值的
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean') # take the mean over the batch_size



    

def main():
    args =  parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])

    model = DeepLabV3Plus(cfg)
    
    
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    
    
    if rank == 0:
        logger.info('Model params: {:.1f}M \n'.format(count_params(model)))

    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    cons_w_unsup = consistency_weight(final_w=cfg['unsupervised_w'], iters_per_epoch=len(trainloader_u),
                                        rampup_ends=int(cfg['epochs']))

    



    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    feature_per_loss = softmax_mse_loss
    pro = Protetype()
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_drop = AverageMeter()
        total_loss_noise_loss = AverageMeter()
        loss_feature_ali_loss = AverageMeter()
        loss_prote_all = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        
        # img_u_w_mix 是经正则化的弱增强图片，img_u_s1_mix 是经过强增强的图片，ignore_mask_mix 是忽略的像素 
        # cutmix_box1 cutmix_box2 是cutmix的坐标
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            img_1 = img_u_w.clone()
            img_2 = img_u_w.clone()
            b,c,h,w = img_1.size()
            img_s_s1 = img_u_s1.clone()
            ignore_mask_for_u_1 = ignore_mask.clone()
            ignore_mask_for_u_2 = ignore_mask.clone()
            ignore_mask_for_s_1 = ignore_mask.clone()
            ignore_mask_for_s_2 = ignore_mask.clone()

            iters = epoch * len(trainloader_u) + i

            weight_u = cons_w_unsup(epoch=epoch,curr_iter=i)

            with torch.no_grad():
                model.eval()

                featur_u_w_mix,pred_u_w_mix = model(img_u_w_mix) # 对增强图片的预测
                pred_u_w_mix = pred_u_w_mix.detach()
                probability_mix = F.softmax(pred_u_w_mix,dim=1)
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0] # 每个类的概率
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1) # 标签

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] # 强增强图片之间的cutmix
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            img_1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_w_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_w_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            ignore_mask_for_u_1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            ignore_mask_for_u_2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            ignore_mask_for_s_1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]
            ignore_mask_for_s_2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            
            
            # 产生的特征都是256维度的
            model.train()
            feature_1,label_u_w_1_large,label_u_w_1 = model(img_1,need_pre_logit = True) # 弱增强的第一个cutmix图片特征
            feature_2,label_u_w_2_large,label_u_w_2 = model(img_2,need_pre_logit = True) # 若增强的第二个cutmix图片特征

            

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            feature_weak, _ , pred_weak = model(img_u_w,need_pre_logit=True)
            probablity_weak = F.softmax(pred_weak,dim=1)

            label_weak_prote = probablity_weak.argmax(dim=1)
            probablity_weak = probablity_weak.max(dim=1)[0]
            feature_strong, _ , pred_strong = model(img_s_s1,need_pre_logit=True)

            feature_all,preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True) # 所有的预测结果 以及 扰动的预测结果  这里没有加入Cutmix
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])

            pred_u_w_fp = preds_fp[num_lb:] # 无标签数据的扰动特征

            feature_u_all, pred_u_all,pre_logit_all = model(torch.cat((img_u_s1, img_u_s2)),need_pre_logit = True) # 经过cutmix之后得到的 预测结果

            pred_u_s1, pred_u_s2 = pred_u_all.chunk(2)
            pre_logit_s1, pre_logit_s2 = pre_logit_all.chunk(2)

            cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)

            #像素级别的特征差异化损失
            feature_u_s1,feature_u_s2 = feature_u_all.chunk(2)
            loss_feature_1 = 1 - cos_dis(feature_1.detach(),feature_u_s1).mean()
            loss_feature_2 = 1 - cos_dis(feature_2.detach(),feature_u_s2).mean()

            # 区域之间的特征差异化损失
            # 先传入弱增强的图片特征 
            bb,cc,hh,ww = feature_u_s2.size()
            ## 进行特征扰动
            Similarity_1 = cos_dis(feature_1.clone().detach(),feature_u_s1).reshape(bb,hh*ww) # B × (H*W)
            Similarity_1 = Similarity_1.mean(1)  # 取均值

            

            Similarity_2 = cos_dis(feature_2.clone().detach(),feature_u_s2).reshape(bb,hh*ww) # B × (H*W)
            Similarity_2 = Similarity_2.mean(1)  # 取均值

            # 从每个类中筛选与当前内存库最相似的前10%的特征
            ls_featrure_per_class_weak = pro.select_feature(feature_weak,label_weak_prote.detach(),ignore_mask,probablity_weak)

            # # 从强增强的数据中选择与内存库最不相似的前50%的特征
            ls_featrure_per_class_strong = pro.select_feature_low(feature_strong,label_weak_prote.detach(),ignore_mask)

            loss_prote = pro.calculate_loss(ls_featrure_per_class_weak,ls_featrure_per_class_strong)

            # Similarity_1_drop = Similarity_1.clone()
            # Similarity_2_drop = Similarity_2.clone()

            # 更新内存库中的内容
            pro.update_memory(feature_weak,label_weak_prote,probablity_weak,ignore_mask)
            
            pred_u_w = pred_u_w.detach() # 弱增强图片的预测
            probability_u_w = F.softmax(pred_u_w,dim=1)
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            probability_clone_mix = probability_mix.clone()
            probability_clone_u_w_1 = probability_u_w.clone()
            probability_clone_u_w_2 = probability_u_w.clone()
            cutmix_box1_p1 = cutmix_box1.clone()
            cutmix_box2_p2 = cutmix_box2.clone()

            cutmix_box1_p1 = cutmix_box1_p1.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            cutmix_box2_p2 = cutmix_box2_p2.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            probability_clone_u_w_1[cutmix_box1_p1 == 1] = probability_clone_mix[cutmix_box1_p1 == 1]
            probability_clone_u_w_2[cutmix_box2_p2 == 1] = probability_clone_mix[cutmix_box2_p2 == 1]

            if epoch >= cfg['warm_up']:
                if cfg['F-Noise']:
                    feats_s1_temp = torch.zeros(bb, cc,hh,ww).type_as(feature_u_s1).cuda()
                    for ii in range(0,bb):
                        # 得到每张图片的均值
                        Similarity_1[ii] = ((Similarity_1[ii] + 1) * 3)/20  # 之前是20
                        uni_dist = Uniform(-1 * Similarity_1[ii].item(), Similarity_1[ii].item())
                        # uni_dist = Uniform(-1 * 0.3, 0.3)
                        noise = uni_dist.sample(feature_u_s1.shape[1:]).cuda().unsqueeze(0)
                        feats_s1_temp[ii] = feature_u_s1[ii].mul(noise) + feature_u_s1[ii]
                        
                    
                    feats_s2_temp = torch.zeros(bb, cc,hh,ww).type_as(feature_u_s2).cuda()
                    
                    for ii in range(0,bb):
                        Similarity_2[ii] = ((Similarity_2[ii] + 1 ) * 3)/20 # 
                        uni_dist_2 = Uniform(-1 * Similarity_2[ii].item(), Similarity_2[ii].item())
                        # uni_dist_2 = Uniform(-1 * 0.3, 0.3)
                        noise_2 = uni_dist_2.sample(feature_u_s2.shape[1:]).cuda().unsqueeze(0)
                        feats_s2_temp[ii] = feature_u_s2[ii].mul(noise_2) + feature_u_s2[ii]
                    
                    # 对添加噪声的数据进行解码
                    Pred_noise_s1 = model(x=feats_s1_temp,only_need_decoder=True)
                    Pred_noise_s1 = F.interpolate(Pred_noise_s1, size=(h, w), mode="bilinear", align_corners=True)
                    #Pred_noise_s1 = model.module.decoder_noise(feats_s1_temp,h,w)
                    Pred_noise_s2 = model(x=feats_s2_temp,only_need_decoder=True)
                    Pred_noise_s2 = F.interpolate(Pred_noise_s2, size=(h, w), mode="bilinear", align_corners=True)
                    # Pred_noise_s2 = model.module.decoder_noise(feats_s2_temp,h,w)

                    loss_noise_1 = softmax_mse_loss(inputs=Pred_noise_s1, targets=probability_clone_u_w_1)
                    loss_noise_2 = softmax_mse_loss(inputs=Pred_noise_s2, targets=probability_clone_u_w_2)

                    loss_noise = (loss_noise_1 + loss_noise_2)/2.0


                if cfg['F-Drop']:
                    drop_mask_feature1 = torch.zeros(bb,cc,hh,ww).cuda()
                    for ii in range(0,bb):
                        Similarity_1[ii] = ((Similarity_1[ii] + 1) * 3)/20
                        attention = torch.mean(feature_u_s1[ii],dim=0,keepdim=True)
                        max_val, _ = torch.max(attention.view(-1),dim=0, keepdim=True)
                        left = (1-Similarity_1[ii]).item() - 0.1
                        right = (1-Similarity_1[ii]).item() + 0.1
                        threshold = max_val * np.random.uniform(left,right)
                        threshold = threshold.view(1, 1,1).expand_as(attention)
                        drop_mask_1 = (attention < threshold).float()
                        drop_mask_feature1[ii] =  feature_u_s1[ii].mul(drop_mask_1)

                    drop_mask_feature2 = torch.zeros(bb,cc,hh,ww).cuda()
                    for ii in range(0,bb):
                        Similarity_2[ii] = ((Similarity_2[ii] + 1) * 3)/20
                        attention_2 = torch.mean(feature_u_s2[ii],dim=0,keepdim=True)
                        max_val_2, _ = torch.max(attention_2.view(-1),dim=0, keepdim=True)
                        left = (1-Similarity_2[ii]).item() - 0.1
                        right = (1-Similarity_2[ii]).item() + 0.1
                        threshold_2 = max_val_2 * np.random.uniform(left, right)
                        threshold_2 = threshold_2.view(1, 1, 1).expand_as(attention_2)
                        drop_mask_2 = (attention_2 < threshold_2).float()
                        drop_mask_feature2[ii] =  feature_u_s2[ii].mul(drop_mask_2)

                    Pred_drop_s1 = model(x=drop_mask_feature1,only_need_decoder=True)
                    Pred_drop_s1 = F.interpolate(Pred_drop_s1, size=(h, w), mode="bilinear", align_corners=True)
                    #Pred_noise_s1 = model.module.decoder_noise(feats_s1_temp,h,w)
                    Pred_drop_s2 = model(x=drop_mask_feature2,only_need_decoder=True)
                    Pred_drop_s2 = F.interpolate(Pred_drop_s2, size=(h, w), mode="bilinear", align_corners=True)

                    loss_drop_s1 = softmax_mse_loss(inputs=Pred_drop_s1, targets=probability_clone_u_w_1)
                    loss_drop_s2 = softmax_mse_loss(inputs=Pred_drop_s2, targets=probability_clone_u_w_2)

                    loss_drop = (loss_drop_s1 + loss_drop_s2)/2.0


            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x) # 有标签的损失

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0    + cfg['feat_ali_weight'] * (loss_feature_1 + loss_feature_2)/2.0

            loss = loss + cfg['feat_nearest'] * loss_prote
            
            if epoch>=cfg['warm_up']:
                if cfg['F-Noise'] and cfg['F-Drop']:
                    loss_pertu = weight_u * (loss_drop +  loss_noise)
                    
                elif cfg['F-Noise']:
                    loss_pertu =  weight_u * loss_noise
                    
                elif cfg['F-Drop']:
                    loss_pertu =  weight_u * loss_drop
                
                loss = loss + loss_pertu
        
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            


            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            loss_feature_ali_loss.update(cfg['feat_ali_weight'] * (loss_feature_1.item() + loss_feature_2.item())/2.0)
            loss_prote_all.update(cfg['feat_nearest'] * loss_prote.item())
            if epoch>=cfg['warm_up']:
                if cfg['F-Noise']:
                    total_loss_noise_loss.update(weight_u * loss_noise.item())
                if cfg['F-Drop']:
                    total_loss_drop.update(weight_u * loss_drop.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, loss_feat_align: {:.5f}, Loss w_noise: {:.14f}, Loss drop_loss: {:.14f}  prote_loss: '
                            '{:.6f}'.format(i, total_loss.avg, total_loss_x.avg,loss_feature_ali_loss.avg,total_loss_noise_loss.avg, 
                                            total_loss_drop.avg, loss_prote_all.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
