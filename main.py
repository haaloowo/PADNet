
import torch
import numpy as np
import os
import time
import json
from PIL import Image
import glob
import torch.nn as nn


from dataset import get_data_transforms
from adtest import evaluation
from feat_decoder import DecoderConcat
from resnet import resnet18, resnet34, resnet50, resnet101, wide_resnet50_2, wide_resnet101_2
from dataset import MVTecDataset, TrainMVTecDataset
from utils import setup_seed, loss_fucntion, gen_mask

import warnings
warnings.filterwarnings("ignore")


class MemoryBank():
    
    def __init__(self,
                 root_path, 
                 image_size, 
                 total_img_num, 
                 device, 
                 channel_dim_lst,
                 encoder, 
                 transform,
                 knn_k
                 ):
        self.root_path = root_path
        self.device = device
        self.transform = transform
        self.knn_k = knn_k
        self.total_img_num = total_img_num
        
        self.relu = nn.ReLU()

        self.img_paths = glob.glob(os.path.join(self.root_path, 'good') + "/*.png")
        self.img_num = len(self.img_paths)
        
        s1 = int(image_size/4)
        s2 = int(image_size/8)
        s3 = int(image_size/16)
        self.feat_bank0 = torch.zeros(self.img_num, channel_dim_lst[0], s1, s1).to(device)
        self.feat_bank1 = torch.zeros(self.img_num, channel_dim_lst[1], s2, s2).to(device)
        self.feat_bank2 = torch.zeros(self.img_num, channel_dim_lst[2], s3, s3).to(device)
        
        if self.total_img_num and (self.img_num > self.total_img_num):
            self.avg_pooled_feat = torch.zeros(self.total_img_num, channel_dim_lst[2]).to(device)
        else:
            self.avg_pooled_feat = torch.zeros(self.img_num, channel_dim_lst[2]).to(device)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.cos_sim_func = torch.nn.CosineSimilarity(dim=1)
        
        self.add2bank(encoder)

    def add2bank(self, encoder):
        print('add2bank')
        if self.total_img_num and (self.img_num > self.total_img_num):
            load_path_lst = self.img_paths[:self.total_img_num]
        else:
            load_path_lst = self.img_paths
            
        for i,img_p in enumerate(load_path_lst):
            img = Image.open(img_p).convert('RGB')
            img = self.transform(img)
            img = img.unsqueeze(dim=0).to(self.device)
            with torch.no_grad():
                inputs = encoder(img)
            x1,x2,x3 = inputs[0], inputs[1], inputs[2]
            self.feat_bank0[i] = x1.squeeze()
            self.feat_bank1[i] = x2.squeeze()
            self.feat_bank2[i] = x3.squeeze()
            
            avg_feat = self.global_avg_pool(x3).squeeze()
            self.avg_pooled_feat[i] = avg_feat
    
    def get_ref(self, query_tensor):
        if self.total_img_num and (self.img_num > self.total_img_num):
            distances = self.cos_sim_func(self.avg_pooled_feat, query_tensor.unsqueeze(dim=0).repeat(self.total_img_num,1))
        else:
            distances = self.cos_sim_func(self.avg_pooled_feat, query_tensor.unsqueeze(dim=0).repeat(self.img_num,1))
            
        top_k_indices = distances.argsort()[-self.knn_k:]
        
        top_k_distances = distances[top_k_indices]
        top_k_weight = top_k_distances.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        top_k_tensors1 = self.feat_bank0[top_k_indices]
        top_k_tensors2 = self.feat_bank1[top_k_indices]
        top_k_tensors3 = self.feat_bank2[top_k_indices]

        return top_k_tensors1, top_k_tensors2, top_k_tensors3, top_k_weight
        
    def cal_diff(self, inputs):
        batchx0 = inputs[0]
        batchx1 = inputs[1]
        batchx2 = inputs[2]
        B,C0,H0,W0 = batchx0.shape
        B,C1,H1,W1 = batchx1.shape
        _,C2,H2,W2 = batchx2.shape
        
        self.batchx0_diff = torch.zeros(B,C0,H0,W0).to(self.device)
        self.batchx1_diff = torch.zeros(B,C1,H1,W1).to(self.device)
        self.batchx2_diff = torch.zeros(B,C2,H2,W2).to(self.device)
        
        for i in range(B):
            x0 = batchx0[i]
            x1 = batchx1[i]
            x2 = batchx2[i]
            
            query_feat = self.global_avg_pool(x2).squeeze()
            top_k_tensors1, top_k_tensors2, top_k_tensors3, top_k_weight = self.get_ref(query_feat)

            x0_diff = torch.abs(x0 - top_k_tensors1.mean(dim=0, keepdim=True))
            x1_diff = torch.abs(x1 - top_k_tensors2.mean(dim=0, keepdim=True))
            x2_diff = torch.abs(x2 - top_k_tensors3.mean(dim=0, keepdim=True))
                
            self.batchx0_diff[i] = x0_diff * 2.0
            self.batchx1_diff[i] = x1_diff * 2.0
            self.batchx2_diff[i] = x2_diff * 2.0

        return self.batchx0_diff, self.batchx1_diff, self.batchx2_diff


    

def train(_class_, channel_dim_lst, epochs):
    
    max_px_auc = 0.0
    max_im_auc = 0.0
    max_px_pro = 0.0
    max_auc_sum = 0.0
    print(_class_)
    
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(DATASET_PATH, _class_,  'train')
    print('train_path:', train_path)
    test_path = os.path.join(DATASET_PATH,  _class_)
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    
    # encoder, bn = resnet18(pretrained=True)
    # encoder, bn = resnet50(pretrained=True)
    # encoder, bn = resnet101(pretrained=True)
    encoder, bn = wide_resnet50_2(pretrained=True)
    # encoder, bn = wide_resnet101_2(pretrained=True)
    
    encoder = encoder.to(device)
    encoder.eval()
    
    membank = MemoryBank(train_path,
                         image_size, 
                         TOTAL_MEMORY_NUM,
                         device,
                         channel_dim_lst,
                         encoder,
                         data_transform, 
                         knn_k = KNN_K
                         )
    
    train_data = TrainMVTecDataset(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    if USE_MEM_DIFF:
        channel_dim_lst = [i*2 for i in channel_dim_lst]
    
    decoder = DecoderConcat(device, channel_dim_lst)
    decoder = decoder.to(device)
    
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=learning_rate, betas=(0.5, 0.999) , weight_decay=1e-6)
    
    avgpool = torch.nn.AvgPool2d(3, 1, 1) 
    
    for epoch in range(epochs):
        decoder.train()
        loss_list = []
        for img in train_dataloader:
            img = img.to(device)

            with torch.no_grad():
                inputs = encoder(img)

            if USE_MEM_DIFF:
                diff_inputs = membank.cal_diff(inputs)
                inputs[0] = torch.cat([inputs[0], diff_inputs[0]], dim=1)
                inputs[1] = torch.cat([inputs[1], diff_inputs[1]], dim=1)
                inputs[2] = torch.cat([inputs[2], diff_inputs[2]], dim=1)
                
            _inputs_ = [i.clone() for i in inputs]
            
            for _i in range(3):
                _inputs_[_i] = avgpool(_inputs_[_i])
            
            outputs = decoder(_inputs_)
    
            loss = loss_fucntion(inputs[used_layer_idx_start:used_layer_idx_end], 
                                 outputs[used_layer_idx_start:used_layer_idx_end]) 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append( loss.item() )
            
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        
        if (epoch + 1) % EVAL_INTERVAL == 0:
            eval_time_s = time.time()
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, 
                                                      decoder, 
                                                      test_dataloader, 
                                                      device,
                                                      used_layer_idx_start, 
                                                      used_layer_idx_end,
                                                      USE_MEM_DIFF,
                                                      membank,
                                                      _class_
                                                      )
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px), 
                  'time used:', round(time.time() - eval_time_s, 3)
                  )
            
            auc_sum = auroc_px + auroc_sp
            if auc_sum > max_auc_sum:
                max_px_auc = auroc_px
                max_im_auc = auroc_sp
                max_auc_sum = auc_sum
                max_px_pro = aupro_px
                
                torch.save({'decoder': decoder.state_dict()}, ckp_path)
            
            
    print('------------------')
    print('MAX Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(max_px_auc, max_im_auc, max_px_pro))
    
    score_dic = {}
    score_dic[_class_] = {
                          'max_px_auc':max_px_auc,
                          'max_im_auc':max_im_auc,
                          'max_px_pro':max_px_pro
                          }
    
    with open('./log/score/{}.json'.format(_class_),'w') as f:
        f.write(json.dumps(score_dic))
        
    return decoder


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ANOMALDETECTION')
    parser.add_argument('--category', default='pill')
    args = parser.parse_args()
    
    CATEGORY = args.category
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('./log/score', exist_ok=True)
    os.makedirs('./log/output', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    batch_size = 16
    image_size = 256
    
    used_layer_idx_start = 0
    used_layer_idx_end = 3
    
    USE_MEM_DIFF = True
    TOTAL_MEMORY_NUM = 0  # Size of normal samples in memory bank; stores all normal samples when set to 0
    KNN_K = 6
    
    learning_rate = 1e-3
    EVAL_INTERVAL = 2
    setup_seed(111)
    
    # channel_dim_lst = [64, 128, 256]   ### resnet18 34
    channel_dim_lst = [256, 512, 1024]   ### resnet50 101
    
    DATASET_PATH = '/media/yf/CODE/Dataset/MVTec'
    item_list = ['screw', 'cable', 'capsule',  'transistor', 'grid',
                  'carpet', 'pill', 'zipper', 'toothbrush', 'bottle',
                  'leather', 'metal_nut', 'hazelnut', 'tile', 'wood']
    
    ep_lst = ['leather', 'wood', 'carpet', 'tile', 'bottle', 'hazelnut', 'metal_nut']
    
    if CATEGORY in ep_lst:
        ep = 40
    else:
        ep = 100
    
    model = train(CATEGORY, channel_dim_lst, ep)

