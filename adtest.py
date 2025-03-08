import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
import os

def cal_anomaly_map(fs_list, ft_list, out_size=256, ):
    anomaly_map = np.zeros([out_size, out_size])
    for i in range(len(fs_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        
        ### cosine sim
        a_map = 1 - F.cosine_similarity(fs, ft)
        
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()

        anomaly_map += a_map
    return anomaly_map

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def evaluation(encoder, 
               decoder, 
               dataloader,
               device,
               layer_start_idx, 
               layer_end_idx,
               USE_MEM_DIFF, 
               membank,
               _class_
               ):
    
    decoder.eval()
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    
    avgpool = torch.nn.AvgPool2d(3, 1, 1) 
    
    img_save_base_path = os.path.join('./log/output', _class_)
    if not os.path.exists(img_save_base_path):
        os.mkdir(img_save_base_path)
    
    img_level_ano_score_dic = {}
    img_level_ano_score_dic[0] = []
    img_level_ano_score_dic[1] = []
    
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

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
            
            anomaly_map = cal_anomaly_map(  inputs[layer_start_idx:layer_end_idx], 
                                            outputs[layer_start_idx:layer_end_idx], 
                                            img.shape[-1],
                                            )
            anomaly_map = gaussian_filter(anomaly_map, sigma=4) # 256*256
            
            image_level_ano_score = np.max(anomaly_map).astype(np.float64)
            if label.item() == 1: 
                img_level_ano_score_dic[1].append(image_level_ano_score)
            else:
                img_level_ano_score_dic[0].append(image_level_ano_score)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if gt.max()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
                
                # aupro_list.append(0)
                
            gt = gt[:, 0, :, :]
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
            
            
            ano_map_norm = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map_norm*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            ano_map = show_cam_on_image(img, ano_map)
            gt = gt.cpu().numpy().astype(int)[0]*255
            
            
            ano_map = draw_closed_boundary(ano_map, ano_map_norm)
            
            cv2.imwrite(os.path.join(img_save_base_path,
                                          '{}_a{}.png'.format(count,'img')), img)
            cv2.imwrite(os.path.join(img_save_base_path,
                                      '{}_c{}.png'.format(count,'gt')), gt)
            cv2.imwrite(os.path.join(img_save_base_path,
                                      '{}_b{}.png'.format(count,'map')), ano_map)
            count += 1
            
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)


def draw_closed_boundary(image, normalized_heatmap, threshold=0.7, area_thres=5):  
    _, binary_mask = cv2.threshold(normalized_heatmap, threshold, 1, cv2.THRESH_BINARY)  

    binary_mask = (binary_mask * 255).astype(np.uint8)  
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    for contour in contours:  
        area = cv2.contourArea(contour)  
        if area > area_thres:  
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # 画出红色闭合轮廓  
    return image

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    # print(amaps.shape, masks.shape)
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    
    # print(set(masks.flatten()))

    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, pd.DataFrame({"pro": [mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
