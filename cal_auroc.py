
import json

category_lst = ['bottle', 'cable', 'capsule', 'carpet', 
                'grid', 'hazelnut', 'leather', 'metal_nut', 
                'pill', 'screw', 'tile', 'toothbrush', 
                'transistor', 'wood', 'zipper']
texture_category_lst = ['leather', 'wood', 'carpet', 'tile', 'grid']

total_im_auc = 0
total_px_auc = 0
total_pro = 0


texture_total_pixel_auc_score = 0
texture_total_image_auc_score = 0
texture_total_pro_score = 0

object_total_pixel_auc_score = 0
object_total_image_auc_score = 0
object_total_pro_score = 0

for category in category_lst:
    with open('./log/score/{}.json'.format(category),'r') as f:
        dic = json.loads(f.read())
        
        
    for i,key in enumerate(dic):
        
        max_px_auc = dic[key]['max_px_auc']
        max_im_auc = dic[key]['max_im_auc']
        max_px_pro = dic[key]['max_px_pro']
        
        
        print( key, 'im px pro:', max_im_auc, max_px_auc, max_px_pro )
        total_im_auc += max_im_auc
        total_px_auc += max_px_auc
        total_pro += max_px_pro
        
        if category in texture_category_lst:
            texture_total_pixel_auc_score += max_px_auc
            texture_total_image_auc_score += max_im_auc
            texture_total_pro_score += max_px_pro
        else:
            object_total_pixel_auc_score += max_px_auc
            object_total_image_auc_score += max_im_auc
            object_total_pro_score += max_px_pro

print('\nall')
print('image avg:', round(total_im_auc / 15, 4))
print('pixel avg:', round(total_px_auc / 15, 4))
print('pro avg:', round(total_pro / 15, 4))

print('\ntexture')
print('image level avg auc:', round(texture_total_image_auc_score / 5, 3))
print('pixel level avg auc:', round(texture_total_pixel_auc_score / 5, 3))
print('avg pro:', round(texture_total_pro_score / 5, 3))
            
print('\nobject')
print('image level avg auc:', round(object_total_image_auc_score / 10, 3))
print('pixel level avg auc:', round(object_total_pixel_auc_score / 10, 3))
print('avg pro:', round(object_total_pro_score / 10, 3))





    





