from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import time

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

    

class TrainMVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, ROTATE_FLAG = False):
        self.rotate = ROTATE_FLAG
        self.root_path = root
        self.transform = transform
        self.imgdic = {}
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        
    def load_dataset(self):
        print('load images...')
        ss = time.time()
        img_paths = glob.glob(os.path.join(self.root_path, 'good') + "/*.png")
        counter = 0
        for i,img_p in enumerate(img_paths):
            img = Image.open(img_p).convert('RGB')
            if self.rotate:
                for r in [0, 90, 180, 270]:
                    imgr = img.rotate(r)
                    imgtrans = self.transform(imgr)            
                    self.imgdic[counter] = imgtrans
                    counter += 1
            
            img = self.transform(img)            
            self.imgdic[i] = img
        print('read {} images time used:'.format(len(img_paths)), time.time() - ss)
        return img_paths

    def __len__(self):
        if self.rotate:
            return len(self.img_paths) * 4
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        
        return self.imgdic[idx]

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        if 'MVTec3D' in root:
            self.gt_path = os.path.join(root, 'test')
        else:
            self.gt_path = os.path.join(root, 'ground_truth')
            
        self.root_path = root
        
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        
        
        ss = time.time()
        self.img_dic = {}
        for i, p in enumerate(self.img_paths):
            img = Image.open(p).convert('RGB')
            img = self.transform(img)
            self.img_dic[i] = img
        
        
        self.gt_dic = {}
        for i, gtp in enumerate(self.gt_paths):
            if gtp == 0:
                gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                gt = Image.open(gtp).convert("L")
                gt = self.gt_transform(gt)
            self.gt_dic[i] = gt
            
        print('load test images time used:', time.time() - ss)
            
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                if 'MVTec3D' in self.root_path:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                if 'MVTec3D' in self.root_path:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type, 'gt') + "/*.png")
                else:
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label, img_type = self.labels[idx], self.types[idx]

        img = self.img_dic[idx]

        gt = self.gt_dic[idx]

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type



