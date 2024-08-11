import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils_tq import read_pt_image as read_image
from .utils_tq import load_list, get_modal_dir

import random
from tqdm import tqdm

     
def transfer_log1p(input_array):
    if isinstance(input_array,np.ndarray):
        return np.sign(input_array)*np.log1p(np.abs(input_array))
    elif isinstance(input_array,torch.Tensor):
        return torch.sign(input_array)*torch.log1p(torch.abs(input_array))
    else:
        raise ValueError('input_array should be numpy array or torch tensor')


def image_preprocess(image_list, image_size = 224, p_flip = 0.5, p_rotate = 90, seed = 1):
    N = len(image_list)
    channels = np.zeros(N, dtype=int)
    for i in range(N):
        image = image_list[i]
        if isinstance(image, np.ndarray):
            image = np.nan_to_num(image, nan=0.0)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size),
            ])
            image = transform(image)
            image_list[i] = image
        elif isinstance(image, torch.Tensor):
            image = image.unsqueeze(0)
            transform = transforms.Compose([
                transforms.Resize(image_size),
            ])
            image = transform(image)
            image_list[i] = image
        else:
            raise ValueError('image should be numpy array or torch tensor')
        channels[i]=image.shape[0]
    
    image = torch.cat(image_list, dim=0)
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p_flip),
            transforms.RandomVerticalFlip(p_flip),
            transforms.RandomRotation(p_rotate),
        ]) 
    image = transform(image)

    for i in range(N):
        image_list[i] = image[channels[:i].sum():channels[:i+1].sum()]

    image = torch.stack(image_list, dim=0)
    return image


def enhance_funciton(image, enhance_type = 'log1p', rescale_value = 1):
    
    if enhance_type == 'log1p':
        image = transfer_log1p(image)
    elif enhance_type == 'None':
        pass
    else:
        raise ValueError('enhance_type should be log1p or None')
    
    image = image*rescale_value
    return image


class Dataset_one_modal(Dataset):
    def __init__(self, modal, exist_idx: np.ndarray, load_imgs = False):

        self.modal = modal
        self.exist_idx = exist_idx
        self.load_imgs = load_imgs

    def __len__(self):
        return len(self.exist_idx)

    def load_images(self, idx_list):
        self.image_list = {}
        for idx in tqdm(idx_list, desc=f'Loading {self.modal} images'):
            idx = int(idx)
            path = get_modal_dir(self.modal, idx)[1]
            img = read_image(path)
            self.image_list[idx] = img
    
    def __getitem__(self, idx):
        if self.load_imgs:
            idx = int(idx)
            img = self.image_list[idx]
            is_exist = True
        else:
            idx = int(idx)
            img_path = get_modal_dir(self.modal, idx)[1]
            is_exist = self.exist_idx[idx]
        return img_path, is_exist

     
class multimodal_dataset(Dataset):
    def __init__(self, modal_list = ['magnet','0094'], load_imgs = False, enhance_list = [224,0.5,90], time_interval = [0,7452000],time_step = 1): #time_step = 1 means 1 min
        # 定义数据集
        self.dataset = [] 
        time_slice = slice(time_interval[0],time_interval[1],time_step)
        if 'magnet' in modal_list:
            # mag_dir_list = load_list('./Data/dir_list/magnet_dir_list_pt.pkl')[time_slice]
            mag_idx = load_list('./Data/idx_list/magnet_exist_idx.pkl')
            self.dataset.append(Dataset_one_modal('magnet',mag_idx))

        if '0094' in modal_list:
            # h0094_dir_list = load_list('./Data/dir_list/0094_dir_list_pt.pkl')[time_slice]
            h0094_idx = load_list('./Data/idx_list/0094_exist_idx.pkl')
            self.dataset.append(Dataset_one_modal('0094',h0094_idx))

        # find the all exist index
        exist_list = []
        for i in range(len(self.dataset)):
            dataset = self.dataset[i]
            exist_list.append(dataset.exist_idx)
            print(f' {modal_list[i]} has {np.sum(exist_list[i])} samples')

        # first test, if all modal has the same exist data
        exist_list = np.array(exist_list)
        self.exist_idx = np.all(exist_list,axis=0)
        self.exist_idx = np.nonzero(self.exist_idx)[0] # get the index of all exist data
        # second test, if the idx follows the time interval
        self.exist_idx = self.exist_idx[self.exist_idx>=time_interval[0]]
        self.exist_idx = self.exist_idx[self.exist_idx<time_interval[1]]
        self.exist_idx = self.exist_idx[self.exist_idx%time_step==0]

        self.enhance_list = enhance_list
        print(f'All modal has {len(self.exist_idx)} samples')

        if load_imgs:
            for i in range(len(self.dataset)):
                self.dataset[i].load_imgs = True
                self.dataset[i].load_images(self.exist_idx)

    def __len__(self):
        return len(self.exist_idx)
    
    def __getitem__(self, idx):
        image_list = []
        for i in range(len(self.dataset)):
            if self.dataset[i].load_imgs:
                img, _ = self.dataset[i][self.exist_idx[idx]]
            else:
                path,_ = self.dataset[i][self.exist_idx[idx]]
                img = read_image(path)

            image_list.append(img)
        image = image_preprocess(image_list, image_size = self.enhance_list[0], p_flip = self.enhance_list[1], p_rotate = self.enhance_list[2])

        return image


def SolarDataLoader(modal_list = ['magnet','0094'], load_imgs = False, enhance_list = [224,0.5,90], batch_size = 32, shuffle=False, num_workers=4):
    dataset = multimodal_dataset(modal_list, load_imgs, enhance_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # return [Batchsize, modal_num, channel, height, width]
    return dataloader




if __name__ == '__main__':
    # a = np.array([1,2,3,4])
    
    dataloader = SolarDataLoader(modal_list = ['magnet','0094'], enhance_list = [['log1p',224,1],['log1p',224,1]], batch_size = 32, shuffle=False, num_workers=1)
    print(len(dataloader))