import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
from Brainmets.utils import *

class MetDataSet(Dataset):
    def __init__(self, df, target='mask', transformer = None, resample = None):
        if resample == 'max_size':
            ro = RandomOverSampler()
            df['max_size_class'] = df['max_met_size']>100
            X_resampled, y_resampled = ro.fit_resample(df.drop(columns=['max_size_class']),df['max_size_class'])
            df = pd.concat([X_resampled, y_resampled], axis = 1)
        self.target = target
        self.img_files = list(df['img_files'])
        self.mask_files = list(df['mask_files'])
        self.met_nums = list(df['met_num'])
        self.img_names = ['_'.join(file.split('/')[-1].split('_')[0:2]) for file in self.img_files]
        self.mask_names = ['_'.join(file.split('/')[-1].split('_')[0:2]) for file in self.mask_files]
        self.transformer = transformer
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        if self.target == 'mask':
            img = read_and_crop(self.img_files[idx],64,256,256).reshape(1,64,256,256)
            mask = read_and_crop(self.mask_files[idx],64,256,256).reshape(1,64,256,256)
            if self.transformer:
                img, mask =self.transformer.transform(img[0], mask[0])
                return img.reshape(1,64,256,256).copy(), mask.reshape(1,64,256,256).copy()
            else:
                return img.copy(), mask.copy()
        if self.target == 'met_num':
            img = read_and_crop(self.img_files[idx],64,256,256).reshape(1,64,256,256)
#             if self.transformer:
#                 img, mask =self.transformer.transform(img[0], mask[0])
#                 return img.reshape(1,64,256,256).copy(), mask.reshape(1,64,256,256).copy()
#             else:
            return torch.tensor(img.copy()).float(), torch.tensor(self.met_nums[idx]).float().view(1)
    def get_name(self,idx):
        img_name = self.img_names[idx]
        mask_name = self.mask_names[idx]
        return img_name,mask_name