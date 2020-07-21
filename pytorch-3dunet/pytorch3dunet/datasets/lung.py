import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from monai import transforms as mn_tf
from tqdm import tqdm
import os
from torchvision import transforms as tf
import torch.utils.data as Data
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


class LungDataset(Dataset):
    def __init__(self, root_dir, transforms, train=True):
        self.root_dir = root_dir

        if train:
            self.image_folder = os.path.join(root_dir, 'train_nii_1')
            self.mask_folder = os.path.join(root_dir, 'train_mask_nii')
        else:
            self.image_folder = os.path.join(root_dir, 'test_nii_1')
            self.mask_folder = os.path.join(root_dir, 'test_mask_nii')

        self.transforms = transforms
        self.target_transforms = tf.ToTensor()
        self.train = train
        self.data, self.targets = self.get_data_targets()

    def get_data_targets(self):
        cases = os.listdir(self.image_folder)
        data = []
        targets = []
        for case in cases:
            data.append(os.path.join(self.image_folder, case))
            targets.append(os.path.join(self.mask_folder, case))
        return data, targets

    def __getitem__(self, idx):
        case_path = self.data[idx]
        label_path = self.targets[idx]
        if self.transforms is not None:
            data_dict = self.transforms({'image': case_path, 'label': label_path})
        else:
            raise RuntimeError('need transform to read .nii file')
        return [data_dict['image'].transpose(0, 1), data_dict['label'].transpose(0, 1)]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    keys = ('image', 'label')
    mn_tfs = mn_tf.Compose([
        mn_tf.LoadNiftiD(keys),
        # mn_tf.AsChannelFirstD('image'),
        mn_tf.AddChannelD(keys),
        # mn_tf.SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
        mn_tf.OrientationD(keys, axcodes='RAS'),
        mn_tf.ThresholdIntensityD('image', threshold=600, above=False, cval=600),
        mn_tf.ThresholdIntensityD('image', threshold=-1000, above=True, cval=-1000),
        # mn_tf.ScaleIntensityD('image', minv=0.0, maxv=225.0), # show image
        mn_tf.ScaleIntensityD('image'),
        mn_tf.ResizeD(keys, (512, 512, 128), mode=('trilinear', 'nearest')),
        mn_tf.AsChannelFirstD(keys),
        mn_tf.RandAffineD(keys, spatial_size=(-1, -1, -1),
                          rotate_range=(0, 0, np.pi / 2),
                          scale_range=(0.1, 0.1),
                          mode=('bilinear', 'nearest'),
                          prob=1.0),
        mn_tf.ToTensorD(keys)

    ])
    dataset = LungDataset(root_dir='../../../../dataset/LCTSC_1', transforms=mn_tfs, train=False)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=4)
    # for item in dataset:
    #     print(item['image'].shape, item['label'].shape)
    #     break
    case = dataset[0]
    # for item in dataloader:
    #     case = item
    #     break

    # slices = case['image']
    # masks = case['label']
    slices = case[0]
    masks = case[1]
    print(slices.shape, masks.shape)
    for idx, item in enumerate(zip(slices, masks)):
        image = item[0][0]
        label = item[1][0] * 255
        if idx >= 40 and idx <= 50:
            print(image.min(), image.max())
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.close()
            plt.imshow(label, cmap='gray')
            plt.show()
            plt.close()
