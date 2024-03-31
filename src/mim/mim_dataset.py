import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data._utils.collate import default_collate
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def _get_img_weights(self, input_image):
        C, H, W = input_image.shape
        
        WEIGHTS = np.zeros((H // self.mask_patch_size, W // self.mask_patch_size))
        for y in range(H // self.mask_patch_size):
            for x in range(W // self.mask_patch_size):
                patch_left = x * self.mask_patch_size
                patch_upper = y * self.mask_patch_size
                patch_right = patch_left + self.mask_patch_size
                patch_lower = patch_upper + self.mask_patch_size
                
                patch = T.functional.crop(input_image, patch_upper, patch_left, patch_lower-patch_upper, patch_right-patch_left)
                
                mean_value = torch.mean(patch)
                
                WEIGHTS[y,x] = mean_value
        
        # WEIGHTS = WEIGHTS
        
        # min-max scaling
        WEIGHTS = (-WEIGHTS - np.min(-WEIGHTS)) / (np.max(-WEIGHTS) - np.min(-WEIGHTS)) + 0.3
        
        return WEIGHTS / np.sum(WEIGHTS)
        
    def __call__(self, input_image):
        print(self._get_img_weights(input_image).flatten())
        mask_idx = np.random.choice(
            self.token_count, size=self.mask_count, p=self._get_img_weights(input_image).flatten(), replace=False
        )
        # mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, args):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(args.img_size),
            # T.RandomHorizontalFlip(), ### Remove this caus molecule information is not symmetric
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        model_patch_size = args.patch_size
        
        self.mask_generator = MaskGenerator(
            input_size=args.img_size,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=args.mask_ratio
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator(img)
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret