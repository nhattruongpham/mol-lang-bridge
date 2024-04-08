
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def inverse_images(norm_images):
    '''
    Convert image from normalized images to RGB form
    '''
    inverse_transform = T.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
        std=[1 / s for s in IMAGENET_DEFAULT_STD]
    )
    
    unnorm_images = inverse_transform(norm_images)
    
    return unnorm_images * 255.
    