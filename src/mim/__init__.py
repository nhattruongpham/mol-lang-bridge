from .mim_dataset import SimMIMTransform, collate_fn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def build_loader_simmim(args):
    transform = SimMIMTransform(args)

    dataset = ImageFolder(args.data_path, transform)
    
    dataloader = DataLoader(dataset, 
                            args.batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True, 
                            drop_last=True, 
                            collate_fn=collate_fn)
    
    return dataloader