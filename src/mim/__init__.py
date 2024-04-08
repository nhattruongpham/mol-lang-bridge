from .mim_dataset import SimMIMTransform, collate_fn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def build_loader_simmim(args):
    transform = SimMIMTransform(args)

    train_dataset = ImageFolder(args.train_data_path, transform)
    valid_dataset = ImageFolder(args.valid_data_path, transform)
    
    train_dataloader = DataLoader(train_dataset, 
                            args.batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True, 
                            drop_last=True, 
                            collate_fn=collate_fn)
    
    val_dataloader = DataLoader(valid_dataset, 
                            args.batch_size, 
                            num_workers=args.num_workers,
                            pin_memory=True, 
                            drop_last=True, 
                            collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader