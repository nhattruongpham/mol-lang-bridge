from .vision.mim_module import SimMIM, SwinTransformerForSimMIM, VisionTransformerForSimMIM

def build_mim_model(args):
    if args.model_name == 'swin':
        encoder = SwinTransformerForSimMIM(**vars(args))
        encoder_stride = 32
    elif args.model_name == 'vit':
        encoder = VisionTransformerForSimMIM(**vars(args))
        encoder_stride = 16
    return SimMIM(encoder=encoder, encoder_stride=encoder_stride)