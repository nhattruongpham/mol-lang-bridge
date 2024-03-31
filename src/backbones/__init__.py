from .vision.swin_mim import SimMIM, SwinTransformerForSimMIM

def build_mim_model(args):
    encoder = SwinTransformerForSimMIM(**vars(args))
    mim_model = SimMIM(encoder=encoder, encoder_stride=32)
    return mim_model