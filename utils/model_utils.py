import torch
from models.stylegan2.model import Generator
from models.face_parsing.model import BiSeNet
from utils.inference_utils import load_encoder
from utils.inference_utils import get_average_image

def load_base_models(opts):
    ckpt = opts.stylegan_path
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    mean_latent = torch.load(ckpt)["latent_avg"].unsqueeze(0).unsqueeze(0).repeat(1,18,1).clone().detach().cuda()

    seg_pretrained_path = opts.seg_path
    seg = BiSeNet(n_classes=16)
    seg.load_state_dict(torch.load(seg_pretrained_path), strict=False)
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()
    seg = seg.cuda()

    return g_ema, mean_latent, seg

def load_sg3_models(opts):
    generator,opts_sg3 = load_encoder(checkpoint_path=opts.stylegan3_weights,generator_path=opts.generator_path3)

    seg_pretrained_path = opts.seg_path
    seg = BiSeNet(n_classes=16)
    seg.load_state_dict(torch.load(seg_pretrained_path), strict=False)
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()
    seg = seg.cuda()

    avg_image = get_average_image(generator)

    return generator, opts_sg3, seg, avg_image
