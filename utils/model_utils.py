import torch
from models.stylegan3.model import Generator
from models.face_parsing.model import BiSeNet
import pickle

def load_base_models():
    ckpt = "/content/gdrive/MyDrive/HairGAN/tam_proposed/stylegan3-t-ffhq-1024x1024.pkl"
    #g_ema = Generator(1024, 512, 8)
    with open(ckpt, 'rb') as f:
        g_ema = pickle.load(f)['G_ema'].cuda()
        mean_latent = pickle.load(f)['latent_avg'].unsqueeze(0).unsqueeze(0).repeat(1,18,1).clone().detach().cuda()

    #mean_latent = torch.load(ckpt)["latent_avg"].unsqueeze(0).unsqueeze(0).repeat(1,18,1).clone().detach().cuda()

    seg_pretrained_path = "/content/gdrive/MyDrive/HairGAN/tam_proposed/seg.pth"
    seg = BiSeNet(n_classes=16)
    seg.load_state_dict(torch.load(seg_pretrained_path), strict=False)
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()
    seg = seg.cuda()

    return g_ema, mean_latent, seg
