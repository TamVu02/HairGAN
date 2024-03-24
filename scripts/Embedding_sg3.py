import torch
from torch import nn
import numpy as np
import os
from utils.bicubic import BicubicDownSample
from criteria.embedding_loss import EmbeddingLossBuilder
from tqdm import tqdm
import PIL
from torchvision import transforms
from skimage import io
import cv2
from PIL import Image
import torch.nn.functional as F
import os
from utils.common import convert_npy_code
from utils.inference_utils import get_average_image


class Embedding_sg3(nn.Module):
    def __init__(self, opts, generator, mean_latent_code):
        super(Embedding_sg3, self).__init__()
        self.opts = opts
        self.generator = generator
        self.mean_latent_code = mean_latent_code
        self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.load_PCA_model()
        self.load_downsampling()
        self.setup_embedding_loss_builder()

    def load_PCA_model(self):
        PCA_path = self.opts.ffhq_pca_path
        if not os.path.isfile(PCA_path):
            print("Can not find the PCA_PATH for FFHQ!")

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().cuda()
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().cuda()
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().cuda()


    def load_downsampling(self):
        factor = 1024 // 256
        self.downsample = BicubicDownSample(factor=factor, cuda=True)


    def setup_FS_optimizer(self, latent_W, F_init):
        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        for i in range(16):
            tmp = latent_W[0, i].clone()
            if i < 5:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True
            latent_S.append(tmp)
        optimizer_FS = torch.optim.Adam(latent_S[5:] + [latent_F], lr=self.opts.lr_embedding)
        return optimizer_FS, latent_F, latent_S

    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)

    def invert_image_in_W(self, image_path=None, latent_dir=None,device=None):
        im_name = os.path.splitext(os.path.basename(image_path))[0]
        latent_W_path = os.path.join(latent_dir, f'{im_name}.npy')
        latent_in=None
        if not os.path.isfile(latent_W_path):
            ref_im = Image.open(image_path).convert('RGB')
            ref_im_L = self.image_transform(ref_im.resize((256, 256), PIL.Image.LANCZOS)).unsqueeze(0).to('cuda')
            gen_im,latent=None,None
            pbar = tqdm(range(self.opts.SG3_steps), desc='Embedding', leave=False)
            for step in pbar:
                if(step==0):
                    avg_image = get_average_image(self.generator)
                    avg_image = avg_image.unsqueeze(0).repeat(ref_im_L.shape[0], 1, 1, 1)
                    x_input = torch.cat([ref_im_L, avg_image], dim=1)
                else:
                    gen_im = self.generator.face_pool(gen_im)
                    x_input = torch.cat([ref_im_L, gen_im], dim=1)
                gen_im,latent = self.generator(x_input,latent=latent, return_latents=True, resize=False)
            latent_in=latent
        else:
            latent_in = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
        return latent_in

    def invert_image_in_FS(self, image_path=None):
        ref_im = Image.open(image_path).convert('RGB')
        ref_im_L = self.image_transform(ref_im.resize((256, 256), PIL.Image.LANCZOS)).unsqueeze(0)#.to('cuda')
        ref_im_H = self.image_transform(ref_im.resize((1024, 1024), PIL.Image.LANCZOS)).unsqueeze(0)

        latent_W = self.invert_image_in_W(image_path=image_path, latent_dir=self.opts.latents_path, device='cuda').clone().detach()
        F_init = self.generator.decoder.synthesis(latent_W, noise_mode='const')
        optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)

        gen_im,latent=None,None

        pbar = tqdm(range(self.opts.FS_steps), desc='Embedding', leave=False)
        for step in pbar:
            optimizer_FS.zero_grad()
            latent_in = torch.stack(latent_S).unsqueeze(0)
            # if(step==0):
            #     avg_image = self.get_avg_img(self.generator)
            #     avg_image = avg_image.unsqueeze(0).repeat(ref_im_L.shape[0], 1, 1, 1)
            #     x_input = torch.cat([ref_im_L, avg_image], dim=1)
            # else:
            #     x_input = torch.cat([ref_im_L, gen_im], dim=1)
            # gen_im,latent = self.generator(x_input,latent=latent_in, return_latents=True, resize=False)
            gen_im = self.generator.decoder.synthesis(latent_in, noise_mode='const')
            im_dict = {
                'ref_im_H': ref_im_H.cuda(),
                'ref_im_L': ref_im_L.cuda(),
                'gen_im_H': gen_im,
                'gen_im_L': self.downsample(gen_im)
            }

            loss, loss_dic = self.cal_loss(im_dict, latent_in)
            gen_im = self.generator.face_pool(gen_im).detach().clone()
            loss.backward()
            optimizer_FS.step()
            pbar.set_description(
                'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))
        return latent_in, latent_F


    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda_embedding * (latent_p_norm.pow(2).mean())
        return p_norm_loss

    def cal_l_F(self, latent_F, F_init):
        return self.opts.l_F_lambda_embedding * (latent_F - F_init).pow(2).mean()

    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic


