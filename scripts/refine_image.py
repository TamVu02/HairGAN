import os
import numpy as np
import torch
import face_alignment
from utils.image_utils import process_display_input
import torch.nn.functional as F
from tqdm import tqdm
from criteria.transfer_loss import TransferLossBuilder
from criteria import masked_lpips
from torchvision import transforms

class RefineProxy(torch.nn.Module):
    def __init__(self, opts, generator, seg):
        super(RefineProxy, self).__init__()
        self.opts = opts
        self.generator = generator
        self.seg = seg
        self.mask_loss = self.weighted_ce_loss()
        self.transfer_loss_builder = TransferLossBuilder()
        self.delta_loss = torch.nn.MSELoss()
        self.landmark_loss = torch.nn.MSELoss()
        self.percept_with_mask = masked_lpips.PerceptualLoss(
                model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=self.opts.device == 'cuda'
            )
        self.percept_with_mask.eval()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def weighted_ce_loss(self):
        weight_tmp = torch.zeros(16).cuda()
        weight_tmp[10] = 1
        weight_tmp[1] = 1
        weight_tmp[6] = 1
        weight_tmp[0] = 1
        return torch.nn.CrossEntropyLoss(weight=weight_tmp).cuda()

    def gen_256_img_hairmask(self, input_image): 
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_hairmask_256 = F.interpolate(input_hairmask.unsqueeze(0).float(), size=(256, 256))
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_hairmask_256
    
    def gen_256_img_facemask(self, input_image):
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_earmask = torch.where((input_seg == 6), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_bgmask = torch.where((input_seg == 0), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_clothes_mask = torch.where((input_seg == 15), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        hair_ear_mask = input_hairmask + input_earmask + input_bgmask + input_clothes_mask
        input_face_256 = 1 - (F.interpolate(hair_ear_mask.unsqueeze(0).float(), size=(256, 256)))
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_face_256

    def forward(self, blended_latent, src_image, ref_img, target_mask):
        image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        ref_img = image_transform(ref_img).unsqueeze(0).cuda()
        ref_img_256, ref_hairmask_256 = self.gen_256_img_hairmask(ref_img)
        _, source_hairmask_256 = self.gen_256_img_hairmask(src_image)
        source_img_256, source_facemask_256 = self.gen_256_img_facemask(src_image)
        blended_latent=blended_latent.requires_grad_(True)
        optimizer = torch.optim.Adam([blended_latent], lr=self.opts.lr_refine)

        visual_list = []
        visual_interval = self.opts.steps_refine // self.opts.visual_num_ref
        pbar = tqdm(range(self.opts.steps_refine))
        for i in pbar:
            optimizer.zero_grad()
            img_gen = self.generator.decoder.synthesis(blended_latent, noise_mode='const')
            img_gen_256 = F.interpolate(img_gen, size=(256, 256))
            #Hair loss
            hair_loss = self.percept_with_mask(img_gen_256, ref_img_256,mask=ref_hairmask_256)
            #Face loss
            no_hair_region = (1 - ref_hairmask_256) * (1 - source_hairmask_256)
            face_loss = self.percept_with_mask(img_gen_256, source_img_256,mask=no_hair_region)
            #mask loss
            ce_loss=0
            if target_mask is not None:
                gen_seg = self.seg(img_gen)[1]
                ce_loss = 1.0 * self.cross_entropy(gen_seg, target_mask)

            loss = hair_loss + face_loss + ce_loss
            
            loss.backward()
            optimizer.step()
            pbar.set_description((f"refine_loss: {loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.opts.steps_ref-1)):
                with torch.no_grad():
                    img_gen = self.generator.decoder.synthesis(blended_latent, noise_mode='const')
                    visual_list.append(process_display_input(img_gen))
        return img_gen, blended_latent, visual_list
