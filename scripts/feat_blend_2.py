import torch
from utils.image_utils import dliate_erode
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.common import tensor2im
from utils.inference_utils import get_average_image

img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def hairstyle_feature_blending_2(generator, seg, src_image, visual_mask, latent_bald, latent_global=None, avg_image=None):

    if latent_global is not None:
        bald_feature = generator.decoder.synthesis(latent_bald, noise_mode='const')
        global_feature = generator.decoder.synthesis(latent_global, noise_mode='const')
        global_proxy = generator.decoder.synthesis(latent_global, noise_mode='const')
        global_proxy_seg = torch.argmax(seg(global_proxy)[1], dim=1).unsqueeze(1).long()

        ear_mask = torch.where(visual_mask==6, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_mask = torch.where(visual_mask==10, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_ear_mask = ear_mask + hair_mask
        bald_blending_mask = dliate_erode(hair_ear_mask.astype('uint8'), 30)
        bald_blending_mask = torch.from_numpy(bald_blending_mask).unsqueeze(0).unsqueeze(0).cuda()
        bald_blending_mask_down = F.interpolate(bald_blending_mask.float(), size=(1024, 1024), mode='bicubic')
        src_image = bald_feature * bald_blending_mask_down + src_image * (1-bald_blending_mask_down)

        global_hair_mask = torch.where(global_proxy_seg==10, torch.ones_like(global_proxy_seg), torch.zeros_like(global_proxy_seg))
        global_hair_mask_down = F.interpolate(global_hair_mask.float(), size=(1024, 1024), mode='bicubic')
        src_image = global_feature * global_hair_mask_down + src_image * (1-global_hair_mask_down)

    feat_out_img = tensor2im(src_image[-1])
    out = img_transforms(feat_out_img).unsqueeze(0).to('cuda')

    with torch.no_grad():
            avg_image = avg_image.unsqueeze(0).repeat(out.shape[0], 1, 1, 1)
            x_input = torch.cat([out, avg_image], dim=1)
            img_gen_blend,blend_latent = generator(x_input,latent=None, return_latents=True, resize=False)
    return src_image, img_gen_blend, blend_latent

def hair_aligning(generator, seg, src_image, visual_mask, latent_global=None):

    if latent_global is not None:
        #Seg on source
        source_seg = torch.argmax(seg(src_image)[1], dim=1).unsqueeze(1).long()
        bg = torch.where(source_seg == 0, torch.zeros_like(source_seg), torch.ones_like(source_seg)) #back ground mask
        hair_mask_source_seg = torch.where(source_seg == 10, torch.ones_like(source_seg), torch.zeros_like(source_seg)) #source hair mask
        without_hair_source_seg = torch.where(source_seg == 10, torch.zeros_like(source_seg), source_seg) #source mask remove hair
        #Seg on ref after align
        ref_aligned = generator.decoder.synthesis(latent_global, noise_mode='const')
        ref_seg = torch.argmax(seg(ref_aligned)[1], dim=1).unsqueeze(1).long()
        hair_mask_ref = torch.where(ref_seg == 10, torch.ones_like(ref_seg), torch.zeros_like(ref_seg)) #ref aligned hair mask
        #New target: ref hair on source image without hair mask
        new_target_seg = torch.where(ref_seg == 10, 10 * torch.ones_like(without_hair_source_seg), source_seg)
        new_target_mean_seg = torch.where((new_target_seg == 0) * (ref_seg != 0), ref_seg, new_target_seg)
        hair_mask_target = torch.where(new_target_mean_seg == 10, torch.ones_like(new_target_mean_seg), torch.zeros_like(new_target_mean_seg)) #hair mask on new target
        hair_mask_target = F.interpolate(hair_mask_target.float().unsqueeze(0), size=(1024,1024), mode='bicubic')
        
        src_image = ref_aligned * hair_mask_target + src_image * (1-hair_mask_target)

    feat_out_img = tensor2im(src_image[-1])
    out = img_transforms(feat_out_img).unsqueeze(0).to('cuda')

    with torch.no_grad():
            avg_image = get_average_image(generator)
            avg_image = avg_image.unsqueeze(0).repeat(out.shape[0], 1, 1, 1)
            x_input = torch.cat([out, avg_image], dim=1)
            img_gen_blend,blend_latent = generator(x_input,latent=None, return_latents=True, resize=False)
    return feat_out_img, img_gen_blend, blend_latent
