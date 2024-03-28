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

def hairstyle_feature_blending(generator, seg, src_latent, src_feature, src_image, visual_mask, latent_bald, latent_global=None, latent_local=None, local_blending_mask=None, n_iter=5):

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

    if latent_local is not None:
        local_feature = generator.decoder.synthesis(latent_local, noise_mode='const')#generator.decoder.synthesis([latent_local], input_is_latent=True, return_latents=True, start_layer=0, end_layer=3)
        local_blending_mask = torch.from_numpy(local_blending_mask[:,:,0]).unsqueeze(0).unsqueeze(0).long().cuda()
        local_blending_mask = torch.where(local_blending_mask==1, torch.ones_like(local_blending_mask), torch.zeros_like(local_blending_mask))
        local_blending_mask_down = F.interpolate(local_blending_mask.float(), size=(32, 32), mode='bicubic')
        src_feature = local_feature * local_blending_mask_down + src_feature * (1-local_blending_mask_down)

    feat_out_img = tensor2im(src_image[-1])
    out = img_transforms(feat_out_img).unsqueeze(0).to('cuda')
    img_gen_blend, latent=None, None

    with torch.no_grad():
        for i in range (n_iter):
            if i==0:
                 avg_image = get_average_image(generator)
                 avg_image = avg_image.unsqueeze(0).repeat(out.shape[0], 1, 1, 1)
                 x_input = torch.cat([out, avg_image], dim=1)
            else:
                 img_gen_blend = generator.face_pool(img_gen_blend).detach().clone()
                 x_input = torch.cat([out, img_gen_blend], dim=1)
            img_gen_blend,latent = generator(x_input,latent=latent, return_latents=True, resize=False)
    return feat_out_img, src_feature, img_gen_blend