import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scripts.Embedding_sg3 import Embedding_sg3
from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from scripts.ref_proxy import RefProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.model_utils import load_sg3_models
from utils.options import Options
from utils.image_utils import process_display_input
import argparse
import random
from skimage.metrics import structural_similarity as ssim
from criteria.embedding_loss import EmbeddingLossBuilder
import csv

def open_csv_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as new_file:
            pass
    return open(file_path, 'a', newline='')

# Calculate SSIM score using skimage
def calculate_ssim_score_skimage(src_tensor, ref_tensor):
    src_np = src_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    ref_np = ref_tensor.squeeze().permute(1, 2, 0).cpu().numpy() 
    ssim_score = ssim(src_np, ref_np, multichannel = True)
    return ssim_score

def main(args):
    #Load args
    opts = Options().parse(jupyter=True)
    print(args)

    #Define image transform
    image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #Load stylegan3 model for generator
    generator, opts_sg3, mean_latent_code, seg = load_sg3_models(opts)
    #Load embedding and loss
    re4e = Embedding_sg3(opts, generator, mean_latent_code[0,0])
    loss_builder = EmbeddingLossBuilder(opts)
    #Load ref proxy
    ref_proxy = RefProxy(opts, generator, seg, re4e)
    #Load interfaceGAN for bald proxy
    editor = FaceEditor(stylegan_generator=generator.decoder, generator_type=GeneratorType.ALIGNED)
    edit_direction='Bald'
    min_value=2
    max_value=7
    separator = '=' * 90
    #open output result metric csv file
    csv_output_file = open_csv_file(args.output_result)
    csv_writer = csv.writer(csv_output_file)
    if os.path.getsize(args.output_result) == 0:
        csv_writer.writerow(['source', 'target', 'lpips_score','ssim_score'])

    for img in args.img_list:
        print(separator)
        if os.path.isfile(os.path.join(opts.src_img_dir,f'{img}.png')):
            print(f"Performing edit on image {img}.png")
            src_name=img
            #Embedding source image
            if not os.path.isfile(os.path.join(opts.src_latent_dir, f"{src_name}.npz")):
                inverted_latent_w_plus, inverted_latent_F = re4e.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.png')
                save_latent_path = os.path.join(opts.src_latent_dir, f'{src_name}.npz')
                np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                         latent_F=inverted_latent_F.detach().cpu().numpy())

            src_latent = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_in']).cuda()
            src_feature = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_F']).cuda()
            src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.png').convert('RGB')).unsqueeze(0).cuda()
            input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

            #Perform interface gan with bald pretrain model
            print(f"Performing edit for {edit_direction}...")
            edit_images, edit_latents = editor.edit(latents=src_latent,
                                        direction=edit_direction,
                                        factor_range=(min_value, max_value),
                                        user_transforms=None,
                                        apply_user_transformations=False)
            print("Done!")
            latent_bald=edit_latents[-1]

            #Retrieve 3 random image in image_list
            img_list_alt=args.img_list.copy()
            random.shuffle(img_list_alt)
            target_img_list=[im for im in img_list_alt if im != src_name][:3]
            
            for target_name in target_img_list:
                if os.path.isfile(os.path.join(opts.src_img_dir,f'{target_name}.png')):
                      print(f"==Performing edit source image on target image {target_name}.png")
                      #Run ref proxy on target image
                      latent_global, visual_global_list=ref_proxy(target_name+'.png', src_image=src_image, m_style=6)
                      #Blending feature
                      _,src_feature, edited_hairstyle_img = hairstyle_feature_blending(generator, seg, src_latent, src_feature, src_image, input_mask,latent_global=latent_global,latent_bald=latent_bald,n_iter=2)
                      lpips_score = loss_builder._loss_lpips(src_image, edited_hairstyle_img)
                      ssim_score = calculate_ssim_score_skimage(src_image,edited_hairstyle_img)
                      print(f'LPIPS score: {lpips_score} \t SSIM score: {ssim_score}')
                      #Save score as format: source_name, target_name, lpips_score, ssim_score
                      csv_writer.writerow([src_name, target_name, lpips_score, ssim_score])
                      #Save output image
                      img_output = Image.fromarray(process_display_input(edited_hairstyle_img))
                      im_path = os.path.join(args.save_output_dir, f'{src_name}_{target_name}.png')
                      img_output.save(im_path)
                      print(f'Done saving output {src_name}_{target_name}.png to {args.save_output_dir}')
                else:
                    print(f'Image target {target_name}.png does not exit in {opts.src_img_dir}')
        else:
            print(f'Image source {img}.png does not exit in {opts.src_img_dir}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HairGAN')

    parser.add_argument('--save_output_dir', type=str ,default='/content/drive/MyDrive/HairGAN/Final_HairGAN/output_img',help='directory for saving images after blending')
    parser.add_argument('--img_list', type=str,nargs='+',help='image list eg: 00004 00006 00131 03177')
    parser.add_argument('--output_result', type=str, default='/content/drive/MyDrive/HairGAN/Final_HairGAN/output_img/result_metric.csv', help='csv file for saving metric result')

    
    args = parser.parse_args()
    main(args)