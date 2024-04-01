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
from scripts.refine_image import RefineProxy
from scripts.feat_blend_2 import hairstyle_feature_blending_2
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
    ref_np = ref_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    ssim_score = ssim(src_np, ref_np, multichannel = True)
    return ssim_score

def psnr(image1, image2):
    # Convert PIL images to NumPy arrays
    array1 = np.array(image1)#.astype(float)
    array2 = np.array(image2)#.astype(float)

    # Calculate mean squared error (MSE)
    mse = np.mean((array1 - array2) ** 2)
    
    # Calculate maximum possible pixel value (assuming 8-bit image)
    max_val = 255

    # Calculate PSNR
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

def main(args):
    #Load args
    opts = Options().parse(jupyter=True)
    opts.W_steps=500
    opts.steps_ref=200
    opts.steps_refine=400

    #Define image transform
    image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    #Load stylegan3 model for generator
    generator, opts_sg3, seg, avg_img = load_sg3_models(opts)
    #Load embedding and loss
    re4e = Embedding_sg3(opts, generator)
    loss_builder = EmbeddingLossBuilder(opts)
    #Load ref proxy
    ref_proxy = RefProxy(opts, generator, seg, re4e)
    refine_proxy = RefineProxy(opts, generator, seg)
    #Load interfaceGAN for bald proxy
    editor = FaceEditor(stylegan_generator=generator.decoder, generator_type=GeneratorType.ALIGNED)
    edit_direction=['Bald','pose']
    separator = '=' * 90
    #open output result metric csv file
    csv_output_file = open_csv_file(args.output_result)
    csv_writer = csv.writer(csv_output_file)
    if os.path.getsize(args.output_result) == 0:
        csv_writer.writerow(['source', 'target', 'lpips_score','ssim_score','psnr_score'])

    for img in args.img_list:
        print(separator)
        if os.path.isfile(os.path.join(opts.src_img_dir,f'{img}.png')):
            print(f"Performing edit on image {img}.png")
            src_name=img
            if not os.path.isfile(os.path.join(opts.latent_dir, f"{src_name}.npy")):
                src_latent = re4e.invert_image_in_W(image_path=os.path.join(opts.src_img_dir,f'{img}.png'), device='cuda', avg_image=avg_img)
            else:
                src_latent = torch.from_numpy(np.load(f'{opts.latent_dir}/{src_name}.npy')).cuda()
            src_pil = Image.open(f'{opts.src_img_dir}/{src_name}.png').convert('RGB')
            src_image = image_transform(src_pil).unsqueeze(0).cuda()
            input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

            #Perform interface gan with bald pretrain model
            print(f"Performing edit for {edit_direction}...")
            bald_feat, edit_latents = editor.edit(src_image,
                                        latents=src_latent,
                                        direction=edit_direction[0],
                                        factor = 5,
                                        user_transforms=None,
                                        apply_user_transformations=False)
            latent_bald=edit_latents[-1].unsqueeze(0)

            #Retrieve 3 random image in image_list
            target_img_list=['06853','06845','06838']
            
            for target_name in target_img_list:
                if os.path.isfile(os.path.join(opts.src_img_dir,f'{target_name}.png')):
                      print(f"\n==Performing edit source image on target image {target_name}.png")
                      if not os.path.isfile(os.path.join(opts.latent_dir, f"{target_name}.npy")):
                        ref_latent = re4e.invert_image_in_W(image_path=os.path.join(opts.src_img_dir,f'{target_name}.png'), device='cuda', avg_image=avg_img)
                      else:
                        ref_latent = torch.from_numpy(np.load(f'{opts.latent_dir}/{target_name}.npy')).cuda()
                      #Warp image base on source image pose att
                      
                      ref_feat, edit_latents = editor.edit(src_image,
                                        latents=ref_latent,
                                        direction=edit_direction[1],
                                        factor_range = (-5,5),
                                        user_transforms=None,
                                        apply_user_transformations=False)
                      latent_global=edit_latents[-1]
                      print(latent_global.shape)
                      print(type(ref_feat[-1][0]))
                      #Blending feature
                      blend_source,_, edited_latent = hairstyle_feature_blending_2(generator, seg, src_image, input_mask,latent_bald, latent_global, avg_img)
                      #Refine blending image
                      target_mask = seg(blend_source)[1]
                      final_image,_,_=refine_proxy(blended_latent=edited_latent, src_image=src_image, ref_img=ref_feat[-1][0],target_mask=target_mask)
                      #Print metric score
                      lpips_score = loss_builder._loss_lpips(src_image, final_image).item()
                      ssim_score = calculate_ssim_score_skimage(src_image,final_image)
                      img_output = Image.fromarray(process_display_input(final_image))
                      psnr_score = psnr(img_output, src_pil)
                      print(f'LPIPS score: {lpips_score} \t SSIM score: {ssim_score} \t PSNR score: {psnr_score}')
                      #save metric as format: source_name, target_name, lpips_score, ssim_score
                      csv_writer.writerow([src_name, target_name, lpips_score, ssim_score, psnr_score])
                      #Save output image
                      im_path = os.path.join(args.save_output_dir, f'{src_name}_{target_name}_refine.png')
                      img_output.save(im_path)
                      print(f'Done saving output {src_name}_{target_name}_refine.png to {args.save_output_dir}')
                else:
                    print(f'Image target {target_name}.png does not exit in {opts.src_img_dir}')
        else:
            print(f'Image source {img}.png does not exit in {opts.src_img_dir}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HairGAN')

    parser.add_argument('--save_output_dir', type=str ,default='/content/drive/MyDrive/HairGAN/Final_HairGAN/output_img/refine_2',help='directory for saving images after blending')
    parser.add_argument('--img_list', type=str,nargs='+',help='image list eg: 00004 00006 00131 03177')
    parser.add_argument('--output_result', type=str, default='/content/drive/MyDrive/HairGAN/Final_HairGAN/output_img/result_metric_refine.csv', help='csv file for saving metric result')

    
    args = parser.parse_args()
    main(args)
