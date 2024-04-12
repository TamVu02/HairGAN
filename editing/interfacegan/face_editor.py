from typing import Optional, Tuple

import numpy as np
import torch

from configs.paths_config import interfacegan_aligned_edit_paths, interfacegan_unaligned_edit_paths
from models.stylegan3.model import GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import tensor2im, generate_random_transform
import face_alignment
import torch.nn.functional as F


class FaceEditor:

    def __init__(self, stylegan_generator: Generator, generator_type=GeneratorType.ALIGNED):
        self.generator = stylegan_generator
        self.kp_extractor = self.load_kp_extractor()
        if generator_type == GeneratorType.ALIGNED:
            paths = interfacegan_aligned_edit_paths
        else:
            paths = interfacegan_unaligned_edit_paths

        self.interfacegan_directions = {
            # 'age': torch.from_numpy(np.load(paths['age'])).cuda(),
            # 'smile': torch.from_numpy(np.load(paths['smile'])).cuda(),
            # 'pose': torch.from_numpy(np.load(paths['pose'])).cuda(),
            # 'Male': torch.from_numpy(np.load(paths['Male'])).cuda(),
            'Bald': torch.from_numpy(np.load(paths['Bald'])).cuda(),
            'pose': torch.from_numpy(np.load(paths['pose'])).cuda(),
        }

    def edit(self, src_image: torch.tensor, latents: torch.tensor, direction: str, factor: int = 1, factor_range: Optional[Tuple[int, int]] = None,
             user_transforms: Optional[np.ndarray] = None, apply_user_transformations: Optional[bool] = False):
        edit_latents = []
        edit_images = []
        direction_ = self.interfacegan_directions[direction]
        initial_diff=100000000
                 
        if direction=='pose':
          kp_source = self.get_kp_extractor(src_image).detach().cpu().numpy()

        if direction=='pose':
            ref_image, _ = self._latents_to_image(latents,
                                                   apply_user_transformations,
                                                   user_transforms)
            initial_diff = self.cal_pose_diff(kp_source, ref_image, 'cuda')
            print(f'Initial keypoint difference: {initial_diff}')
            if initial_diff < 0.005:
                return None, None
            else:
                edit_latents.append(latents)
            
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction_
                edit_image, user_transforms = self._latents_to_image(edit_latent,
                                                                     apply_user_transformations,
                                                                     user_transforms)
                if direction=='pose':
                  cur_diff_score = self.cal_pose_diff(kp_source, edit_image, 'cuda')
                  if cur_diff_score < initial_diff:
                    print(f'Current updated keypoint difference: {cur_diff_score}')
                    edit_images.append(edit_image)
                    edit_latents.append(edit_latent)
                    initial_diff = cur_diff_score
                else:
                  edit_latents.append(edit_latent)
                  edit_images.append(edit_image)
        else:
            edit_latents = latents + factor * direction_
            edit_images, _ = self._latents_to_image(edit_latents, apply_user_transformations)
        return edit_images, edit_latents

    def _latents_to_image(self, all_latents: torch.tensor, apply_user_transformations: bool = False,
                          user_transforms: Optional[torch.tensor] = None):
        with torch.no_grad():
            if apply_user_transformations:
                if user_transforms is None:
                    # if no transform provided, generate a random transformation
                    user_transforms = generate_random_transform(translate=0.3, rotate=25)
                # apply the user-specified transformation
                if type(user_transforms) == np.ndarray:
                    user_transforms = torch.from_numpy(user_transforms)
                self.generator.synthesis.input.transform = user_transforms.cuda().float()
            # generate the images
            images = self.generator.synthesis(all_latents, noise_mode='const')
        return images, user_transforms

    def get_kp_extractor(self, input_image):
        return self.kp_extractor.face_alignment_net(((F.interpolate(input_image, size=(256, 256)) + 1) / 2).clamp(0, 1))

    def cal_pose_diff(self,kp1, im_2, device):
        kp2 = self.get_kp_extractor(im_2).detach().cpu().numpy()
        kp_diff = np.mean(np.abs(kp1 - kp2))
        return kp_diff

    def load_kp_extractor(self):
        kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')
        for param in kp_extractor.face_alignment_net.parameters():
            param.requires_grad = False
        return kp_extractor

