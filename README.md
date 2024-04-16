# HAIR GAN
> **Hair-GAN: Aligning Portrait Images with Desired Hairstyles**<br/>
[Vu Le Bang Tam](https://github.com/TamVu02),
[Nguyen Minh Tuan](https://github.com/MinhTuan0510),
[Chu Quang Linh](https://github.com/linhchu1),

![Example result](assets/quantitive_result.png)

>  **Abstract** Past studies point that although recent methods have notably enhanced the intricacies of hair depiction, they frequently yield sub-optimal outputs when the pose of
the source image diverges substantially from that of the reference hair image.To address this primary challenge, our focus is to devise an high-performing
method for altering hairstyles given in source image and reference image for inputs while ensuring high quality results. We propose a novel approach uti-lizing StyleGAN3 to address this issue.
The methodology involves generating a latent vector representation of input source image using the StyleGAN3 encoder, optimizing the latent space, employing InterfaceGAN for hair and pose manipulation,
blending and applying StyleGAN3 decoder to generate a new version, refining hair and facial features of this version in final step. Experimental results demonstrate
the effectiveness of the proposed approach in achieving desired editing outcomes.
> 

## Installation dependencies
```
!pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
!pip install ftfy regex tqdm matplotlib jupyter ipykernel opencv-python scikit-image kornia==0.6.7 face-alignment==1.3.5
```
```
!pip install Ninja
!pip install pyrallis
```

## Getting Started  
Produce the results:
```
!python main_edit.py --img_list 06836 06838 06845 06853 06854
```
