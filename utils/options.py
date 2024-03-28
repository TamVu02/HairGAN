from argparse import ArgumentParser
from models.stylegan3.model import GeneratorType


class Options:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for pretrained model path
		self.parser.add_argument('--stylegan_path', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/pretrained_models/ffhq.pt", type=str, help='Path to StyleGAN model checkpoint')
		self.parser.add_argument('--seg_path', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/pretrained_models/seg.pth", type=str, help='Path to face parsing model checkpoint')
		self.parser.add_argument('--ffhq_pca_path', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/pretrained_models/ffhq_PCA.npz", type=str, help='Path to FFHQ PCA')

		# StyleGAN3 setting
		self.parser.add_argument('--stylegan3_weights', type=str, default='/content/drive/MyDrive/HairGAN/Final_HairGAN/pretrained_models/restyle_e4e_ffhq.pt')
		self.parser.add_argument('--generator_path3', type=str, default='/content/drive/MyDrive/HairGAN/Final_HairGAN/pretrained_models/sg3-r-ffhq-1024.pt')
		self.parser.add_argument('--generator_type', type=str, default=GeneratorType.ALIGNED)
		self.parser.add_argument('--stylegan3_size', type=int, default=1024)
		self.parser.add_argument('--stylegan3_truncation', type=float, default=0.7)
		self.parser.add_argument('--latents_path', type=str, default='/content/drive/MyDrive/HairGAN/Final_HairGAN/data_npy/W+',help='Folder of latent.npy')

		# arguments for image and latent dir path
		self.parser.add_argument('--src_img_dir', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/Asian_Dataset", type=str, help='Folder of source image')
		self.parser.add_argument('--ref_img_dir', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/Asian_Dataset", type=str, help='Folder of reference image')
		self.parser.add_argument('--latent_dir', default="/content/drive/MyDrive/HairGAN/Final_HairGAN/data_npy/W", type=str, help='Folder of reference latent')

		# arguments for embedding
		self.parser.add_argument('--W_steps', default=1100, type=int, help='Step for W plus inversion')
		self.parser.add_argument('--SG3_steps', default=1, type=int, help='Step for stylegan3 decoder iter')
		self.parser.add_argument('--lr_embedding', default=0.02, type=float, help='Learning rate for embedding')
		self.parser.add_argument('--l2_lambda_embedding', default=1.0, type=float, help='L2 loss lambda for embedding')
		self.parser.add_argument('--percept_lambda_embedding', default=1.0, type=float, help='LPIPS loss lambda for embedding')
		self.parser.add_argument('--p_norm_lambda_embedding', default=0.001, type=float, help='P norm loss lambda for embedding')
		self.parser.add_argument('--l_F_lambda_embedding', default=0.1, type=float, help='F norm loss lambda for embedding')

		# arguments for reference proxy
		self.parser.add_argument('--style_lambda_ref', default=40000.0, type=float, help='Style loss lambda for reference proxy')
		self.parser.add_argument('--delta_w_lambda_ref', default=1000.0, type=float, help='Delta W loss lambda for reference proxy')
		self.parser.add_argument('--hair_mask_lambda_ref', default=1.0, type=float, help='Hair mask loss lambda for reference proxy')
		self.parser.add_argument('--landmark_lambda_ref', default=1000.0, type=float, help='Landmark loss lambda for reference proxy')
		self.parser.add_argument('--lr_ref', default=0.01, type=float, help='Learning rate for reference proxy')
		self.parser.add_argument('--steps_ref', default=200, type=int, help='Step for reference proxy optimization')
		self.parser.add_argument('--visual_num_ref', default=10, type=int, help='Visual image number during reference proxy optimization')
		self.parser.add_argument('--n_iter_infer', default=5, type=int, help='Iteration for generating final blending output')

		# arguments for refine proxy
		self.parser.add_argument('--steps_refine', default=400, type=int, help='Step for reference proxy optimization')
		self.parser.add_argument('--lr_refine', default=0.01, type=float, help='Learning rate for reference proxy')
                self.parser.add_argument('--device', default='cuda', type=str)


	def parse(self, jupyter=False):
		if jupyter:
			opts = self.parser.parse_args(args=[])
		else:
			opts = self.parser.parse_args()
		return opts
