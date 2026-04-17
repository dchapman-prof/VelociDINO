import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

from bicubic import Bicubic

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def center_crop_and_resize(img, out_size):
	W, H = img.size
	side = min(W, H)

	left = (W - side) // 2
	top = (H - side) // 2
	right = left + side
	bottom = top + side

	img = img.crop((left, top, right, bottom))
	img = img.resize((out_size, out_size), Image.BILINEAR)
	return img


def read_img(path, size, device):
	print('ReadImg', path)

	img = Image.open(path).convert('RGB')
	img = center_crop_and_resize(img, size)
	img = np.array(img, dtype=np.uint8)
	img = torch.tensor(img, device=device, dtype=torch.uint8, requires_grad=False)

	return img


def normalize_feature_map(x):
	x = x - x.min()
	if x.max() > 0:
		x = x / x.max()
	return x


def save_feature_map_gray(feature_map, save_path):
	feature_map = feature_map.detach().cpu().numpy()
	#feature_map = normalize_feature_map(torch.tensor(feature_map)).numpy()

	plt.figure(figsize=(4, 4))
	#plt.imshow(feature_map, cmap='gray')
	plt.imshow(feature_map, norm=colors.CenteredNorm(), cmap=cm.coolwarm)
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def save_average_map_gray(feature_maps, save_path):
	avg_map = torch.mean(feature_maps, dim=0)
	#avg_map = normalize_feature_map(avg_map)

	plt.figure(figsize=(4, 4))
	#plt.imshow(avg_map.detach().cpu().numpy(), cmap='gray')
	plt.imshow(avg_map.detach().cpu().numpy(), norm=colors.CenteredNorm(), cmap=cm.coolwarm)
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
	plt.close()


def VisImg(img):
	vis = img.detach().cpu().numpy()
	vis = 255.0 * ( (vis*img_std) + img_mean )
	vis[vis<0] = 0.0
	vis[vis>255] = 255.0
	vis = vis.astype(np.uint8)
	return vis

def ShowImg(img):
	#vis = VisImg(img)
	vis = img.detach().cpu().numpy()
	vis = ( (vis*img_std) + img_mean )
	plt.figure(figsize=(10,10))
	plt.imshow(vis)
	plt.show()
	#vis = Image.fromarray(vis)
	#vis.show()

def ShowImgPlain(img):
	#vis = 255.0 * img.detach().cpu().numpy()
	#vis[vis<0] = 0.0
	#vis[vis>255] = 255.0
	vis = img.detach().cpu().numpy()
	vis[vis<0.0] = 0.0
	vis[vis>1.0] = 1.0
	plt.figure(figsize=(10,10))
	plt.imshow(vis)
	plt.show()
	#vis = Image.fromarray(vis)
	#vis.show()

img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]
img_mean = np.reshape(img_mean, (1,1,3))
img_std  = np.reshape(img_std,  (1,1,3))
img_mean_torch = torch.tensor(img_mean, device='cuda', dtype=torch.float32, requires_grad=False)
img_std_torch  = torch.tensor(img_std,  device='cuda', dtype=torch.float32, requires_grad=False)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_dir', type=str, default='images')
	parser.add_argument('--out_dir', type=str, default='dino_featuremaps')
	parser.add_argument('--model_name', type=str, default='dinov2_vits14')
	parser.add_argument('--img_size', type=int, default=448)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--save_all_channels', action='store_true')
	parser.add_argument('--max_channels', type=int, default=32)
	args = parser.parse_args()

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)

	mkdir(args.out_dir)

	print('--------------------')
	print(' Create the Bicubic augmenter')
	print('--------------------')
	bcb = Bicubic(in_shape=(48,448,448,4),
	  img_shape=(48,224,224,4),
	  mask_shape=(48,32,32,384))
	
	print('--------------------')
	print('Load model')
	print('--------------------')
	model = torch.hub.load('facebookresearch/dinov2', args.model_name)
	model.eval().to(device)

	print('--------------------')
	print('Read images')
	print('--------------------')
	img_names = sorted(os.listdir(args.img_dir))
	img_names = [x for x in img_names if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))]

	print('img_names', img_names)
	print('n_img', len(img_names))

	if len(img_names) == 0:
		print('No images found in:', args.img_dir)
		return

	img_tensor = torch.zeros((len(img_names), args.img_size, args.img_size, 4), dtype=torch.uint8, device=device)
	img_tensor[:,:,:,3] = 255

	for i, img_name in enumerate(img_names):
		img = read_img(os.path.join(args.img_dir, img_name), args.img_size, device)
		img_tensor[i,:,:,0:3] = img

	print('img_tensor.shape', img_tensor.shape, img_tensor.device, img_tensor.dtype)

	img_tensor_dino = img_tensor[:,:,:,0:3].to(torch.float32) / 255.0
	img_tensor_dino = (img_tensor_dino-img_mean_torch) / img_std_torch
	img_tensor_dino_back = img_tensor_dino
	
	img_tensor_dino = torch.permute(img_tensor_dino, (0,3,1,2))
	
	print('img_tensor_dino.shape', img_tensor_dino.shape, img_tensor_dino.device, img_tensor_dino.dtype)

	print('--------------------')
	print('Run DINO')
	print('--------------------')
	with torch.no_grad():
		features = model.forward_features(img_tensor_dino)
		patch_tokens = features['x_norm_patchtokens']		# [B, N, C]

	print('patch_tokens.shape', patch_tokens.shape)

	B, N, C = patch_tokens.shape
	pY = args.img_size // 14
	pX = args.img_size // 14

	if pY * pX != N:
		raise ValueError(f'Patch grid mismatch: pY*pX={pY*pX}, but N={N}')

	patch_tokens = patch_tokens.reshape(B, pY, pX, C)		# [B, pY, pX, C]
	print('reshaped patch_tokens.shape', patch_tokens.shape)


	print('--------------------')
	print(' Run Augmentation!!!')
	print('--------------------')
	img_tensor  =img_tensor.contiguous()
	patch_tokens=patch_tokens.contiguous()
	bcb.mask_data_raw = patch_tokens
	bcb.roll_dice()
	
	#print('img_corners', bcb.img_corners)
	#print('img_aa_ker', bcb.img_aa_ker)
	#print('blur_sigmas_x', bcb.blur_sigmas_x)
	#print('blur_sigmas_y', bcb.blur_sigmas_y)
	for i in range(48):
		#print('blur sigma x %.4f  y %.4f' % (bcb.blur_sigmas_x[i], bcb.blur_sigmas_y[i]))
		print('noise_scale %.4f' % bcb.noise_scales[i])

	torch.set_printoptions(sci_mode=False)
	print('mask corners', bcb.mask_corners)
	print('mask corners.shape', bcb.mask_corners.shape)
	#sys.exit(1)
	
	bcb.bicubic_masks()
	bcb.augment_images(img_tensor)

	for n in range(48):
		#ShowImgPlain(bcb.img_data_bicubic[n,:,:,0:3])
		#ShowImgPlain(bcb.img_data_bicubic[n,:,:,0:3])
		#ShowImgPlain(bcb.img_data_blur[n,:,:,0:3])
		ShowImg(bcb.img_data[n,:,:,0:3])
		#ShowImg(img_tensor_dino_back[n,:,:,0:3])
		#plt.figure(figsize=(10,10))
		#plt.imshow(img_tensor[n,:,:,0:3].detach().cpu().numpy())
		#plt.show()


	sys.exit(1)

	print('--------------------')
	print('Save feature maps')
	print('--------------------')
	for i, img_name in enumerate(img_names):
		base_name = os.path.splitext(img_name)[0]
		img_out_dir = os.path.join(args.out_dir, base_name)
		mkdir(img_out_dir)

		feat = patch_tokens[i]						# [pY, pX, C]
		feat = torch.permute(feat, (2, 0, 1))				# [C, pY, pX]

		np.save(os.path.join(img_out_dir, 'patch_tokens.npy'), feat.detach().cpu().numpy())

		save_average_map_gray(feat, os.path.join(img_out_dir, 'average_featuremap.png'))

		if args.save_all_channels:
			n_save = feat.shape[0]
		else:
			n_save = min(args.max_channels, feat.shape[0])

		for c in range(n_save):
			save_path = os.path.join(img_out_dir, f'feature_{c:04d}.png')
			save_feature_map_gray(feat[c], save_path)

		print(f'Saved feature maps for {img_name} in {img_out_dir}')

	print('Done!')


if __name__ == '__main__':
	main()
