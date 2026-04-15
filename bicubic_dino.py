import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from patchify import *
import bicubic

def VisImg(img):
	vis = img.detach().cpu().numpy()
	vis = 255.0 * ( (vis*img_std) + img_mean )
	vis[vis<0] = 0.0
	vis[vis>255] = 255.0
	vis = vis.astype(np.uint8)
	return vis

def ShowImg(img):
	vis = VisImg(img)
	plt.figure(figsize=(10,10))
	plt.imshow(vis)
	plt.show()
	#vis = Image.fromarray(vis)
	#vis.show()

def PyrDown(img):
	H,W,C = img.shape
	H__2 = H//2
	W__2 = W//2
	img = torch.reshape(img, (H__2,2,W__2,2,C))
	img = torch.mean(img, dim=(1,3))
	return img

print('--------------------')
print(' Obtain device')
print('--------------------')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)

print('--------------------')
print(' Load the model   dinov2_vits14')
print('--------------------')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval().cuda()

print(' Define image normalization')
img_mean=[0.485, 0.456, 0.406]
img_std=[0.229, 0.224, 0.225]
img_mean = np.reshape(img_mean, (1,1,3))
img_std  = np.reshape(img_std,  (1,1,3))
img_mean_torch = torch.tensor(img_mean, device=device, dtype=torch.float32, requires_grad=False)
img_std_torch  = torch.tensor(img_std,  device=device, dtype=torch.float32, requires_grad=False)

print('--------------------')
print(' Compile the kernel')
print('--------------------')
bicubic.Compile()


with torch.no_grad():

	print('--------------------')
	print(' Read the image')
	print('--------------------')
	imgname = 'horses_full.jpg'
	print('   ', imgname)

	img = Image.open(imgname).convert('RGB')
	img = np.array(img, dtype=np.uint8)
	print('img.shape', img.shape)
	print('img.dtype', img.dtype)

	img_uint8 = torch.tensor(img, device=device, dtype=torch.uint8, requires_grad=False)
	img_uint8 = img_uint8.contiguous();

	#sys.exit(1)
	print('--------------------')
	print(' Run aa uint8 interpolation')
	print('--------------------')
	print('img_uint8', type(img_uint8), img_uint8.dtype, img_uint8.shape)
	#corners = (((300,100),(800,300),(600,800),(100,600)),)
	corners = (
		((300,100),(800,300),(600,800),(100,600)),
		((0,0),(5,0),(5,5),(0,5)),
		((891,891),(896,891),(896,896),(891,896)),
		((0,0),(5,0),(5,896),(0,896))
		)
	aa_ker = (
		((8,8),
		 (1,1),
		 (1,1),
		 (8,1))
		 )
	#corners = (
	#	((0,0),(5,0),(5,896),(0,896)),
	#	)
	corners = torch.tensor(corners, device=device, requires_grad=False, dtype=torch.float32)
	aa_ker = torch.tensor(aa_ker, device=device, requires_grad=False, dtype=torch.int32)
	B = corners.shape[0]

	in_img = torch.reshape(img_uint8, (1,img_uint8.shape[0],img_uint8.shape[1],img_uint8.shape[2])).contiguous()
	in_img = in_img.repeat((B,1,1,1)).contiguous()
	
	out_img = torch.zeros((B,128,128,3), device=device, requires_grad=False, dtype=torch.uint8)
	out_img = out_img.contiguous()
	
	torch.cuda.synchronize()
	bicubic.bcb__module.bicubic_aa_uint8_cuda(out_img, in_img, corners, aa_ker)
	torch.cuda.synchronize()
	
	for j in range(B):
		vis = out_img[j].detach().cpu().numpy()
		plt.figure(figsize=(10,10))
		plt.imshow(vis)
		plt.show()
	
	print('--------------------')
	print('Perform standard normalization')
	print('--------------------')
	img = torch.tensor(img, device=device, requires_grad=False, dtype=torch.uint8)
	img = img.to(torch.float32) / 255.0
	img = (img-img_mean_torch) / img_std_torch
	img = StandardImg(img)

	print('--------------------')
	print(' Run bicubic interpolation')
	print('--------------------')
	print('img', type(img), img.dtype, img.shape)
	#corners = (((300,100),(800,300),(600,800),(100,600)),)
	#corners = (
	#	((300,100),(800,300),(600,800),(100,600)),
	#	((0,0),(5,0),(5,5),(0,5)),
	#	((891,891),(896,891),(896,896),(891,896)),
	#	((0,0),(5,0),(5,896),(896,0))
	#	)
	corners = (
		((0,0),(5,0),(5,896),(0,896)),
		)
	corners = torch.tensor(corners, device=device, requires_grad=False, dtype=torch.float32)
	B = corners.shape[0]

	in_img = torch.reshape(img, (1,896,896,3)).contiguous()
	in_img = in_img.repeat((B,1,1,1)).contiguous()
	
	out_img = torch.zeros((B,800,800,3), device=device, requires_grad=False, dtype=torch.float32)
	
	torch.cuda.synchronize()
	bicubic.bcb__module.bicubic_float_cuda(out_img, in_img, corners)
	torch.cuda.synchronize()

	#out_img = torch.reshape(out_img, (800,800,3))

	#vis = VisImg(img)
	for j in range(B):
		vis = VisImg(out_img[j])
		plt.figure(figsize=(10,10))
		plt.imshow(vis)
		plt.show()
	
	print('--------------------')
	print(' Run DINO')
	print('--------------------')
	img_tensor = img.permute(2,0,1)
	print('img_tensor.shape', img_tensor.shape)
	img_tensor = torch.reshape(img, (1,3,896,896))

	print('img_tensor.shape', img_tensor.shape)

	# 'forward_features' returns a dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
	features = model.forward_features(img_tensor)
	patch_tokens = features['x_norm_patchtokens']

	print(f"Patch tokens shape: {patch_tokens.shape}") 
	# Result: [batch_size, num_patches, embedding_dim] -> [1, 256, 384] for ViT-S


	print('patch_tokens.shape', patch_tokens.shape)
	
	sY=sX=896//14
	chan =patch_tokens.shape[2]
	featvec = torch.reshape(patch_tokens, (sY,sX,chan))
	
	featvec = featvec.detach().cpu().numpy()
	print('featvec.shape', featvec.shape, 'dtype', featvec.dtype, 'nbytes', featvec.nbytes)
	
	#featzip = Compress3D(featvec, -256, 256)
	
	# Zip the feature vectors
	featzip = Compress8bit(featvec)
	print('featzip type', type(featzip), 'len', len(featzip))

	# Unzip the feature vectors
	featvec2 = Decompress8bit(featzip)
	
	# How close are they
	diff = featvec2 - featvec
	maxdiff = np.max(np.abs(diff))
	avgdiff = np.mean(np.abs(diff))
	print('maxdiff', maxdiff)
	print('avgdiff', avgdiff)