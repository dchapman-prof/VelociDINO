import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import blockreader_sa1b
import bicubic
import cascade_unet as cu







def mkdir(path):
	try:
		os.makedirs(path, exist_ok=True)
	except Exception as e:
		print("Could not create directory:", path, e)




print('---------------------------------------')
print(' Load the model and weights')
print('---------------------------------------')
indir = 'autoencoder'

device = 'cuda'

encoder = cu.Encoder().to(device)
encoder_path = indir + '/encoder_00000.pth'
print(encoder_path)
encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
encoder.eval()


decoder = cu.Decoder().to(device)
decoder_path = indir + '/decoder_00000.pth'
print(decoder_path)
decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
decoder.eval()



print('---------------------------------------------')
print(' Initialize Blockreader')
print('---------------------------------------------')
batch_size = 64
dataset_folder = "/nvme0/sa1b"
print(f"Initializing dataset from: {dataset_folder}")

# Initialize the dataset
# Note: Using your folder-based __init__ requirement
reader = blockreader_sa1b.SA1B_DINO_blockreader(dataset_folder, batch_size)
n_img   = reader.len()
#n_img   = 150
n_batch = (n_img+batch_size-1) // batch_size
#n_batch = n_img // batch_size    # NO partial batches
n_epoch = 90

print(f"N images: {n_img}")
print(f"N batches: {n_batch}")
print(f"Batch size: {batch_size}")
print("-" * 50)

start_time = time.time()
last_batch_time = start_time




print('---------------------------------------------')
print(' Initialize Bicubic')
print('---------------------------------------------')
bcb = bicubic.Bicubic(
  in_shape=(batch_size,1500,1500,4),
  img_shape=(batch_size,896,896,4),
  mask_shape=(batch_size,64,64,384),
  img_mean=(0.485, 0.456, 0.406),
  img_std=(0.229, 0.224, 0.225),
  device='cuda')




print('---------------------------------------------')
print(' For every batch (for visualization)')
print('---------------------------------------------')
n_batch = 1

with torch.no_grad():
	for batch in range(n_batch):

		print('----------------')
		print(' Read the images/features from lmdb')
		print('----------------')
		images, features_np, keys = reader.read_batch()
		if features_np.shape[0]!=batch_size:    # NO partial batches
			continue

		features = torch.tensor(features_np, dtype=torch.uint8, device='cuda', requires_grad=False)

		# Convert from [N,H,W,C] to [N,C,H,W]
		bcb.restore_uint8_features(features)
		features = bcb.mask_data_raw
		features = torch.permute(features, (0,3,1,2)).contiguous()
				
		print('----------------')
		print(' Run Autoencoder')
		print('----------------')
		features_laplace = LaplacianTranform(features)
		features_enc = encoder(features_laplace)
		features_dec = decoder(features_enc)

		print(keys)



print('Success!')
