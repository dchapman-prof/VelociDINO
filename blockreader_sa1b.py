import lmdb
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np

#-------------------------------
# A PyTorch Dataset that reads SA-1B images
#  and DINO features from an LMDB database.
#
# This dataset:
# 1. Loads filenames (keys) from a provided text file.
# 2. Fetches the raw JPEG binary images LMDB database.
# 3. Decompresses the JPEG using PIL.
# 4. Center-crops the image to a fixed size (e.g., 1500x1500px).
#
# This assumes that the short side of the input images is >= the target crop size.
# (This is true for the standard SA-1B release, which has a 1500px short side).
#-------------------------------

class SA1B_DINO_blockreader:

	#--------------------
	#	__init__:
	#		folder_path (str): Path to folder with the following
	#		   images.lmdb  (images)   dino.lmdb  (features)  images.txt  (image list)
	#		transform (callable, optional): Additional torchvision transforms to apply after cropping
	#										(e.g., transforms.ToTensor()).
	#--------------------
	def __init__(self, folder_path, batch_size):
		self.folder_path = folder_path
		self.crop_size = (1500, 1500)
		self.images_lmdb_path = os.path.join(folder_path, 'images.lmdb')
		self.feat_lmdb_path   = os.path.join(folder_path, 'dino.lmdb')
		self.keys_txt_path  = os.path.join(folder_path,   'images.txt')
		self.batch_size = batch_size

		# 1. Read the keys from the text file (fast)
		if not os.path.exists(self.keys_txt_path):
			raise FileNotFoundError("Could not find keys file", self.keys_txt_path)

		with open(self.keys_txt_path, 'r') as f:
			# Strip whitespace and ignore empty lines
			self.keys = [line.strip().encode('ascii') for line in f if line.strip()]

		# 2. LMDB Setup (environment initialized later, lazily, for better multiprocessing support)
		self.images_env = None
		self.images_txn = None
		self.feat_env = None
		self.feat_txn = None

		# Define the center crop transform
		self.center_crop = transforms.CenterCrop(self.crop_size)

		#-----------------
		#	Initializes the LMDB environment lazily.
		#	This is crucial when using multiple worker processes in a DataLoader
		#	to avoid file descriptor issues.
		#-----------------

		# Open the environment in read-only mode, with no write locks for performance
		# (unless user specifically requested locks)
		self.images_env = lmdb.open(self.images_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
		self.images_txn = self.images_env.begin(write=False)
		self.feat_env = lmdb.open(self.feat_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
		self.feat_txn = self.feat_env.begin(write=False)

		#-------
		# Allocate the feature buffers
		#-------
		self.len_frame = 1574412   # length of a DINO frame, ToDo make adaptive
		self.images_np   = np.zeros((batch_size, 1500, 1500, 3), dtype=np.uint8)
		self.features_np = np.zeros((batch_size, self.len_frame), dtype=np.uint8)

		#-------
		# Set the start and end indices
		#-------
		self.sidx = 0
		self.eidx = 0

	def len(self):
		return len(self.keys)

	def read_batch(self):

		# Obtain the list of keys
		n_keys = len(self.keys)
		self.sidx = self.eidx % n_keys
		self.eidx = min(self.sidx+self.batch_size, n_keys)
		B = self.eidx - self.sidx

		# 1. Get the key (filename)
		keys = self.keys[self.sidx:self.eidx]

		#-----------
		# Read the images
		#-----------
		for b in range(B):

			# Read the JPEG bytes
			jpeg_bin = self.images_txn.get(keys[b])
			if jpeg_bin is None:
				raise KeyError('Key ' + keys[b] + 'not found in LMDB' + self.images_lmdb_path)

			# Convert and crop the image
			image = Image.open(io.BytesIO(jpeg_bin)).convert('RGB')
			image_cropped = self.center_crop(image)
			image_np = np.array(image_cropped, dtype=np.uint8)

			# Copy to numpy array
			self.images_np[b,:,:,:] = image_np

		#-----------
		# Read the features
		#-----------
		for b in range(B):

			# Read the features
			feat_bin = self.feat_txn.get(keys[b])
			if feat_bin is None:
				raise KeyError('Key ' + keys[b] + 'not found in LMDB' + self.feat_lmdb_path)
			feat_np  = np.frombuffer(feat_bin, dtype=np.uint8)

			# Copy to numpy array
			self.features_np[b,:] = feat_np

		#-----------
		# Return the batches
		#-----------
		if B==self.batch_size:
			return self.images_np, self.features_np
		else:
			return self.images_np[0:B], self.features_np[0:B]



#---------------------------------------
#---------------------------------------
# Testing and benchmarking SA1B + DINO  LMDB batches
#---------------------------------------
#---------------------------------------
if __name__ == "__main__":

	#---------------
	# Tests the SA-1B LMDB pipeline with shuffling and timing.
	#---------------
	batch_size = 128
	num_batches = 1000
	dataset_folder = "/data/developers/sa1b"


	print(f"Initializing dataset from: {dataset_folder}")

	# Initialize the dataset
	# Note: Using your folder-based __init__ requirement
	reader = SA1B_DINO_blockreader(dataset_folder, batch_size)
	n_img   = reader.len()
	n_batch = (n_img+batch_size-1) // batch_size

	print(f"N images: {n_img}")
	print(f"N batches: {n_batch}")
	print(f"Batch size: {batch_size}")
	print("-" * 50)

	start_time = time.time()
	last_batch_time = start_time

	# Iterate through the dataloader
	for i in range(n_batch):

		images, features = reader.read_batch()

		current_time = time.time()
		batch_duration = current_time - last_batch_time
		total_elapsed = current_time - start_time

		# Print shapes and dtypes to verify the 'collate' logic
		# Images should be [B, 1500, 1500, 3], Features [B, F]
		print(f"Batch {i+1}/{num_batches}")
		print(f"  Images:   Shape {images.shape} | Dtype {images.dtype}")
		print(f"  Features: Shape {features.shape} | Dtype {features.dtype}")
		print(f"  Timing:   Batch: {batch_duration:.4f}s | Total: {total_elapsed:.4f}s")
		print("-" * 30, flush=True)

		last_batch_time = current_time


	final_time = time.time()
	avg_speed = (num_batches * batch_size) / (final_time - start_time)

	print("\nTest Complete!")
	print(f"Total time for {num_batches} batches: {final_time - start_time:.2f}s")
	print(f"Average throughput: {avg_speed:.2f} images/sec")

