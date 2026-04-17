import lmdb
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as transforms


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

class SA1B_DINO(Dataset):

	#--------------------
	#	__init__:
	#		folder_path (str): Path to folder with the following
	#		   images.lmdb  (images)   dino.lmdb  (features)  images.txt  (image list)
	#		transform (callable, optional): Additional torchvision transforms to apply after cropping
	#										(e.g., transforms.ToTensor()).
	#--------------------
	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.crop_size = (1500, 1500)
		self.transform = transform
		self.images_lmdb_path = os.path.join(folder_path, 'images.lmdb')
		self.feat_lmdb_path   = os.path.join(folder_path, 'dino.lmdb')
		self.keys_txt_path  = os.path.join(folder_path,   'images.txt')
		
		# 1. Read the keys from the text file (fast)
		if not os.path.exists(self.keys_txt_path):
			raise FileNotFoundError("Could not find keys file", self.keys_txt_path)
			
		with open(self.keys_txt_path, 'r') as f:
			# Strip whitespace and ignore empty lines
			self.keys = [line.strip() for line in f if line.strip()]
			
		# 2. LMDB Setup (environment initialized later, lazily, for better multiprocessing support)
		self.env = None
		self.txn = None

		# Define the center crop transform
		self.center_crop = transforms.CenterCrop(self.crop_size)

	#-----------------
	#	Initializes the LMDB environment lazily.
	#	This is crucial when using multiple worker processes in a DataLoader
	#	to avoid file descriptor issues.
	#-----------------
	def _init_lmdb(self):
		# Open the environment in read-only mode, with no write locks for performance
		# (unless user specifically requested locks)
		self.images_env = lmdb.open(self.images_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
		self.images_txn = self.env.begin(write=False)
		self.feat_env = lmdb.open(self.feat_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
		self.feat_txn = self.env.begin(write=False)

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		# 0. Lazy initialization of LMDB env on the first access (important for workers)
		if self.env is None:
			self._init_lmdb()

		# 1. Get the key (filename)
		key_str = self.keys[index]
		key_bytes = key_str.encode('ascii') # Convert string key to bytes for LMDB fetch
		
		# 2. Fetch binary JPEG data from LMDB
		# Use the transaction (txn) to fetch
		jpeg_bin = self.images_txn.get(key_bytes)
		feat_bin = self.feat_txn.get(key_bytes)
		
		# Handle potential missing keys gracefully or raise an error
		# For now, raise an error as the keys file should match the database content.
		if jpeg_bin is None:
			raise KeyError('Key ' + key_str + 'not found in LMDB' + self.images_lmdb_path)
		if feat_bin is None:
			raise KeyError('Key ' + key_str + 'not found in LMDB' + self.feat_lmdb_path)

		# 3. Decompress the JPEG using PIL from memory
		image = Image.open(io.BytesIO(jpeg_bin)).convert('RGB')

		# 4. Apply Center Crop
		image_cropped = self.center_crop(image)

		# Convert image_out and features to a numpy byte arrays
		image_out_np = np.array(image_out, dtype=np.uint8)
		features_np = np.frombuffer(feat_bin, dtype=np.uint8)

		return image_out_np, features_np

if __name__ == "__main__":
	# Path to the folder containing img.lmdb, feat.lmdb, and keys.txt
	FOLDER_PATH = "."
	
	test_data_pipeline(
		dataset_folder=FOLDER_PATH,
		batch_size=8,	  # Start small to monitor memory
		num_workers=4,	 # Adjust based on your CPU cores
		num_batches=10	 # Just a quick smoke test
	)

	#---------------
	# Tests the SA-1B LMDB pipeline with shuffling and timing.
	#---------------
	
	# 1. Construct paths based on your folder-only __init__ logic
	# (Adjust these filenames if they differ in your directory)
	print(f"Initializing dataset from: {dataset_folder}")
	
	# Initialize the dataset
	# Note: Using your folder-based __init__ requirement
	dataset = SA1B_LowLevel_Dataset(dataset_folder)
	
	# 2. Setup DataLoader
	# pin_memory=True is recommended when sending data to GPU later,
	# though with uint8 it has less impact than with float32.
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,		  # Essential for training
		num_workers=num_workers,
		drop_last=True		 # Ensures all batches are exactly the same size
	)

	print(f"Dataset size: {len(dataset)}")
	print(f"Batch size: {batch_size}")
	print(f"Workers: {num_workers}")
	print("-" * 50)

	start_time = time.time()
	last_batch_time = start_time

	# Iterate through the dataloader
	for i, (images, features) in enumerate(dataloader):
		if i >= num_batches:
			break
			
		current_time = time.time()
		batch_duration = current_time - last_batch_time
		total_elapsed = current_time - start_time
		
		# Print shapes and dtypes to verify the 'collate' logic
		# Images should be [B, 1500, 1500, 3], Features [B, F]
		print(f"Batch {i+1}/{num_batches}")
		print(f"  Images:   Shape {images.shape} | Dtype {images.dtype}")
		print(f"  Features: Shape {features.shape} | Dtype {features.dtype}")
		print(f"  Timing:   Batch: {batch_duration:.4f}s | Total: {total_elapsed:.4f}s")
		print("-" * 30)
		
		last_batch_time = current_time


	final_time = time.time()
	avg_speed = (num_batches * batch_size) / (final_time - start_time)
	
	print("\nTest Complete!")
	print(f"Total time for {num_batches} batches: {final_time - start_time:.2f}s")
	print(f"Average throughput: {avg_speed:.2f} images/sec")
