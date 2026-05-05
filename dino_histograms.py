


def mkdir(path):
	try:
		os.makedirs(path, exist_ok=True)
	except Exception as e:
		print("Could not create directory:", path, e)


def main():

	import time
	import blockreader_sa1b
	import bicubic
	import sys
	import torch
	import torch.nn.functional as F

	sys.path.insert(0, '../DCF-Copula/cuda-version/')
	import histogrammer as hgm

	print('---------------------------------------------')
	print(' Initialize Histogrammer')
	print('---------------------------------------------')

	device='cuda'
	nFilters = 384
	nBins = 1000
	hg = hgm.Histogrammer(device, nFilters, nBins)
	hg_epochs = 2



	print('---------------------------------------------')
	print(' Initialize Blockreader')
	print('---------------------------------------------')
	batch_size = 64
	#dataset_folder = "/nvme0/sa1b"
	dataset_folder = "/data/developers/sa1b"
	print(f"Initializing dataset from: {dataset_folder}")

	# Initialize the dataset
	# Note: Using your folder-based __init__ requirement
	reader = blockreader_sa1b.SA1B_DINO_blockreader(dataset_folder, batch_size)
	n_img   = reader.len()
	#n_img   = 150
	n_batch = (n_img+batch_size-1) // batch_size
	n_batch = 5              # Hack small subset for now
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
	print(' Make output folders')
	print('---------------------------------------------')
	outdir = 'dino_histograms'
	mkdir(outdir)


	with torch.no_grad():
		print('---------------------------------------------')
		print(' For every epoch')
		print('---------------------------------------------')
		for epoch in range(n_epoch):



			print('-----------------')
			print(' For every batch')
			print('-----------------')
			reader.reset()
			hg.begin_epoch()

			# Iterate through the dataloader
			for i in range(n_batch):

				#----------------
				# Read the images/features from lmdb
				#----------------
				images, features_np, keys = reader.read_batch()

				if features_np.shape[0]!=batch_size:    # NO partial batches
					continue

				features = torch.tensor(features_np, dtype=torch.uint8, device='cuda', requires_grad=False)

				# Restore the features using bicubic
				bcb.restore_uint8_features(features)


				#----------------
				# Run the histogrammer on the batch
				#----------------
				N,H,W,C = features.shape
				print('N', N, 'H', H, 'W', W, 'C', C)
				NHW = N*H*W
				batch = torch.reshape(features)
				hg.add_batch(batch)

				#----------------
				# Print timing info
				#----------------
				current_time = time.time()
				batch_duration = current_time - last_batch_time
				total_elapsed = current_time - start_time

				print(f"Batch {i+1}/{n_batch}")
				print(f"  Images:   Shape {images.shape} | Dtype {images.dtype}")
				print(f"  Features: Shape {features.shape} | Dtype {features.dtype}  |  Type {type(features)}")
				print(f"  Timing:   Batch: {batch_duration:.4f}s | Total: {total_elapsed:.4f}s")
				print("-" * 30, flush=True)

				last_batch_time = current_time

			hg.end_epoch()


			# print the histogram
			for b in range(hg.nBins):
				print('bin (%.3f %.3f)   count %d ' % (hg.steps[0,b], hg.steps[0,b+1], hg.histogram[0,b]))


			# print the histogram
			for b in range(hg.nBins+1):
				print('lo (%.3f %.3f)   hi (%.3f %.3f)   guess  (%.3f %.3f)' % (
					hg.quantiles_lo_x[0,b], hg.quantiles_lo_y[0,b],
					hg.quantiles_hi_x[0,b], hg.quantiles_hi_y[0,b],
					hg.quantiles[0,b],   (float)( b / hg.nBins )) )



	final_time = time.time()
	avg_speed = (n_batch * batch_size) / (final_time - start_time)

	print("\nTest Complete!")
	print(f"Total time for {n_batch} batches: {final_time - start_time:.2f}s")
	print(f"Average throughput: {avg_speed:.2f} images/sec")

	print('Success!')

if __name__ == "__main__":
	main()

		# Debugging printing out raw values
		#idx = 0
		#for y in range(64):
		#	for x in range(64):
		#		print('%d ' % features_np[0, 12 + 384*2 + 384*2 + idx], end='')
		#		idx+=384
		#	print('')


