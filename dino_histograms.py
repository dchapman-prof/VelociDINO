# system imports
import os
import sys
import time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Must be called before pyplot
import matplotlib.pyplot as plt

# project imports
import blockreader_sa1b
import bicubic

# project dependency
sys.path.insert(0, '../DCF-Copula/cuda-version/')
import histogrammer as hgm



def mkdir(path):
	try:
		os.makedirs(path, exist_ok=True)
	except Exception as e:
		print("Could not create directory:", path, e)


def main():




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
	batch_size = 128
	#dataset_folder = "/nvme0/sa1b"
	dataset_folder = "/data/developers/sa1b"
	print(f"Initializing dataset from: {dataset_folder}")

	# Initialize the dataset
	# Note: Using your folder-based __init__ requirement
	reader = blockreader_sa1b.SA1B_DINO_blockreader(dataset_folder, batch_size)
	n_img   = reader.len()
	#n_img   = 150
	n_batch = (n_img+batch_size-1) // batch_size
	#n_batch = 5              # Hack small subset for now
	#n_batch = n_img // batch_size    # NO partial batches
	#n_epoch = 90

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
		for epoch in range(hg_epochs):



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
				features = bcb.mask_data_raw

				#----------------
				# Run the histogrammer on the batch
				#----------------

				N,H,W,C = features.shape
				print('features   N', N, 'H', H, 'W', W, 'C', C, 'dtype', features.dtype)
				NHW = N*H*W
				batch = torch.reshape(features, (NHW,C))
				print('batch', batch.shape, batch.dtype)
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






		print('---------------------------------------------')
		print(' Run QDistrib')
		print('---------------------------------------------')
		if hg.cdf == None:
			hg.calc_cdf()


		print('extract upper cdf')
		X_upper = hg.steps
		cdf_upper = hg.cdf

		print('extract lower cdf')
		X_lower   = (  0.0 - torch.flip(X_upper, dims=(1,))    ).contiguous()
		cdf_lower = (  1.0 - torch.flip(cdf_upper, dims=(1,))  ).contiguous()


		print('fit upper distribution')
		qd_upper = hgm.QDistr(nFilters, nBins,
			Y0 = 0.99,
			distrib_type=hgm.QDISTR_WEIBULL,
			metric=hgm.QD_METRIC_WASSERSTEIN)

		qd_upper.fit(X_upper, cdf_upper)

		print('fit lower distribution')
		qd_lower = hgm.QDistr(nFilters, nBins,
			Y0 = 0.99,
			distrib_type=hgm.QDISTR_WEIBULL,
			metric=hgm.QD_METRIC_WASSERSTEIN)

		qd_lower.fit(X_lower, cdf_lower)


		print('write out upper distribution')
		outpath = '%s/weibull_upper.csv' % outdir
		print(outpath)
		fcsv = open(outpath,'w')
		fcsv.write('chan\ttheta\tlamda\tk\twas_dist\tkl_diver\tN0\n')
		for c in range(qd_upper.nFilters):
			wei_lamda =     qd_upper.distr[c, qd_upper.DISTR_PARAM1]
			wei_k     =     qd_upper.distr[c, qd_upper.DISTR_PARAM2]
			wei_theta =     1.0 / wei_k
			was_dist  =     qd_upper.distr[c, qd_upper.DISTR_WAS_DIST]
			kl_diver  =     qd_upper.distr[c, qd_upper.DISTR_KL_DIVER]
			N0        = int(qd_upper.distr[c, qd_upper.DISTR_N0])
			print('filter %d  weibull   theta %.6f  lamda %.6f  k %.6f   was %.6f  kl %.6f' %
			(c,  wei_theta, wei_lamda, wei_k, was_dist, kl_diver))

			fcsv.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\n' %
				(c, wei_theta, wei_lamda, wei_k, was_dist, kl_diver, N0) )

			outpath = '%s/weibull_upper_%03d.png' % (outdir, c)
			print(outpath)
			X_np,Y_np    = hgm.PlotDistrib(X_upper[c],qd_upper.Y[c],N0=N0)
			X_np,Yhat_np = hgm.PlotDistrib(X_upper[c],qd_upper.Yhat[c],N0=N0)
			plt.figure()
			plt.plot(X_np,Y_np)
			plt.plot(X_np,Yhat_np)
			title = 'Feature %d theta %.6f  lam %.6f  k %.6f' %  (c, wei_theta, wei_lamda, wei_k)
			plt.title(title)
			plt.savefig(outpath)
			#plt.show()
		fcsv.close()


		print('write out lower distribution')
		outpath = '%s/weibull_lower.csv' % outdir
		print(outpath)
		fcsv = open(outpath,'w')
		fcsv.write('chan\ttheta\tlamda\tk\twas_dist\tkl_diver\tN0\n')
		for c in range(qd_lower.nFilters):
			wei_lamda =     qd_lower.distr[c, qd_lower.DISTR_PARAM1]
			wei_k     =     qd_lower.distr[c, qd_lower.DISTR_PARAM2]
			wei_theta =     1.0 / wei_k
			was_dist  =     qd_lower.distr[c, qd_lower.DISTR_WAS_DIST]
			kl_diver  =     qd_lower.distr[c, qd_lower.DISTR_KL_DIVER]
			N0        = int(qd_lower.distr[c, qd_lower.DISTR_N0])
			print('filter %d  weibull   theta %.6f  lamda %.6f  k %.6f   was %.6f  kl %.6f' %
			(c,  wei_theta, wei_lamda, wei_k, was_dist, kl_diver))

			fcsv.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\n' %
				(c, wei_theta, wei_lamda, wei_k, was_dist, kl_diver, N0) )

			outpath = '%s/weibull_lower_%03d.png' % (outdir, c)
			print(outpath)
			X_np,Y_np    = hgm.PlotDistrib(X_lower[c],qd_lower.Y[c],N0=N0)
			X_np,Yhat_np = hgm.PlotDistrib(X_lower[c],qd_lower.Yhat[c],N0=N0)
			plt.figure()
			plt.plot(X_np,Y_np)
			plt.plot(X_np,Yhat_np)
			title = 'Feature %d theta %.6f  lam %.6f  k %.6f' %  (c, wei_theta, wei_lamda, wei_k)
			plt.title(title)
			plt.savefig(outpath)
			#plt.show()
		fcsv.close()



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


