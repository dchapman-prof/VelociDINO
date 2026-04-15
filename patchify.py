import numpy as np
import zlib
import torch
import torchvision.transforms.v2.functional as F_tran
import sys

#-------------------------------
# Center crop and standardize to sXY length
#-------------------------------
def StandardImg(img, sXY=896):

	# Image dimensions
	rows,cols,chan = img.shape
	print('rows', rows, 'cols', cols, 'chan', chan)
	
	# Center Crop
	if rows<cols:
		side = rows
		y0 = 0
		y1 = rows
		x0 = cols//2 - side//2
		x1 = x0 + side
	else:
		side = cols
		x0 = 0
		x1 = cols
		y0 = rows//2 - side//2
		y1 = y0 + side
	print('x0', x0, 'x1', x1, 'y0', y0, 'y1', y1)
	img = img[y0:y1, x0:x1, 0:chan]
	
	# Resize to 896
	img = torch.permute(img, (2,0,1))    # chan,rows,cols
	print('img.shape', img.shape)
	img = F_tran.resize(
		img, 
		size=(sXY, sXY), 
		interpolation=F_tran.InterpolationMode.BICUBIC, 
		antialias=True)
	print('img.shape', img.shape)
	img = torch.permute(img, (1,2,0))     # rows,cols,chan
	print('img.shape', img.shape)
	return img
	
#-------------------------------
# Patchify       sYxsXxC  ---->   
#           PYxPX overlapping of HxWxC
#-------------------------------
def Patchify(img, PY, PX, device='cuda'):
	
	# What is the size of the image
	sY,sX,C = img.shape
	
	# What is the size of the patches ?
	H__2 = (sY//(PY+1))
	W__2 = (sX//(PX+1))
	H = 2 * H__2
	W = 2 * W__2
	
	# Create the patches
	patches = torch.zeros((PY,PX,H,W,C), device=device, dtype=torch.float32)
	
	# Copy the patches
	y0=0;  y1=H
	for y in range(PY):
		x0=0;  x1=W
		for x in range(PX):
			patches[y,x,:,:,:] = img[y0:y1, x0:x1, :]
			x0 += W__2;  x1 += W__2
		y0 += H__2;  y1 += H__2
	
	return patches
	
#-------------------------------
# Stitch  7x7 overlapping of 14x14xC
#          ---->  56x56xC
#-------------------------------
def Stitch(feat_patches, device='cuda'):
	PY,PX,H,W,C = feat_patches.shape
	H__2 = H // 2
	W__2 = W // 2
	
	#---
	# Construct triangular kernel
	#---
	
	# Construct a triangular kernel  (height)
	Hm = (H+1)//2
	H_ker = torch.zeros((H,), device=device, dtype=torch.float32)
	H_ker[0:Hm] = torch.arange(1,Hm+1,1)
	H_ker[H-Hm:H] = torch.arange(Hm,0,-1)

	# Construct a triangular kernel  (width)
	Wm = (H+1)//2
	W_ker = torch.zeros((W,), dtype=torch.float32)
	W_ker[0:Wm] = torch.arange(1,Wm+1,1)
	W_ker[W-Wm:W] = torch.arange(Wm,0,-1)
	
	# Create the 2D triangular kernel
	ker = torch.outer(H_ker, W_ker)
	
	#----
	# Allocate output array
	#----
	sY = (H // 2) * (PY + 1)
	sX = (W // 2) * (PX + 1)
	total = torch.zeros( (sY,sX,C), device=device, dtype=torch.float32)
	count = torch.zeros( (sY,sX,C), device=device, dtype=torch.float32)
	
	#----
	# Stitch the patches
	#----
	y0=0;  y1=H
	for y in range(PY):
		x0=0;  x1=W
		for x in range(PX):
			total[y0:y1, x0:x1, :] += ker * feat_patches[y,x,:,:,:]
			count[y0:y1, x0:x1, :] += ker
			x0 += W__2;  x1 += W__2
		y0 += H__2;  y1 += H__2
	
	#----
	# Average the patches
	#----
	
	
	
#-------------------------------
# Input    X  (HxWxC  fp32  numpy)   minval (float)  maxval (float)
# Output   header payload  (bytearray)
#   header  H (int32) W (int32) C (int32)  minval (float32)  maxval (float32)
#   payload    HxWxC  (zlib uint16 difference)
#-------------------------------
def Compress3D(X, minval, maxval):

	H,W,C = X.shape
	
	# Convert to 16 bit
	X16 = 65535 * (X - minval) / (maxval - minval)
	X16[X16 < 0] = 0
	X16[X16 > 65535] = 65535
	X16 = X16.astype(np.uint16)
	
	# Convert to difference
	X16[1:,:,:] = X16[1:,:,:] - X16[:-1,:,:]
	
	# zlib compress it
	zipped = zlib.compress(X16.tobytes(), level=1)
	
	# append the header
	ba = bytearray()
	HWC    = np.array([H,W,C], dtype=np.int32)
	minmax = np.array([minval,maxval], dtype=np.float32)
	ba.extend(HWC.tobytes())
	ba.extend(minmax.tobytes())
	
	# append the payload
	ba.extend(zipped)
	
	return ba


#-------------------------------
# Input   header payload  (bytearray)
#   header  H (int32) W (int32) C (int32)  minval (float32)  maxval (float32)
#   payload    HxWxC  (zlib uint16 difference)
# Output    X  (HxWxC  fp32  numpy)   minval (float)  maxval (float)
#-------------------------------	
def Decompress3D(ba):

	b0 = 0
	b1 = 12
	b2 = 20
	b3 = len(ba)
	
	# Read the dimensions of the data
	HWC    = np.frombuffer(ba[b0:b1], dtype=np.int32)
	minmax = np.frombuffer(ba[b1:b2], dtype=np.float32)
	H = HWC[0]
	W = HWC[1]
	C = HWC[2]
	minval = minmax[0]
	maxval = minmax[1]
	
	# Decompress the data
	unzipped = zlib.decompress(ba[b2:b3], bufsize=H*W*C*2)
	
	# Create the numpy array
	X16 = np.frombuffer(unzipped, dtype=np.uint16)
	X16 = np.reshape(X16, (H,W,C))
	
	# Undo difference encoding
	X16 = np.cumsum(X16, axis=0, dtype=np.uint16)
	
	# Upsample to fp32
	X = X16.astype(np.float32)
	X = minval + (maxval-minval)*X/65535
	
	return X

#-------------------------------
# Input    X  (HxWxC  fp32  numpy)
# Output   header minvals maxvals payload  (bytearray)
#   header  H (int32) W (int32) C (int32)  minval (float32)  maxval (float32)
#   minvals (C fp16 numpy)
#   maxvals (C fp16 numpy)
#   payload    HxWxC  (zlib uint8 difference)
#-------------------------------
def Compress8bit(X):

	# Convert to fp16
	H,W,C = X.shape
	HW = H*W
	X = X.astype(np.float16)
	
	# Calculate minvals and maxvals
	minvals = np.min(X, axis=(0,1), keepdims=True)   # (1,1,C)
	maxvals = np.max(X, axis=(0,1), keepdims=True)   # (1,1,C)
	
	print('  minvals', minvals.shape, minvals.nbytes)
	print('  maxvals', maxvals.shape, maxvals.nbytes)
	
	# Convert to uint8
	X8 = 255.0 * (X-minvals) / (maxvals-minvals + 0.00000001)
	X8[X8<0.0] = 0.0
	X8[X8>255.0] = 255.0
	X8 = X8.astype(np.uint8)
	
	# Convert to difference
	X8[1:,:,:] = X8[1:,:,:] - X8[:-1,:,:]

	# zlib compress it
	unzipped = X8.tobytes()
	zipped = zlib.compress(unzipped, level=1)
	#input('enter')
	
	# append the header
	ba = bytearray()
	HWC    = np.array([H,W,C], dtype=np.int32)
	ba.extend(HWC.tobytes())

	# append minvals and maxvals
	ba.extend(minvals.tobytes())
	ba.extend(maxvals.tobytes())
	
	# append the payload
	ba.extend(zipped)
	
	return ba

#-------------------------------
# Output    X  (HxWxC  fp32  numpy)
# Input   header minvals maxvals payload  (bytearray)
#   header  H (int32) W (int32) C (int32)  minval (float32)  maxval (float32)
#   minvals (C fp16 numpy)
#   maxvals (C fp16 numpy)
#   payload    HxWxC  (zlib uint8 difference)
#-------------------------------	
def Decompress8bit(ba):
	
	# Header byte offsets
	b0 = 0
	b1 = 12
	
	# Read the dimensions of the data
	HWC    = np.frombuffer(ba[b0:b1], dtype=np.int32)
	H = HWC[0]
	W = HWC[1]
	C = HWC[2]
	
	# Payload byte offsets
	b2 = b1 + 2*C
	b3 = b2 + 2*C
	b4 = len(ba)
	
	# Read the minvals
	minvals = np.frombuffer(ba[b1:b2], dtype=np.float16)
	maxvals = np.frombuffer(ba[b2:b3], dtype=np.float16)
	minvals = np.reshape(minvals, (1,1,C))
	maxvals = np.reshape(maxvals, (1,1,C))
	
	# Decompress the data
	unzipped = zlib.decompress(ba[b3:b4], bufsize=H*W*C)
	
	# Create the numpy array
	X8 = np.frombuffer(unzipped, dtype=np.uint8)
	X8 = np.reshape(X8, (H,W,C))
	
	# Undo difference encoding
	X8 = np.cumsum(X8, axis=0, dtype=np.uint8)
	
	# Upsample to fp32
	X = X8.astype(np.float32)
	X = minvals + (maxvals-minvals)*X/255.0
	
	return X
