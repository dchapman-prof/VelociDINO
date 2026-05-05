import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalMSAConvBlock(nn.Module):
	def __init__(self, C=32, M=4):
		super().__init__()
		self.C = C	 # number of channels
		self.M = M	 # number of heads
		
		if (C%H != 0):
			raise Exception('Error: channels must be divisible by heads')
		self.d = C // H
		
		# Shared QK projection (symmetric)
		self.qkv_proj = nn.Conv2d(C, C, 1)
				
		# Batch Norm
		self.norm1 = nn.BatchNorm2d(C)
		self.norm2 = nn.BatchNorm2d(C)
		
		# Conv2d MLP stage
		self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')
		self.conv2 = nn.Conv2d(channels, channels, 3, padding='same')
		
		# Temperature scaling
		self.temp  = nn.Conv2d(channels, channels, 1, groups=channels)
			
	def local_msa(self, x):
		B, C, H, W = x.shape
		d = self.d
		
		# Shared QKV projection
		v = self.qkv_proj(x)  # [B, C, H, W]
		
		# Split values to obtain q,k
		v  = torch.reshape(qkv,  (B, M, d, H, W))
		
		# L2 normalization
		qk_norm = torch.sum(v*v, dim=2)
		qk_norm = torch.rsqrt(qk_norm + 0.000001)
		qk = v * qk_norm	# B M d H W
	
		# Calculate qk dependence at all offsets
		score_	 = torch.ones((B,M,1,H,W,1), device=qk.device,requires_grad=False,dtype=torch.float32)		 # [B M  H  W]
		score_lr   = (qk[:,:,:,:,0:(W-1)]	   * qk[:,:,:,:,1:W]).sum(dim=2)	   # [B M  H  W-1]
		score_bt   = (qk[:,:,:,0:(H-1),:]	   * qk[:,:,:,1:H,:]).sum(dim=2)	   # [B M H-1  W ]
		score_d1   = (qk[:,:,:,0:(H-1),0:(W-1)] * qk[:,:,:,1:H,1:W]).sum(dim=2)	 # [B M H-1 W-1]
		score_d2   = (qk[:,:,:,0:(H-1),1:W]	 * qk[:,:,:,1:H,0:(W-1)]).sum(dim=2) # [B M H-1 W-1]
	
		# Pad to get the scores in a 3x3 neighborhood
		score_l	= torch.pad(score_lr, (1,0,0,0))   # [B M H W]
		score_r	= torch.pad(score_lr, (0,1,0,0))
		score_b	= torch.pad(score_bt, (0,0,1,0))
		score_t	= torch.pad(score_bt, (0,0,0,1))
		score_bl   = torch.pad(score_d1, (1,0,1,0))
		score_tl   = torch.pad(score_d2, (1,0,0,1))
		score_br   = torch.pad(score_d2, (0,1,1,0))
		score_tr   = torch.pad(score_d1, (0,1,0,1))
		score = torch.cat(
			score_,
			torch.reshape(score_l,  (B,M,1,H,W,1)),
			torch.reshape(score_r,  (B,M,1,H,W,1)),
			torch.reshape(score_b,  (B,M,1,H,W,1)),
			torch.reshape(score_t,  (B,M,1,H,W,1)),
			torch.reshape(score_bl, (B,M,1,H,W,1)),
			torch.reshape(score_tl, (B,M,1,H,W,1)),
			torch.reshape(score_br, (B,M,1,H,W,1)),
			torch.reshape(score_tr, (B,M,1,H,W,1)),
				dim=5)								# [B M 1 H W 9]
		
		# Shift the values around (there has to be a better way...)
		v = torch.reshape(v, (B,C,H,W))
		v_l  = torch.pad(v[:,:,:,0:(W-1)], (1,0,0,0))
		v_r  = torch.pad(v[:,:,:,1:W],	 (0,1,0,0))
		v_b  = torch.pad(v[:,:,0:(H-1),:], (0,0,1,0))
		v_t  = torch.pad(v[:,:,1:H,:],	 (0,0,0,1))
		v_bl = torch.pad(v[:,:,0:(H-1),0:(W-1)], (1,0,1,0))
		v_tl = torch.pad(v[:,:,0:H,	0:(W-1)], (1,0,0,1))
		v_br = torch.pad(v[:,:,0:(H-1),1:W],	 (0,1,1,0))
		v_tr = torch.pad(v[:,:,1:H,	1:W],	 (0,1,0,1))  # [B C H W]
		v_stack = torch.cat(
			torch.reshape(v,	(B,M,d,H,W,1)),
			torch.reshape(v_l,  (B,M,d,H,W,1)),
			torch.reshape(v_r,  (B,M,d,H,W,1)),
			torch.reshape(v_b,  (B,M,d,H,W,1)),
			torch.reshape(v_t,  (B,M,d,H,W,1)),
			torch.reshape(v_bl, (B,M,d,H,W,1)),
			torch.reshape(v_tl, (B,M,d,H,W,1)),
			torch.reshape(v_br, (B,M,d,H,W,1)),
			torch.reshape(v_tr, (B,M,d,H,W,1)),
				dim=5)								# [B M d H W 9]
	
		# Perform softmax of scores
		score = F.softmax(score, dim=5)			   # [B M 1 H W 9]
		
		# Weighted sum
		z = (score * v_stack).sum(dim=5)  # [B, M, d, H, W]
		z = torch.reshape(z, (B,C,H,W))
		
		return z
	
	def forward(self, x):
		# MSA stage (like transformer MSA block)
		z = self.local_msa(x)
		
		# Conv stage (like transformer MLP block)
		z = self.conv1(z)
		z = self.norm1(z)
		z = F.relu(z)
		z = self.conv2(z)
		z = self.norm2(z)
		z = self.temp(z)
		
		return z
		

def act_fn(activation_type):
	activation_type = (activation_type or "relu").lower()

	if activation_type == "relu":
		return F.relu
	elif activation_type == "leaky_relu":
		return F.leaky_relu
	else:
		raise ValueError(f"Unsupported activation: {activation_type}")


class ResNetStem(nn.Module):

	def __init__(self, in_ch, out_ch, act):
		super().__init__()

		self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_ch // 2)

		self.conv2 = nn.Conv2d(out_ch // 2, out_ch // 2, 3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_ch // 2)

		self.conv3 = nn.Conv2d(out_ch // 2, out_ch, 3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_ch)

		self.act = act
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.act(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.act(x)

		x = self.pool(x)

		return x

class LocalMSAConvBlock(nn.Module):
	def __init__(self, C=32, M=4):
		super().__init__()
		self.C = C	 # number of channels
		self.M = M	 # number of heads
		
		if (C%M != 0):
			raise Exception('Error: channels must be divisible by heads')
		self.d = C // M
		
		# Shared QK projection (symmetric)
		self.qkv_proj = nn.Conv2d(C, C, 1)
				
		# Batch Norm
		self.norm1 = nn.BatchNorm2d(C)
		self.norm2 = nn.BatchNorm2d(C)
		
		# Conv2d MLP stage
		self.conv1 = nn.Conv2d(C, C, 3, padding='same')
		self.conv2 = nn.Conv2d(C, C, 3, padding='same')
		
		# Temperature scaling
		self.temp  = nn.Conv2d(C, C, 1, groups=C)
			
	def local_msa(self, x):
		B, C, H, W = x.shape
		M = self.M
		d = self.d
		
		# Shared QKV projection
		v = self.qkv_proj(x)  # [B, C, H, W]
		
		# Split values to obtain q,k
		v  = torch.reshape(v,  (B, M, d, H, W))
		
		#print('x.shape', x.shape)
		#print('v.shape', v.shape)
		
		
		# L2 normalization
		qk_norm = torch.sum(v*v, dim=2, keepdim=True)
		qk_norm = torch.rsqrt(qk_norm + 0.000001)
		#print('qk_norm.shape', qk_norm.shape)
		qk = v * qk_norm	# B M d H W
	
		# Calculate qk dependence at all offsets
		score_	 = torch.ones((B,M,1,H,W,1), device=qk.device, dtype=x.dtype,requires_grad=False)#dtype=torch.float32)		 # [B M  H  W]
		score_lr   = (qk[:,:,:,:,0:(W-1)]	   * qk[:,:,:,:,1:W]).sum(dim=2)	   # [B M  H  W-1]
		score_bt   = (qk[:,:,:,0:(H-1),:]	   * qk[:,:,:,1:H,:]).sum(dim=2)	   # [B M H-1  W ]
		score_d1   = (qk[:,:,:,0:(H-1),0:(W-1)] * qk[:,:,:,1:H,1:W]).sum(dim=2)	 # [B M H-1 W-1]
		score_d2   = (qk[:,:,:,0:(H-1),1:W]	 * qk[:,:,:,1:H,0:(W-1)]).sum(dim=2) # [B M H-1 W-1]
	
		# Pad to get the scores in a 3x3 neighborhood
		score_l	= F.pad(score_lr, (1,0,0,0))   # [B M H W]
		score_r	= F.pad(score_lr, (0,1,0,0))
		score_b	= F.pad(score_bt, (0,0,1,0))
		score_t	= F.pad(score_bt, (0,0,0,1))
		score_bl   = F.pad(score_d1, (1,0,1,0))
		score_tl   = F.pad(score_d2, (1,0,0,1))
		score_br   = F.pad(score_d2, (0,1,1,0))
		score_tr   = F.pad(score_d1, (0,1,0,1))
		score = torch.cat((
			score_,
			torch.reshape(score_l,  (B,M,1,H,W,1)),
			torch.reshape(score_r,  (B,M,1,H,W,1)),
			torch.reshape(score_b,  (B,M,1,H,W,1)),
			torch.reshape(score_t,  (B,M,1,H,W,1)),
			torch.reshape(score_bl, (B,M,1,H,W,1)),
			torch.reshape(score_tl, (B,M,1,H,W,1)),
			torch.reshape(score_br, (B,M,1,H,W,1)),
			torch.reshape(score_tr, (B,M,1,H,W,1))),
				dim=5)								# [B M 1 H W 9]
		
		# Shift the values around (there has to be a better way...)
		v = torch.reshape(v, (B,C,H,W))
		v_l  = F.pad(v[:,:,:,0:(W-1)], (1,0,0,0))
		v_r  = F.pad(v[:,:,:,1:W],	 (0,1,0,0))
		v_b  = F.pad(v[:,:,0:(H-1),:], (0,0,1,0))
		v_t  = F.pad(v[:,:,1:H,:],	 (0,0,0,1))
		v_bl = F.pad(v[:,:,0:(H-1),0:(W-1)], (1,0,1,0))
		v_tl = F.pad(v[:,:,1:H,	0:(W-1)], (1,0,0,1))
		v_br = F.pad(v[:,:,0:(H-1),1:W],	 (0,1,1,0))
		v_tr = F.pad(v[:,:,1:H,	1:W],	 (0,1,0,1))  # [B C H W]
		v_stack = torch.cat((
			torch.reshape(v,	(B,M,d,H,W,1)),
			torch.reshape(v_l,  (B,M,d,H,W,1)),
			torch.reshape(v_r,  (B,M,d,H,W,1)),
			torch.reshape(v_b,  (B,M,d,H,W,1)),
			torch.reshape(v_t,  (B,M,d,H,W,1)),
			torch.reshape(v_bl, (B,M,d,H,W,1)),
			torch.reshape(v_tl, (B,M,d,H,W,1)),
			torch.reshape(v_br, (B,M,d,H,W,1)),
			torch.reshape(v_tr, (B,M,d,H,W,1))),
				dim=5)								# [B M d H W 9]
	
		# Perform softmax of scores
		score = F.softmax(score, dim=5)			   # [B M 1 H W 9]
		
		# Weighted sum
		z = (score * v_stack).sum(dim=5)  # [B, M, d, H, W]
		z = torch.reshape(z, (B,C,H,W))
		
		return z
	
	def forward(self, x):
		# MSA stage (like transformer MSA block)
		z = self.local_msa(x)
		
		# Conv stage (like transformer MLP block)
		z = self.conv1(z)
		z = self.norm1(z)
		z = F.relu(z)
		z = self.conv2(z)
		z = self.norm2(z)
		z = self.temp(z)
		
		return z


class CascadeUnetLayer(nn.Module):

	def __init__(self, baseC=1, baseM=2, H=64, W=64, C_factor=2, M_factor=1, u_depth=5):
		super().__init__()

		self.baseC = baseC
		self.baseM = baseM
		self.H = H
		self.W = W
		self.C_factor = C_factor
		self.M_factor = M_factor
		self.u_depth = u_depth


		self.enc = nn.ModuleList()
		self.dec = nn.ModuleList()
		self.up_proj = nn.ModuleList()

		C_list = []
		M_list = []

		C = baseC
		M = baseM

		for u in range(u_depth):
			C_list.append(C)
			M_list.append(M)

			C *= C_factor
			M *= M_factor

		for u in range(u_depth):
			self.enc.append(LocalMSAConvBlock(C_list[u], M_list[u]))

			if u < u_depth - 1:
				self.dec.append(LocalMSAConvBlock(C_list[u], M_list[u]))

				self.up_proj.append(nn.Conv2d(C_list[u + 1], C_list[u], kernel_size=1, bias=False))
	
	def forward(self, x_pyr):
		u_depth = self.u_depth

		enc_pyr = [None for _ in range(u_depth)]
		dec_pyr = [None for _ in range(u_depth)]

		enc_pyr[0] = self.enc[0](x_pyr[0])

		for u in range(1, u_depth):
			in_vec = x_pyr[u]
			enc_pyr[u] = self.enc[u](in_vec)

		dec_pyr[u_depth - 1] = enc_pyr[u_depth - 1]

		for u in range(u_depth - 2, -1, -1):
			in_vec = F.interpolate(dec_pyr[u + 1],size=enc_pyr[u].shape[-2:],mode="bilinear",align_corners=False)

			in_vec = self.up_proj[u](in_vec)

			in_vec = in_vec + enc_pyr[u]
			dec_pyr[u] = self.dec[u](in_vec)

		return dec_pyr


class Encoder(nn.Module):
	def __init__(self, fullC=384, baseC=1, baseM=2, H=64, W=64, C_factor=2, M_factor=1, u_depth=5):
		super().__init__()

		self.fullC = fullC
		self.baseC = baseC
		self.baseM = baseM
		self.H = H
		self.W = W
		self.C_factor = C_factor
		self.M_factor = M_factor
		self.u_depth = u_depth


		self.enc = nn.ModuleList()

		C_list = []
		M_list = []
		C = baseC
		for u in range(u_depth):
			C_list.append(C)
			M_list.append(M)
			C *= C_factor
			M *= M_factor

		for u in range(u_depth):
			self.enc.append(LocalMSAConvBlock(C_list[u], M_list[u]))

			if u < u_depth - 1:
				self.dec.append(LocalMSAConvBlock(C_list[u], M_list[u]))

				self.up_proj.append(nn.Conv2d(C_list[u + 1], C_list[u], kernel_size=1, bias=False))
				

	
class LinearLens(nn.Module):

	def __init__(self, outC=384, baseC=16, C_factor=2, u_depth0=2, u_depth1=5):
		super().__init__()
	
		self.baseC	= baseC
		self.C_factor = C_factor
		self.u_depth0 = u_depth0
		self.u_depth1 = u_depth1
		
		
		C_list = []

		C = baseC
		for u in range(u_depth1):
			C_list.append(C)
			C *= C_factor

		self.up_proj = nn.ModuleList()

		for u in range(u_depth1 - 1):
			self.up_proj.append(nn.Conv2d(C_list[u + 1], C_list[u], kernel_size=1, bias=False))

		self.proj = nn.Conv2d(C_list[u_depth0], outC, 1)



	def forward(self, x_pyr):

		u_depth1 = self.u_depth1
		u_depth0 = self.u_depth0

		in_vec = x_pyr[u_depth1 - 1]

		for u in range(u_depth1 - 2, u_depth0 - 1, -1):
		
			in_vec = F.interpolate(in_vec,size=x_pyr[u].shape[-2:],mode="bilinear",align_corners=False)
			in_vec = self.up_proj[u](in_vec)
			in_vec = in_vec + x_pyr[u]
		out_vec = self.proj(in_vec)

		return out_vec


class VelociNet(nn.Module):
	
	def __init__(self, in_shape=(3,256,256), baseH=64, baseW=64, n_layer=10, outC=384, baseC=16, baseM=4, C_factor=2, M_factor=1, u_depth0=2, u_depth1=5):
		super().__init__()
	
		self.in_shape = in_shape
		self.baseH	= baseH
		self.baseW	= baseW
		self.n_layer  = n_layer
		self.outC	 = outC
		self.baseC	= baseC
		self.baseM	= baseM
		self.C_factor = C_factor
		self.M_factor = M_factor
		self.u_depth0 = u_depth0
		self.u_depth1 = u_depth1	

		self.layers = nn.ModuleList()
		#self.stem   = # add stem layer
		act = act_fn("relu")
		inC = in_shape[0]

		self.stem = ResNetStem(in_ch = inC, out_ch = baseC, act = act)
		
		self.pyr_proj = nn.ModuleList()

		C = baseC

		for u in range(1, u_depth1):
			
			self.pyr_proj.append(nn.Conv2d(C, C * C_factor, kernel_size=1, bias=False))

			C *= C_factor
			
		self.lens   = LinearLens(outC=outC,baseC=baseC,C_factor=C_factor,u_depth0=u_depth0,u_depth1=u_depth1)
	
		for i in range(n_layer):
			self.layers.append( CascadeUnetLayer(baseC=baseC, baseM=baseM, H=baseH, W=baseW, C_factor=C_factor, M_factor=M_factor, u_depth=u_depth1) )


	def forward(self, x, return_featvec=False, detach_featvec=True):

		if return_featvec:
			featvec = []

		#-------
		# Construct the residual stream
		#-------
		stream = []
		stream.append(self.stem(x))

		N = x.shape[0]
		C = self.baseC
		H = self.baseH
		W = self.baseW

		for u in range(1, self.u_depth1):
			C = C * self.C_factor
			H = H // 2
			W = W // 2

			z = torch.zeros(
				(N, C, H, W),
				device=x.device,
				dtype=x.dtype,
				requires_grad=False
			)

			stream.append(z)

		#--------
		# Run through each cascade
		#--------
		for i in range(self.n_layer):

			out_stream = self.layers[i](stream)

			# Skip connection
			for u in range(self.u_depth1):
				out_stream[u] = out_stream[u] + stream[u]

			stream = out_stream

			# Featvec
			if return_featvec:
				feat_i = self.lens(stream)

				if detach_featvec:
					feat_i = feat_i.detach()

				featvec.append(feat_i)

		#--------+
		# Final lens
		#--------
		out = self.lens(stream)

		if return_featvec:
			return out, featvec

		return out







#-----------------------------------------------
#-----------------------------------------------
# Laplacian Transform (for auto-encoder)
#-----------------------------------------------
#-----------------------------------------------
def LaplacianTranform(x, u_depth=5):
	
	# Obtain dimensions
	N,C,H,W = x.shape
	
	# Peform downsampling
	r = []   # residual maps
	z = x
	for u in range(0, u_depth):
		
		# Calculate residual
		z_down = F.avg_pool2d(z, kernel_size=2, stride=2, count_include_pad=False)
		z_up   = F.interpolate(z_down, scale_factor=2.0, mode='bilinear', align_corners=False)
		residual = z - z_up
		r.append(residual)
		
		# Downsample the working image
		z = z_down
		
	# throw the last image in the residual set
	r.append(z)
	return r


def InverseLaplacianTransform(r):
	
	u_depth = len(r)
	
	z = r[u_depth-1]
	for u in range(u_depth-1, -1, -1):
		z_down = z
		z_up   = F.interpolate(z_down, scale_factor=2.0, mode='bilinear', align_corners=False)
		z = r[u] + z_up

	return z


#-----------------------------------------------
#-----------------------------------------------
# Shuffle 1x1 convolution for linear encoder projection
#-----------------------------------------------
#-----------------------------------------------
def channel_shuffle(x, groups=4):
	N,C,H,W = x.size()
	channels_per_group = C // groups
	x = x.view(N, groups, channels_per_group, H, W)
	x = x.transpose(1, 2).contiguous()
	x = x.view(N,C,H,W)
	return x

class ShuffleConv1x1(nn.Module):
	
	def __init__(self, inC=16, midC=256, outC=768, groups=4):
		super().__init__()
		self.inC = inC
		self.midC = midC
		self.outC = outC
		self.groups = groups
		self.conv1 = nn.Conv2d(inC, midC, kernel_size=1, groups=groups, bias=False)
		self.conv2 = nn.Conv2d(midC, outC, kernel_size=1, groups=4, bias=False)

	def forward(self, x):
		x = self.conv1(x)
		x = channel_shuffle(x, groups=self.groups)
		x = self.conv2(x)
		return x

#-----------------------------------------------
#-----------------------------------------------
# Encoder using Laplacing Transform + 
#  Pointwise gating (Overcomplete Dynamic Coordinate Scaling)
#-----------------------------------------------
#-----------------------------------------------

class ResidualOverproj(nn.Module):
	
	def __init__(self, inC=384, midC=768, overC=768, groups=4):
		super().__init__()
		
		self.conv3x3  = nn.Conv2d(inC, inC, 3, groups=inC, padding='same')
		self.overproj = ShuffleConv1x1(inC,midC,overC,groups)

	def forward(self, x):
		
		x = self.conv3x3(x)
		x = self.overproj(x)
		return x


def BlowUpChannels(x, overscale):
	N,C,H,W = x.shape
	x = torch.reshape(x, (N, 1, C, H, W))
	x = x.repeat((1,overscale,1,1,1))
	x = torch.reshape(x, (N,overscale*C,H,W)).contiguous()
	return x

def BlowDownChannels(x, overscale):
	N,C,H,W = x.shape
	C = C // overscale
	x = torch.reshape(x, (N, overscale, C, H, W))
	x = torch.mean   (x, dim=1)
	return x

#-------------------------
#
#  input Laplace transform
#      r0[64 64 384]  r1[32 32 384]   r2[16 16 384]   r3[8 8 384]   r4[4 4 384]   z5[2 2 384]
#
#  output Encoded features
#      e0[64 64 16]   e1[32 32 32]    e2[16 16 64]    e3[8 8 128]   e4[4 4 256]   z5[2 2 384]
# 
#-------------------------
class Encoder(nn.Module):
	
	def __init__(self, inC=384, baseC=16, C_factor=2, u_depth=5, groups=4, overscale=2):
		super().__init__()
		self.inC = inC
		self.baseC = baseC
		self.u_depth = u_depth
		self.groups = groups
		self.overscale = overscale
		self.C_factor = C_factor
		
		self.downproj      = nn.ModuleList()
		self.scale_weights = nn.ModuleList()
		
		# For each layer in u_depth
		overC = inC * overscale
		outC = baseC
		for u in range(u_depth):

			# Convert inC into overscale   project 384->768 with 3x3 conv
			self.scale_weights.append(  ResidualOverproj(inC,overC,overC,groups)  )   

			# Calculate basis dimensions  768 -> 16 (depth dependent)
			self.downproj.append(  ShuffleConv1x1(overC, inC, outC, groups)  )
			
			outC *= C_factor
	
	# x is a laplace transform of input
	def forward(self, x):
		
		overC = self.inC * self.overscale
		
		# Start at the end of the pyramid
		outC = self.baseC
		for u in range(self.u_depth-1):
			outC *= self.C_factor

		# Make an empty pyramid
		enc_pyr = []
		for u in range(self.u_depth+1):
			enc_pyr.append( None )

		
		# Perform the inverse pyramid
		z_down = x[self.u_depth]
		enc_pyr[self.u_depth] = z_down
		for u in range(self.u_depth-1, -1, -1):

			# Calculate scaled weights
			weights = self.scale_weights[u](z_down)
			weights = F.sigmoid(weights)
			weights = F.interpolate(weights, scale_factor=2.0, mode='bilinear', align_corners=False)
			
			# Blow up the current layer (to overscale)
			r = BlowUpChannels(x[u], self.overscale)
			
			# Multiply by scaled weights
			r = r*weights
			
			# Downproject channels  768 -> 16
			e = self.downproj[u](r)
			
			# Put into pyramid
			enc_pyr[u] = e
			
			# Next z
			z_down = F.interpolate(z_down, scale_factor=2.0, mode='bilinear', align_corners=False)
			z_down = z_down + x[u]   # add residual
			
		
		return enc_pyr
		


#-------------------------
#
#  input Encoded features
#      e0[64 64 16]   e1[32 32 32]    e2[16 16 64]    e3[8 8 128]   e4[4 4 256]   z5[2 2 384]
# 
#  output Laplace transform
#      r0[64 64 384]  r1[32 32 384]   r2[16 16 384]   r3[8 8 384]   r4[4 4 384]   z5[2 2 384]
#
#-------------------------
class Decoder(nn.Module):
	
	def __init__(self, inC=384, baseC=16, C_factor=2, u_depth=5, groups=4, overscale=2):
		super().__init__()
		self.inC = inC
		self.baseC = baseC
		self.u_depth = u_depth
		self.groups = groups
		self.overscale = overscale
		self.C_factor = C_factor
		
		self.upproj        = nn.ModuleList()
		self.scale_weights = nn.ModuleList()
		
		# For each layer in u_depth
		overC = inC * overscale
		outC = baseC
		for u in range(u_depth):
	
			# Convert inC into overscale   project 384->768 with 3x3 conv
			self.scale_weights.append(  ResidualOverproj(inC,overC,overC,groups)  )   

			# Calculate basis dimensions  16 -> 768 (depth dependent)
			self.upproj.append(  ShuffleConv1x1(outC, inC, overC, groups)  )
			
			outC *= C_factor
	
	
	def forward(self, e):
		
		overC = self.inC * self.overscale
		
		# Start at the end of the pyramid
		outC = self.baseC
		for u in range(self.u_depth-1):
			outC *= self.C_factor

		# Make an empty pyramid
		dec_pyr = []
		for u in range(self.u_depth+1):
			dec_pyr.append( None )

		
		# Perform the inverse pyramid
		z_down = e[self.u_depth]
		dec_pyr[self.u_depth] = z_down
		for u in range(self.u_depth-1, -1, -1):
		
			# Calculate scaled weights
			weights = self.scale_weights[u](z_down)
			weights = F.sigmoid(weights)
			weights = F.interpolate(weights, scale_factor=2.0, mode='bilinear', align_corners=False)
			
			# Upproject channels  16 -> 768
			r = self.upproj[u](e[u])
		
			# Multiply by scaled weights
			r = r*weights
			
			# Blow Down Channels
			r = BlowDownChannels(r, self.overscale)
			
			# Put into pyramid
			dec_pyr[u] = r
			
			# Up to next z layer
			z_down = F.interpolate(z_down, scale_factor=2.0, mode='bilinear', align_corners=False)
			z_down = z_down + r
		
		return dec_pyr




def mkdir(path):
	try:
		os.makedirs(path, exist_ok=True)
	except Exception as e:
		print("Could not create directory:", path, e)


def main():
	import time
	import blockreader_sa1b
	import bicubic
	
	print('---------------------------------------------')
	print(' Initialize Autoencoder')
	print('---------------------------------------------')

	device = 'cuda'

	encoder = Encoder().to(device)
	decoder = Decoder().to(device)


	
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
	print(' Optimizer')
	print('---------------------------------------------')
	parameters = list(encoder.parameters()) + list(decoder.parameters())
	optimizer = torch.optim.AdamW(
		parameters,
		lr= 0.01,
		#lr = lr,
		#weight_decay=0.05
		weight_decay=0.02)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

	
	encoder_params = sum(p.numel() for p in encoder.parameters())
	decoder_params = sum(p.numel() for p in decoder.parameters())
	print(f"Encoder parameters: {encoder_params}")
	print(f"Decoder parameters: {decoder_params}")
	input('enter')

	print('---------------------------------------------')
	print(' Make output folders')
	print('---------------------------------------------')
	outdir = 'autoencoder'
	mkdir(outdir)
	

	floss = open(outdir+'/loss.csv', 'w')

	print('---------------------------------------------')
	print(' For every epoch')
	print('---------------------------------------------')
	for epoch in range(n_epoch):

		# reset losses per epoch
		epoch_total_loss = 0.0
		epoch_layer_loss = []
		for u in range(6):
			epoch_layer_loss.append(0.0)

	
		print('-----------------')
		print(' For every batch')
		print('-----------------')
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
			
			# Convert from [N,H,W,C] to [N,C,H,W]
			with torch.no_grad():
				bcb.restore_uint8_features(features)
				features = bcb.mask_data_raw
				features = torch.permute(features, (0,3,1,2)).contiguous()
			
			
			#----------------
			# Run Autoencoder
			#----------------

			with torch.no_grad():
				features_laplace = LaplacianTranform(features)

			#for i in range(len(features_laplace)):
			#	print('features_laplace', i, features_laplace[i].shape)
			optimizer.zero_grad()


			features_enc = encoder(features_laplace)

			#for i in range(len(features_enc)):
			#	print('features_enc', i, features_enc[i].shape)
		
			features_dec = decoder(features_enc)

			#for i in range(len(features_dec)):
			#	print('features_dec', i, features_dec[i].shape)

			# Calculate the layer loss
			total_loss = 0.0
			layer_loss = []
			for u in range(len(features_dec)):
				pred = features_dec[u]
				obs  = features_laplace[u]
				
				# mse loss
				loss = pred-obs
				loss = torch.mean(loss*loss)
				layer_loss.append(loss)
				total_loss = loss + total_loss
				print('layer', u, 'loss', float(loss))
			
			total_loss.backward()
			optimizer.step()
			print('total_loss', float(total_loss))

			# append epoch losses
			epoch_total_loss = total_loss + epoch_total_loss
			for u in range(len(layer_loss)):
				epoch_layer_loss[u] = layer_loss[u] + epoch_layer_loss[u]

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

		print('-----------------')
		print(' Epoch', epoch, 'loss')
		print('-----------------')

		# Divide by number of batches
		epoch_total_loss /= n_batch
		for u in range(len(layer_loss)):
			epoch_layer_loss[u] /= n_batch

		# write the loss
		floss.write('%.1f\t%d\t%.6f\t' % (total_elapsed, epoch, float(epoch_total_loss)))
		for u in range(len(epoch_layer_loss)):
			floss.write('%.6f\t' % float(epoch_layer_loss[u]))
		floss.write('\n')
		floss.flush()
		
		print('%d\t%.6f\t' % (epoch, float(epoch_total_loss)), end='')
		for u in range(len(epoch_layer_loss)):
			print('%.6f\t' % float(epoch_layer_loss[u]), end='')
		print('', flush=True)

		print('-----------------')
		print('Save the weights')
		print('-----------------')

		encoder_path = '%s/encoder_%05d.pth' % (outdir, epoch)
		print(encoder_path)
		torch.save(encoder.state_dict(), encoder_path)

		decoder_path = '%s/decoder_%05d.pth' % (outdir, epoch)
		print(decoder_path)
		torch.save(decoder.state_dict(), decoder_path)

	floss.close()


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


