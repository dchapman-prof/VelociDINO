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
		qk = v * qk_norm    # B M d H W
	
		# Calculate qk dependence at all offsets
		score_     = torch.ones((B,M,1,H,W,1), device=qk.device,requires_grad=False,dtype=torch.float32)         # [B M  H  W]
		score_lr   = (qk[:,:,:,:,0:(W-1)]       * qk[:,:,:,:,1:W]).sum(dim=2)       # [B M  H  W-1]
		score_bt   = (qk[:,:,:,0:(H-1),:]       * qk[:,:,:,1:H,:]).sum(dim=2)       # [B M H-1  W ]
		score_d1   = (qk[:,:,:,0:(H-1),0:(W-1)] * qk[:,:,:,1:H,1:W]).sum(dim=2)     # [B M H-1 W-1]
		score_d2   = (qk[:,:,:,0:(H-1),1:W]     * qk[:,:,:,1:H,0:(W-1)]).sum(dim=2) # [B M H-1 W-1]
	
		# Pad to get the scores in a 3x3 neighborhood
		score_l    = torch.pad(score_lr, (1,0,0,0))   # [B M H W]
		score_r    = torch.pad(score_lr, (0,1,0,0))
		score_b    = torch.pad(score_bt, (0,0,1,0))
		score_t    = torch.pad(score_bt, (0,0,0,1))
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
				dim=5)                                # [B M 1 H W 9]
		
		# Shift the values around (there has to be a better way...)
		v = torch.reshape(v, (B,C,H,W))
		v_l  = torch.pad(v[:,:,:,0:(W-1)], (1,0,0,0))
		v_r  = torch.pad(v[:,:,:,1:W],     (0,1,0,0))
		v_b  = torch.pad(v[:,:,0:(H-1),:], (0,0,1,0))
		v_t  = torch.pad(v[:,:,1:H,:],     (0,0,0,1))
		v_bl = torch.pad(v[:,:,0:(H-1),0:(W-1)], (1,0,1,0))
		v_tl = torch.pad(v[:,:,0:H,    0:(W-1)], (1,0,0,1))
		v_br = torch.pad(v[:,:,0:(H-1),1:W],     (0,1,1,0))
		v_tr = torch.pad(v[:,:,1:H,    1:W],     (0,1,0,1))  # [B C H W]
		v_stack = torch.cat(
			torch.reshape(v,    (B,M,d,H,W,1)),
			torch.reshape(v_l,  (B,M,d,H,W,1)),
			torch.reshape(v_r,  (B,M,d,H,W,1)),
			torch.reshape(v_b,  (B,M,d,H,W,1)),
			torch.reshape(v_t,  (B,M,d,H,W,1)),
			torch.reshape(v_bl, (B,M,d,H,W,1)),
			torch.reshape(v_tl, (B,M,d,H,W,1)),
			torch.reshape(v_br, (B,M,d,H,W,1)),
			torch.reshape(v_tr, (B,M,d,H,W,1),
				dim=5)                                # [B M d H W 9]
	
		# Perform softmax of scores
		score = F.softmax(score, dim=5)               # [B M 1 H W 9]
		
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
		qk = v * qk_norm    # B M d H W
	
		# Calculate qk dependence at all offsets
		score_     = torch.ones((B,M,1,H,W,1), device=qk.device, dtype=x.dtype,requires_grad=False)#dtype=torch.float32)         # [B M  H  W]
		score_lr   = (qk[:,:,:,:,0:(W-1)]       * qk[:,:,:,:,1:W]).sum(dim=2)       # [B M  H  W-1]
		score_bt   = (qk[:,:,:,0:(H-1),:]       * qk[:,:,:,1:H,:]).sum(dim=2)       # [B M H-1  W ]
		score_d1   = (qk[:,:,:,0:(H-1),0:(W-1)] * qk[:,:,:,1:H,1:W]).sum(dim=2)     # [B M H-1 W-1]
		score_d2   = (qk[:,:,:,0:(H-1),1:W]     * qk[:,:,:,1:H,0:(W-1)]).sum(dim=2) # [B M H-1 W-1]
	
		# Pad to get the scores in a 3x3 neighborhood
		score_l    = F.pad(score_lr, (1,0,0,0))   # [B M H W]
		score_r    = F.pad(score_lr, (0,1,0,0))
		score_b    = F.pad(score_bt, (0,0,1,0))
		score_t    = F.pad(score_bt, (0,0,0,1))
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
				dim=5)                                # [B M 1 H W 9]
		
		# Shift the values around (there has to be a better way...)
		v = torch.reshape(v, (B,C,H,W))
		v_l  = F.pad(v[:,:,:,0:(W-1)], (1,0,0,0))
		v_r  = F.pad(v[:,:,:,1:W],     (0,1,0,0))
		v_b  = F.pad(v[:,:,0:(H-1),:], (0,0,1,0))
		v_t  = F.pad(v[:,:,1:H,:],     (0,0,0,1))
		v_bl = F.pad(v[:,:,0:(H-1),0:(W-1)], (1,0,1,0))
		v_tl = F.pad(v[:,:,1:H,    0:(W-1)], (1,0,0,1))
		v_br = F.pad(v[:,:,0:(H-1),1:W],     (0,1,1,0))
		v_tr = F.pad(v[:,:,1:H,    1:W],     (0,1,0,1))  # [B C H W]
		v_stack = torch.cat((
			torch.reshape(v,    (B,M,d,H,W,1)),
			torch.reshape(v_l,  (B,M,d,H,W,1)),
			torch.reshape(v_r,  (B,M,d,H,W,1)),
			torch.reshape(v_b,  (B,M,d,H,W,1)),
			torch.reshape(v_t,  (B,M,d,H,W,1)),
			torch.reshape(v_bl, (B,M,d,H,W,1)),
			torch.reshape(v_tl, (B,M,d,H,W,1)),
			torch.reshape(v_br, (B,M,d,H,W,1)),
			torch.reshape(v_tr, (B,M,d,H,W,1))),
				dim=5)                                # [B M d H W 9]
	
		# Perform softmax of scores
		score = F.softmax(score, dim=5)               # [B M 1 H W 9]
		
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

	def __init__(self, baseC, baseM, H=64, W=64, C_factor=2, M_factor=1, u_depth=5):
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


			
				

	
class LinearLens(nn.Module):

	def __init__(self, outC=384, baseC=16, C_factor=2, u_depth0=2, u_depth1=5):
		super().__init__()
	
		self.baseC    = baseC
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
		self.baseH    = baseH
		self.baseW    = baseW
		self.n_layer  = n_layer
		self.outC     = outC
		self.baseC    = baseC
		self.baseM    = baseM
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

		#--------
		# Final lens
		#--------
		out = self.lens(stream)

		if return_featvec:
			return out, featvec

		return out
			

