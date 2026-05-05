import os
# Must be set BEFORE import torch
#os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import inspect

def show_call(fn, args):
    print("Function:", fn)
    print("Signature/doc:")
    try:
        print(inspect.signature(fn))
    except Exception as e:
        print("inspect.signature failed:", e)
        print(getattr(fn, "__doc__", None))

    print("\nActual arguments:")
    for i, a in enumerate(args):
        print(f"[{i}] type={type(a)}")# value={a!r}")
        if isinstance(a, torch.Tensor):
            print(
                f"    tensor: dtype={a.dtype}, device={a.device}, "
                f"shape={tuple(a.shape)}, contiguous={a.is_contiguous()}"
            )
            
            
bcb__cuda_source = None
bcb__cpp_source = None
bcb__cuda_flags = None
bcb__module = None

#----------------------------------------
# Compile the histogram cuda kernel
#----------------------------------------
def Compile():
	global bcb__cuda_source
	global bcb__cpp_source
	global bcb__cuda_flags
	global bcb__module

	with open('bicubic.cu', 'r') as fin:
		bcb__cuda_source = fin.read()

	print(bcb__cuda_source)

	bcb__cpp_source = '''
		void bicubic_float_cuda(torch::Tensor out, torch::Tensor in, torch::Tensor corners);
		void bicubic_aa_uint8_cuda(torch::Tensor out, torch::Tensor in, torch::Tensor corners, torch::Tensor aa_ker);
		void gaussian_blur_h_trans_cuda(torch::Tensor input,torch::Tensor output,torch::Tensor sigmas);
		unsigned int augment_photometric_cuda(torch::Tensor input, torch::Tensor output, torch::Tensor h_shifts, torch::Tensor s_factors, torch::Tensor v_factors, torch::Tensor sol_thresholds, torch::Tensor noise_scales, torch::Tensor mean, torch::Tensor std, unsigned int seed);
		void restore_uint8_features_cuda(torch::Tensor input,torch::Tensor output);
		unsigned int roll_dice_cuda(float horiz_flip, float area_scale_lo, float area_scale_hi,
			float aspect_ratio_lo, float aspect_ratio_hi, float rotation, float corner_jitter,
			float sigma_blur, float blur_aspect_lo, float blur_aspect_hi, float h_shift,
			float s_factor_lo, float s_factor_hi, float v_factor_lo, float v_factor_hi,
			float sol_threshold_lo, float sol_threshold_hi,
			float sol_chance, float noise_scale,
			int iH, int iW, int oH, int oW, int mH, int mW,
			torch::Tensor img_corners, torch::Tensor img_aa_ker, torch::Tensor mask_corners, 
			torch::Tensor blur_sigmas_x, torch::Tensor blur_sigmas_y, torch::Tensor h_shifts, torch::Tensor s_factors,
			torch::Tensor v_factors, torch::Tensor sol_thresholds, torch::Tensor noise_scales, unsigned int seed);
	'''

	def get_cuda_arch_flags():
		"""Detects local GPU and returns the 'sm_XX' flag."""
		if not torch.cuda.is_available():
			return []
		
		major, minor = torch.cuda.get_device_capability()
		arch_version = f"{major}{minor}"
		
		# Optional: Safety cap for Blackwell (compute 12+) if using older compilers
		# Many compilers in early 2026 still prefer sm_90 for stability
		if major >= 12:
			print(f"Targeting Blackwell ({arch_version}) with sm_90 compatibility mode.")
			return [f"-arch=sm_90"]
			
		return [f"-arch=sm_{arch_version}"]

	# Get the flags dynamically
	bcb__cuda_flags = get_cuda_arch_flags()

	# Compiles on the fly!
	bcb__module = load_inline(
		name='inline_extension',
		cpp_sources=[bcb__cpp_source],
		cuda_sources=[bcb__cuda_source],
		functions=[
			'bicubic_float_cuda', 
			'bicubic_aa_uint8_cuda',
			'gaussian_blur_h_trans_cuda',
			'augment_photometric_cuda',
			'restore_uint8_features_cuda',
			'roll_dice_cuda'],
		with_cuda=True,
		extra_cuda_cflags=bcb__cuda_flags
	)
	
class Bicubic:

	def __init__(self, 
	  in_shape=(128,1500,1500,4),
	  img_shape=(128,896,896,4),
	  mask_shape=(128,64,64,384),
	  img_mean=(0.485, 0.456, 0.406),
	  img_std=(0.229, 0.224, 0.225),
	  device='cuda'):
		
		# Make sure the cuda code is compiled
		if bcb__module is None:
			Compile()
		
		#help(bcb__module.roll_dice_cuda)
		#sys.exit(1)
		
		# Remember the shapes
		self.in_shape   = in_shape
		self.img_shape  = img_shape
		self.mask_shape = mask_shape
		
		# Define default hyperparameters for augmentation
		self.horiz_flip = 0.5                   # Bernoulli(0.5)              Standard symmetry.
		self.area_scale_lo = 0.08               # Log-Uniform(0.08, 1.0)      Forces scale invariance across octaves.
		self.area_scale_hi = 1.0
		self.aspect_ratio_lo = 0.75             # Log-Uniform(0.75, 1.3333)   Handles different sensor shapes.
		self.aspect_ratio_hi = 1.3333
		self.rotation = 15.0                    # Normal(0, 15°)              90% of the time; helps with camera tilt.
		self.corner_jitter = 0.08               # Uniform(-0.08*L, 0.08*L)	   Adds non-rigid perspective robustness.
		self.sigma_blur = 1.0                   #  (pixels)​ HalfNormal(sig=1.0)         Most images have σ<1.0. Only rare ones are very blurry.
		self.blur_aspect_lo = 0.75              # Log-Uniform(0.75,1.3333)      Ratio of σx​/σy​,"20% of the time, apply this to simulate motion blur."
		self.blur_aspect_hi = 1.3333
		self.h_shift = 7.0                      #  Normal(0,7)                 Hue is sensitive. A little goes a long way. 0.02 is approx ±7∘.
		self.s_factor_lo = 0.7                  #  LogUniform(0.7,1.4)         Multiplicative. Avoids ""gray-scale"" unless you explicitly want it."
		self.s_factor_hi = 1.4
		self.v_factor_lo = 0.5                  #  LogUniform(0.5,1.5)         Simulates exposure/lighting. Log-uniform makes ""half-light"" as likely as ""double-light."""
		self.v_factor_hi = 1.5
		self.sol_threshold_lo = 0.6             #  Uniform(0.6,0.95)           Flip only the brightest pixels to create a halo
		self.sol_threshold_hi = 0.95
		self.sol_chance = 0.05                  #  Bernoulli(0.05)             (Solarize) Use sparingly. Only apply with a 5% probability.
		self.noise_scale = 20.0                 #  Exponential(λ=20)           Most images should have low noise; few should be very grainy.

		img_shape_tran = (img_shape[0], img_shape[2], img_shape[1], img_shape[3])   # transposed image shape
		N = img_shape[0]

		# Allocate tensors for data
		self.mask_data_raw         = torch.zeros(mask_shape,      dtype=torch.float32, requires_grad=False, device=device)     # Before BICUBIC
		self.mask_data             = torch.zeros(mask_shape,      dtype=torch.float32, requires_grad=False, device=device)     # After BICUBIC
		self.in_data_raw           = torch.zeros(in_shape,        dtype=torch.uint8,   requires_grad=False, device=device)     # RGBA input pixels
		self.img_data_bicubic      = torch.zeros(img_shape,       dtype=torch.float32, requires_grad=False, device=device)     # Image data after BICUBIC
		self.img_data_h_blur_trasp = torch.zeros(img_shape_tran,  dtype=torch.float32, requires_grad=False, device=device)     # Image data after horizontal blur (with transpose)
		self.img_data_blur         = torch.zeros(img_shape,       dtype=torch.float32, requires_grad=False, device=device)     # Image data after blur
		self.img_data              = torch.zeros(img_shape,       dtype=torch.float32, requires_grad=False, device=device)     # Final augmented image shape

		# Allocate tensors for hyperparameters
		self.img_corners     = torch.zeros((N,4,2),  dtype=torch.float32, requires_grad=False, device=device)   #  img_corners  [N 4 2]   corner points  (float32)
		self.img_aa_ker      = torch.zeros((N,2),    dtype=torch.int32,   requires_grad=False, device=device)   #  img_aa_ker   [N 2]     anti-alaising kernel (int32)
		self.mask_corners    = torch.zeros((N,4,2),  dtype=torch.float32, requires_grad=False, device=device)   #  mask_corners [N 4 2]   corner points  (float32)
		self.blur_sigmas_x   = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  blur_sigmas_x  [N]     float32    blur kernel sigmas array (horizontal)
		self.blur_sigmas_y   = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  blur_sigmas_y  [N]     float32    blur kernel sigmas array (vertical)
		self.h_shifts        = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  h_shifts  [N]          float32    hue shift
		self.s_factors       = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  s_factors [N]          float32    saturation factor
		self.v_factors       = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  v_fators  [N]          float32    value factor
		self.sol_thresholds  = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  sol_thresholds [N]     float32    threshold for solarization
		self.noise_scales    = torch.zeros((N,),     dtype=torch.float32, requires_grad=False, device=device)   #  noise_scales [N]       float32    noise scales to add

		# Save the image mean and stdev
		self.img_mean = torch.tensor(img_mean, dtype=torch.float32, requires_grad=False, device=device)
		self.img_std  = torch.tensor(img_std,  dtype=torch.float32, requires_grad=False, device=device)

		# Initialize random seed
		self.seed = 2*16777216

	#--------------------------
	# restore_uint8_features_cuda
	#
	#   Restores the DINO features 
	#     from    uint8     frames
	#     to      float32   tensor
	#
	#   mask_data_uint8     [N frame]   (utf8)
	#   self.mask_data_raw  [N H W C]
	#
	#   frame format:
	#     header   minvals  maxvals  payload
	#
	#   header:    H (uint32)  W (uint32)  C (uint32)
	#   minvals:   [C]  (float16)
	#   maxvals:   [C]  (float16)
	#   payload:   [H W C]  (uint8)  0 minval  255 maxval
	#--------------------------
	def restore_uint8_features(self, mask_data_uint8):
		
		torch.cuda.synchronize()
		bcb__module.restore_uint8_features_cuda(mask_data_uint8, self.mask_data_raw)
		torch.cuda.synchronize()

	#--------------------------
	#  roll_dice_kernel
	#
	#    bicubic parameters
	#  horiz_flip         Bernoulli(0.5)              Standard symmetry.
	#  area_scale	       Log-Uniform(0.08, 1.0)      Forces scale invariance across octaves.
	#  aspect_ratio       Log-Uniform(0.75, 1.3333)   Handles different sensor shapes.
	#  rotation           Normal(0, 15°)              90% of the time; helps with camera tilt.
	#  corner_jitter      Uniform(-0.08*L, 0.08*L)	   Adds non-rigid perspective robustness.
	#                       note L = sqrt(area_scale)
	#
	#    gaussian blur
	#  sigma_blur(pixels)​ HalfNormal(sig=1.0)         Most images have σ<1.0. Only rare ones are very blurry.
	#  blur_aspect        Log-Uniform(0.75,1.3333)    Ratio of σx​/σy​,"20% of the time, apply this to simulate motion blur."
	#
	#    photometric parameters
	#  h_shift (Hue)      Normal(0.0,7.0) degrees     Hue is sensitive. A little goes a long way. approx +- 7 degrees.
	#  s_factor (Sat)     LogUniform(0.7,1.4)         Multiplicative. Avoids ""gray-scale"" unless you explicitly want it."
	#  v_factor (Value)   LogUniform(0.5,1.5)         Simulates exposure/lighting. Log-uniform makes ""half-light"" as likely as ""double-light."""
	#  sol_threshold      Uniform(0.6,0.95)           Flip only the brightest pixels to create a halo
	#  sol_chance         Bernoulli(0.05)             (Solarize) Use sparingly. Only apply with a 5% probability.
	#  noise_scale        Exponential(λ=20)           Most images should have low noise; few should be very grainy.
	#
	#    shapes
	#  iH iW   oH oW   mH iW     inimage (i) outimg (o) and mask (m) dimensions
	#
	#    outputs
	#  self.img_corners  [N 4 2]   corner points  (float32)
	#  self.img_aa_ker   [N 2]     anti-alaising kernel (int32)
	#  self.mask_corners [N 4 2]   corner points  (float32)
	#  self.blur_sigmas_x  [N]     float32    blur kernel sigmas array (horizontal)
	#  self.blur_sigmas_y  [N]     float32    blur kernel sigmas array (vertical)
	#  self.h_shifts  [N]          float32    hue shift
	#  self.s_factors [N]          float32    saturation factor
	#  self.v_fators  [N]          float32    value factor
	#  self.sol_thresholds [N]     float32    threshold for solarization
	#  self.noise_scales [N]       float32    noise scales to add
	#--------------------------
	def roll_dice(self):
	
		iH = self.in_shape[1]
		iW = self.in_shape[2]
		oH = self.img_shape[1]
		oW = self.img_shape[2]
		mH = self.mask_shape[1]
		mW = self.mask_shape[2]


		print('horiz_flip',    self.horiz_flip)                               	# Bernoulli(0.5)              Standard symmetry.
		print('area_scale',    self.area_scale_lo, self.area_scale_hi)       	# Log-Uniform(0.08, 1.0)      Forces scale invariance across octaves.
		print('aspect_ratio',  self.aspect_ratio_lo, self.aspect_ratio_hi)   	# Log-Uniform(0.75, 1.3333)   Handles different sensor shapes.
		print('rotation',      self.rotation)                                 	# Normal(0, 15°)              90% of the time; helps with camera tilt.
		print('corner_jitter', self.corner_jitter)                            	# Uniform(-0.08*L, 0.08*L)	   Adds non-rigid perspective robustness.
		print('sigma_blur', self.sigma_blur)                               	#  (pixels)​ HalfNormal(sig=1.0)         Most images have σ<1.0. Only rare ones are very blurry.
		print('blur_aspect', self.blur_aspect_lo, self.blur_aspect_hi)     	# Log-Uniform(0.75,1.3333)    Ratio of σx​/σy​,"20% of the time, apply this to simulate motion blur."
		print('h_shift', self.h_shift)                                  	#  Normal(0,0.02)              Hue is sensitive. A little goes a long way. 0.02 is approx ±7∘.
		print('s_factor', self.s_factor_lo, self.s_factor_hi)           	#  LogUniform(0.7,1.4)         Multiplicative. Avoids ""gray-scale"" unless you explicitly want it."
		print('v_factor', self.v_factor_lo, self.v_factor_hi)           	#  LogUniform(0.5,1.5)         Simulates exposure/lighting. Log-uniform makes ""half-light"" as likely as ""double-light."""
		print('sol_threshold', self.sol_threshold_lo, self.sol_threshold_hi) 	#  Uniform(0.6,0.95)           Flip only the brightest pixels to create a halo
		print('sol_chance', self.sol_chance)                               	#  Bernoulli(0.05)             (Solarize) Use sparingly. Only apply with a 5% probability.
		print('noise_scale', self.noise_scale)                              	#  Exponential(λ=20)           Most images should have low noise; few should be very grainy.
		print('iH, iW, oH, oW, mH, mW', iH, iW, oH, oW, mH, mW) 	# Image, outimg, and mask dimensions
		print('img_corners.shape', self.img_corners.shape)
		print('img_aa_ker.shape', self.img_aa_ker.shape)
		print('mask_corners.shape', self.mask_corners.shape)
		print('blur_sigmas_x.shape', self.blur_sigmas_x.shape)  	# Blur kernel sigmas array [N]
		print('blur_sigmas_y.shape', self.blur_sigmas_y.shape)  	# Blur kernel sigmas signams array [N]
		print('h_shifts.shape', self.h_shifts.shape)       	# Hue shift array [N]  (-180.0 to 180.0)
		print('s_factors.shape', self.s_factors.shape)      	# Saturation factor array [N]
		print('v_factors.shape', self.v_factors.shape)      	# Value (Brightness) factor array [N]
		print('sol_thresholds.shape', self.sol_thresholds.shape) 	# Solarization threshold array [N]
		print('noise_scales.shape', self.noise_scales.shape)   	# Noise scales for gaussian additive noise
		print('seed', self.seed)

		torch.cuda.synchronize()
		self.seed = bcb__module.roll_dice_cuda(
			float(self.horiz_flip),                               	# Bernoulli(0.5)              Standard symmetry.
			float(self.area_scale_lo), float(self.area_scale_hi),       	# Log-Uniform(0.08, 1.0)      Forces scale invariance across octaves.
			float(self.aspect_ratio_lo), float(self.aspect_ratio_hi),   	# Log-Uniform(0.75, 1.3333)   Handles different sensor shapes.
			float(self.rotation),                                 	# Normal(0, 15°)              90% of the time; helps with camera tilt.
			float(self.corner_jitter),                            	# Uniform(-0.08*L, 0.08*L)	   Adds non-rigid perspective robustness.
			float(self.sigma_blur),                               	#  (pixels)​ HalfNormal(sig=1.0)         Most images have σ<1.0. Only rare ones are very blurry.
			float(self.blur_aspect_lo), float(self.blur_aspect_hi),     	# Log-Uniform(0.75,1.3333)    Ratio of σx​/σy​,"20% of the time, apply this to simulate motion blur."
			float(self.h_shift),                                  	#  Normal(0,0.02)              Hue is sensitive. A little goes a long way. 0.02 is approx ±7∘.
			float(self.s_factor_lo), float(self.s_factor_hi),           	#  LogUniform(0.7,1.4)         Multiplicative. Avoids ""gray-scale"" unless you explicitly want it."
			float(self.v_factor_lo), float(self.v_factor_hi),           	#  LogUniform(0.5,1.5)         Simulates exposure/lighting. Log-uniform makes ""half-light"" as likely as ""double-light."""
			float(self.sol_threshold_lo), float(self.sol_threshold_hi), 	#  Uniform(0.6,0.95)           Flip only the brightest pixels to create a halo
			float(self.sol_chance),                               	#  Bernoulli(0.05)             (Solarize) Use sparingly. Only apply with a 5% probability.
			float(self.noise_scale),                              	#  Exponential(λ=20)           Most images should have low noise; few should be very grainy.
			int(iH), int(iW), int(oH), int(oW), int(mH), int(mW), 	# Image, outimg, and mask dimensions
			self.img_corners,
			self.img_aa_ker,
			self.mask_corners,
			self.blur_sigmas_x,  	# Blur kernel sigmas array [N]
			self.blur_sigmas_y,  	# Blur kernel sigmas signams array [N]
			self.h_shifts,       	# Hue shift array [N]  (-180.0 to 180.0)
			self.s_factors,      	# Saturation factor array [N]
			self.v_factors,      	# Value (Brightness) factor array [N]
			self.sol_thresholds, 	# Solarization threshold array [N]
			self.noise_scales,   	# Noise scales for gaussian additive noise
			int(self.seed))
		torch.cuda.synchronize()

	#---------------------------
	#  bicubic_float_cuda
	#    out    [B oH oW C]   output image   (float32)
	#     in    [B oH oW C]   input image    (float32)
	#  corners  [B 4 2]       corner points  (float32)
	#---------------------------
	def bicubic_masks(self):
	
		torch.cuda.synchronize()
		bcb__module.bicubic_float_cuda(
			self.mask_data,
			self.mask_data_raw,
			self.mask_corners)		
		torch.cuda.synchronize()

	#---------------------------
	#  bicubic_aa_uint8_kernel
	#      B iH iW oH oW C    input dimesions
	#    out_data    [B oH oW C]   output image   (float32)   (0.0 to 1.0)
	#     in_data    [B iH iW C]   input image    (uint8_t)   (0 to 255)
	#  corners_data  [B 4 2]       corner points  (float32)
	#   aa_ker_data  [B 2]   anti-alaising kernel   (int32)
	#---------------------------
	def bicubic_images(self, in_data_raw):
		
		torch.cuda.synchronize()
		bcb__module.bicubic_aa_uint8_cuda(
			self.img_data_bicubic,
			in_data_raw,
			self.img_corners,
			self.img_aa_ker)
		torch.cuda.synchronize()

	#---------------------------
	# gaussian_blur_trans_kernel
	#   Performs horizontal gaussian blur and transposes 
	#    the result.  Run this twice to get a full (separable)
	#    horizontal/vertical gaussian blur.
	#   Also divides by kernel integral, which handles
	#    boundaries more accurately than edge padding.
	#  input   [N H W 4]   RGBA float32
	#  output  [N W H 4]   RGBA float32  (transposed!)
	#  sigmas  [N]         blur kernel sigmas (0.5 to 2.0)
	#---------------------------
	def gaussian_blur(self):
	
		# Horizontal Blur
		torch.cuda.synchronize()
		bcb__module.gaussian_blur_h_trans_cuda(
			self.img_data_bicubic,	         # [N H W 4]
			self.img_data_h_blur_trasp,      # [N W H 4]   Transposed!
			self.blur_sigmas_x)              # Per-image sigma [N]
		torch.cuda.synchronize()
	
		# Vertical Blur
		torch.cuda.synchronize()
		bcb__module.gaussian_blur_h_trans_cuda(
			self.img_data_h_blur_trasp,	 # [N H W 4]
			self.img_data_blur,              # [N W H 4]   Transposed!
			self.blur_sigmas_y)              # Per-image sigma [N]
		torch.cuda.synchronize()
		
	#--------------------------
	# Photometric Kernel: Operates on float32 RGBA in range [0, 1].
	# Performs: Naive Solarization, HSV-based Color Jitter, and Noise.
	# Final pass performs Clamping and Mean/Std Normalization.
	#
	#  input     [N H W 4]  float32    input image (0 to 1)
	#  output    [N H W 4]  float32    output image (0 to 1)
	#  h_shifts  [N]        float32    hue shift
	#  s_factors [N]        float32    saturation factor
	#  v_fators  [N]        float32    value factor
	#  sol_thresholds [N]   float32    threshold for solarization
	#  noise_scales [N]     float32    noise scales to add
	#  mean                 float3     mean values          (normalization)
	#  std                  float3     standard deviations  (normalization)
	#  N H W                int32
	#  seed                            seed for normal distribution hashing
	#  
	#  return    updated seed
	#--------------------------
	def photometric(self):
		
		torch.cuda.synchronize()
		self.seed = bcb__module.augment_photometric_cuda(
			self.img_data_blur,
			self.img_data,
			self.h_shifts,       	# Hue shift array [N]  (-180.0 to 180.0)
			self.s_factors,      	# Saturation factor array [N]
			self.v_factors,      	# Value (Brightness) factor array [N]
			self.sol_thresholds, 	# Solarization threshold array [N]
			self.noise_scales,   	# Noise scales for gaussian additive noise
			self.img_mean,          # Pre-calculated normalization mean
			self.img_std,           # Pre-calculated normalization std
			self.seed)
		torch.cuda.synchronize()
	
	#--------------------------
	#  THE BIG RED BUTTON
	#   Runs everything after 'roll_dice' for just images
	#    (not masks)
	#--------------------------
	def augment_images(self, in_data_raw):
	
		self.bicubic_images(in_data_raw)
		self.gaussian_blur()
		self.photometric()
		return self.img_data










