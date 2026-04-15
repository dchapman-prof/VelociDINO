import os
# Must be set BEFORE import torch
#os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


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
			'augment_photometric_cuda'],
		with_cuda=True,
		extra_cuda_cflags=bcb__cuda_flags
	)

