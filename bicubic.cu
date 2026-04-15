#include <iostream>
#include <cstdio>
#include <torch/extension.h>

//--------------------
// Calculations for cubic spline interpolation
//  y = a x^3 + b x^2 + c x + d
//
//  Given values
//     (-1, y0)    (0, y1)    (1, y2)    (2, y3)
//  Then we have
//     y0 = -a + b - c + d
//     y1 = d
//     y2 =  a + b + c + d
//     y3 = 8a + 4b + 2c + d
//
//  So 'd' is solved
//     y0 - y1 = -a + b  - c
//     y2 - y1 =  a + b  + c
//     y3 - y1 = 8a + 4b + 2c
//  Thus
//     (y0 + y2 - 2y1)/2 = b
//  Now 'b' is solved
//     y3 - y1 + 2y0 - 2y1 = 6a + 6b
//     y3 - 3y1 + 2y0 - 6b = 6a
//     y3 - 3y1 + 2y0 - 3y0 - 3y2 + 6y1 = 6a
//     -y0 + 3y1 -3y2 + y3 = 6a
//     (-y0 + 3y1 - 3y2 + y3)/6 = a
//  Now 'a' is solved
//     y2 -(a) -(b) -(d) = c  
//     y2 + (1/6)y0 - (1/2)y1 + (1/2)y2 - (1/6)y3 - 0.5y0 - 0.5y2 + y1 - y1 = c
//     (-2/6)y0 - (1/2)y1 + y2 - (1/6)y3 = c
//  Now 'c' is solved
//
//  So in summary
//     a = -(1/6)y0 + (1/2)y1 - (1/2)y2 + (1/6)y3
//     b =  (1/2)y0      - y1 + (1/2)y2
//     c = (-2/6)y0 - (1/2)y1 + y2 - (1/6)y3
//     d =                 y1
//
//  Quadratic
//   y0 = A - B + C
//   y1 = C
//   y2 = A + B + C
//------------------------------------



//------------------------------------
//   2B = y2 - y0
//   B = (1/2)y2 - (1/2)y0
//   2A + 2C = y2 + y0
//   A = (1/2)y2 + (1/2)y0 - y1
//------------------------------------



//--------------------
// Calculations for cubic spline interpolation
//  y  = a x^3 + b x^2 + c x + d
//  y' =       3a x^2 + 2b x + c
//
//  Given values
//     (-1, y0)    (0, y1)    (1, y2)    (2, y3)
//  Then we have
//     y1  = d
//     y2  =  a + b + c + d
//    y1'  = 0.5*(y2-y0)
//    y2'  = 0.5*(y3-y1)
//
//  So (d) is solved
//    (y2-y1) = a + b + c
//     y1' = c
//  So (c) is solved
//    y2 - y1 - y1' = a + b
//     y2' = 3a + 2b + c
//    y2' - y1' = 3a + 2b
//
//    y2' - y1' - 2(y2 - y1 - y1') = a
//    y2' - y1' - 2y2 + 2y1 + 2y1' = a
//    2y1 - 2y2 + y1' + y2' = a
//  So (a) is solved
//    y2 - y1 - y1' - (2y1 - 2y2 + y1' + y2') = b
//    y2 - y1 - y1' - 2y1 + 2y2 - y1' - y2' = b
//    - 3y1 + 3y2 - 2y1' - y2' = b
//
// Therefore
//   a = 2y1 - 2y2 + y1' + y2'
//   b = - 3y1 + 3y2 - 2y1' - y2'
//   c = y1'
//   d = y1
//------------------------------------------

__device__ float clamp(float val, float min, float max) {
    return fminf(fmaxf(val, min), max);
}

//---------------------------
//  bicubic_float_cuda
//      B iH iW oH oW C    input dimesions
//    out_data    [B oH oW C]   output image   (float32)
//     in_data    [B iH iW C]   input image    (float32)
//  corners_data  [B 4 2]       corner points  (float32)
//---------------------------
__global__
void bicubic_float_kernel(
	int B, int iH, int iW, int oH, int oW, int C,
	float*        __restrict__ out_data,
	const float * __restrict__ in_data,
	const float *corners_data)
{
	// calculate rank
	int rank_x = blockIdx.x * blockDim.x + threadIdx.x;
	int rank_y = blockIdx.y * blockDim.y + threadIdx.y;
	int rank_z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (rank_x>=oW || rank_y>=oH || rank_z>=B)
		return;

	// Which 'slice' are we looking at?
	float *out           =     out_data + ( rank_z * (oH*oW*C) );
	const float * in     =      in_data + ( rank_z * (iH*iW*C) );
	const float *corners = corners_data + ( rank_z * 8 );
	
	//------
	// Determine the (c_x,c_y) coordinate within the input image
	//------
	
	// What are the corner points of the image
	float c_x00 = corners[0];
	float c_y00 = corners[1];
	float c_x01 = corners[2];
	float c_y01 = corners[3];
	float c_x11 = corners[4];
	float c_y11 = corners[5];
	float c_x10 = corners[6];
	float c_y10 = corners[7];
	
	// What is delta location within the output image
	float delta_x = (rank_x+0.5) / (float)oW;
	float delta_y = (rank_y+0.5) / (float)oH;
	
	// Bilinear Interpolate between four corner points
	float c_x0 = c_x00 + delta_x * (c_x01-c_x00);
	float c_y0 = c_y00 + delta_x * (c_y01-c_y00);
	float c_x1 = c_x10 + delta_x * (c_x11-c_x10);
	float c_y1 = c_y10 + delta_x * (c_y11-c_y10);
	float c_x  = c_x0  + delta_y *  (c_x1-c_x0);
	float c_y  = c_y0  + delta_y *  (c_y1-c_y0);
	
	// Out of bounds
	c_x = clamp(c_x, 0.00001, iW-0.00001);
	c_y = clamp(c_y, 0.00001, iH-0.00001);
		
	//------
	// Solve the 'bicubic' interpolation coefficients
	//------
	
	// determine grid cell and delta
	int i_x = int(c_x);
	int i_y = int(c_y);
	float d_x = c_x - i_x;
	float d_y = c_y - i_y;
	float d_x2 = d_x*d_x;
	float d_y2 = d_y*d_y;
	float d_x3 = d_x2 * d_x;
	float d_y3 = d_y2 * d_y;
	
	
	// Indices for the 4x4 kernel   idx_y_x
	int idx_1_1 = i_y*iW*C + i_x*C;
	int idx_2_1 = idx_1_1 + iW*C;
	int idx_0_1 = (i_y>0)    ? (idx_1_1 - iW*C) : idx_1_1;
	int idx_3_1 = (i_y<iW-1) ? (idx_2_1 + iW*C) : idx_2_1;
	int idx_0_2 = idx_0_1 + C;
	int idx_1_2 = idx_1_1 + C;
	int idx_2_2 = idx_2_1 + C;
	int idx_3_2 = idx_3_1 + C;
	int idx_0_0 = idx_0_1 - C;
	int idx_1_0 = idx_1_1 - C;
	int idx_2_0 = idx_2_1 - C;
	int idx_3_0 = idx_3_1 - C;
	int idx_0_3 = idx_0_1 - C;
	int idx_1_3 = idx_1_1 - C;
	int idx_2_3 = idx_2_1 - C;
	int idx_3_3 = idx_3_1 - C;
	
	// Output index
	int idx_out = rank_y*oW*C + rank_x*C;
	
	// For every channel
	//  (four polynomials)   b=bottom  t=top  l=left r=right
	//                     Y1  Y2  (derivatives)   y1 y2  (values)
	//                 this notation matches math above.
	for (int ic=0; ic<C; ic++)
	{
		//----
		// Horizontal cubic interpolation
		//----
		
		// Read derivatives
		float Y01 = (i_x>0)    ? ( 0.5*(in[idx_0_2]-in[idx_0_0]) ) : (in[idx_0_2]-in[idx_0_1]);
		float Y02 = (i_x<iW-1) ? ( 0.5*(in[idx_0_3]-in[idx_0_1]) ) : (in[idx_0_2]-in[idx_0_1]);
		float Y11 = (i_x>0)    ? ( 0.5*(in[idx_1_2]-in[idx_1_0]) ) : (in[idx_1_2]-in[idx_1_1]);
		float Y12 = (i_x<iW-1) ? ( 0.5*(in[idx_1_3]-in[idx_1_1]) ) : (in[idx_1_2]-in[idx_1_1]);
		float Y21 = (i_x>0)    ? ( 0.5*(in[idx_2_2]-in[idx_2_0]) ) : (in[idx_2_2]-in[idx_2_1]);
		float Y22 = (i_x<iW-1) ? ( 0.5*(in[idx_2_3]-in[idx_2_1]) ) : (in[idx_2_2]-in[idx_2_1]);
		float Y31 = (i_x>0)    ? ( 0.5*(in[idx_3_2]-in[idx_3_0]) ) : (in[idx_3_2]-in[idx_3_1]);
		float Y32 = (i_x<iW-1) ? ( 0.5*(in[idx_3_3]-in[idx_3_1]) ) : (in[idx_3_2]-in[idx_3_1]);
		
		// Read values
		float y01 = in[idx_0_1];
		float y02 = in[idx_0_2];
		float y11 = in[idx_1_1];
		float y12 = in[idx_1_2];
		float y21 = in[idx_2_1];
		float y22 = in[idx_2_2];
		float y31 = in[idx_3_1];
		float y32 = in[idx_3_2];
		
		// Interpolation coefficients
		//   a = 2y1 - 2y2 + y1' + y2'
		//   b = - 3y1 + 3y2 - 2y1' - y2'
		//   c = y1'
		//   d = y1
		float a0 = 2.0*y01 - 2.0*y02 + Y01 + Y02;
		float a1 = 2.0*y11 - 2.0*y12 + Y11 + Y12;
		float a2 = 2.0*y21 - 2.0*y22 + Y21 + Y22;
		float a3 = 2.0*y31 - 2.0*y32 + Y31 + Y32;
		float b0 = -3.0*y01 + 3.0*y02 - 2.0*Y01 - Y02;
		float b1 = -3.0*y11 + 3.0*y12 - 2.0*Y11 - Y12;
		float b2 = -3.0*y21 + 3.0*y22 - 2.0*Y21 - Y22;
		float b3 = -3.0*y31 + 3.0*y32 - 2.0*Y31 - Y32;
		float c0 = Y01;
		float c1 = Y11;
		float c2 = Y21;
		float c3 = Y31;
		float d0 = y01;
		float d1 = y11;
		float d2 = y21;
		float d3 = y31;
		
		// Cubic horizontal interpolation
		float y0 = a0*d_x3 + b0*d_x2 + c0*d_x + d0;
		float y1 = a1*d_x3 + b1*d_x2 + c1*d_x + d1;
		float y2 = a2*d_x3 + b2*d_x2 + c2*d_x + d2;
		float y3 = a3*d_x3 + b3*d_x2 + c3*d_x + d3;
		
		//----
		// Vertical bi-cubic interpolation
		//----
		
		// Get derivatives
		float Y1 = (i_y>0)    ? ( 0.5*(y2-y0) ) : ( y2-y1 );
		float Y2 = (i_y<iH-1) ? ( 0.5*(y3-y1) ) : ( y2-y1 );
		
		// Bi-cubic coefficients
		float a = 2.0*y1 - 2.0*y2 + Y1 + Y2;
		float b = -3.0*y1 + 3.0*y2 - 2.0*Y1 - Y2;
		float c = Y1;
		float d = y1;
		
		// Bi-cubic interpolation
		float y = a*d_y3 + b*d_y2 + c*d_y + d;
		
		//----
		// Store the output value
		//----
		out[ idx_out ] = y;
		
		// Next index (channel)
		idx_0_0++;	idx_0_1++;	idx_0_2++;	idx_0_3++;
		idx_1_0++;	idx_1_1++;	idx_1_2++;	idx_1_3++;
		idx_2_0++;	idx_2_1++;	idx_2_2++;	idx_2_3++;
		idx_3_0++;	idx_3_1++;	idx_3_2++;	idx_3_3++;
		idx_out++;
	}
}



//---------------------------
//  bicubic_float_cuda
//    out    [B oH oW C]   output image   (float32)
//     in    [B oH oW C]   input image    (float32)
//  corners  [B 4 2]       corner points  (float32)
//---------------------------
void bicubic_float_cuda(
	torch::Tensor out,
	torch::Tensor in,
	torch::Tensor corners)
{
	printf("BEGIN: bicubic_float_cuda\n");
	
	// Pointer to the data
	float *out_data     = out.data_ptr<float>();
	float *in_data      = in.data_ptr<float>();
	float *corners_data = corners.data_ptr<float>();
	
	// What are the shapes of the input tensors
	int oB = out.sizes()[0];
	int oH = out.sizes()[1];
	int oW = out.sizes()[2];
	int oC = out.sizes()[3];
	int iB = in.sizes()[0];
	int iH = in.sizes()[1];
	int iW = in.sizes()[2];
	int iC = in.sizes()[3];
	int cB = corners.sizes()[0];
	int c4 = corners.sizes()[1];
	int c2 = corners.sizes()[2];
	printf("oB %d oH %d oW %d oC %d\n", oB, oH, oW, oC);
	printf("iB %d iH %d iW %d iC %d\n", iB, iH, iW, iC);
	printf("cB %d c4 %d c2 %d\n",           cB, c4, c2);
	
	// Shape Assertions
	if (oB!=iB || oB!=cB) {
		printf("ERROR: bicubic_float_cuda inconsistent batch sizes\n");
		exit(1);
	}
	if (iC!=oC) {
		printf("ERROR: bicubic_float_cuda inconsistent channels\n");
		exit(1);
	}
	if (c4!=4 || c2!=2) {
		printf("ERROR: bicubic_float_cuda corners must be [B 4 2]\n");
		exit(1);
	}
	
	// Assign the blocking schedule    dim3 is (x, y, z)
	dim3 threadsPerBlock(32, 16, 1); // 32x16 block in x-y plane, 4 deep in z
	dim3 numBlocks((oW+31)/32, (oH+15)/16, oB); 

	bicubic_float_kernel<<<numBlocks, threadsPerBlock>>>(
		iB, iH, iW, oH, oW, iC,
		out_data,
		in_data,
		corners_data);

	printf("END: bicubic_float_cuda\n");
}




//---------------------------
//  bicubic_aa_uint8_kernel
//      B iH iW oH oW C    input dimesions
//    out_data    [B oH oW C]   output image   (float32)   (0.0 to 1.0)
//     in_data    [B iH iW C]   input image    (uint8_t)   (0 to 255)
//  corners_data  [B 4 2]       corner points  (float32)
//   aa_ker_data  [B 2]   anti-alaising kernel   (int32)
//---------------------------
__global__
void bicubic_aa_uint8_kernel(
	int B, int iH, int iW, int oH, int oW, int C,
	float*          __restrict__ out_data,
	const uint8_t*  __restrict__ in_data,
	const float*    corners_data,
	const int*      aa_ker_data)
{
	// calculate rank
	int rank_x = blockIdx.x * blockDim.x + threadIdx.x;
	int rank_y = blockIdx.y * blockDim.y + threadIdx.y;
	int rank_z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (rank_x>=oW || rank_y>=oH || rank_z>=B)
		return;
	
	// Which 'slice' are we looking at?
	float   *out   =     out_data + ( rank_z * (oH*oW*C) );
	const uint8_t * in   =      in_data + ( rank_z * (iH*iW*C) );
	const float *corners = corners_data + ( rank_z * 8 );
	const int   * aa     =  aa_ker_data + ( rank_z * 2 );
	
	// How many anti-alaised batches are we doing?
	int AAY = aa[0];
	int AAX = aa[1];
	float out_y[4] = {0.0, 0.0, 0.0, 0.0};
	for (int iy=0; iy<AAY; iy++)
	{
		for (int ix=0; ix<AAX; ix++)
		{
			int rx = rank_x*AAX + ix;
			int ry = rank_y*AAY + iy;

			//------
			// Determine the (c_x,c_y) coordinate within the input image
			//------
			
			// What are the corner points of the image
			float c_x00 = corners[0];
			float c_y00 = corners[1];
			float c_x01 = corners[2];
			float c_y01 = corners[3];
			float c_x11 = corners[4];
			float c_y11 = corners[5];
			float c_x10 = corners[6];
			float c_y10 = corners[7];
			
			// What is delta location within the output image
			float delta_x = (rx+0.5) / (float)(oW*AAX);
			float delta_y = (ry+0.5) / (float)(oH*AAY);
			
			// Bilinear Interpolate between four corner points
			float c_x0 = c_x00 + delta_x * (c_x01-c_x00);
			float c_y0 = c_y00 + delta_x * (c_y01-c_y00);
			float c_x1 = c_x10 + delta_x * (c_x11-c_x10);
			float c_y1 = c_y10 + delta_x * (c_y11-c_y10);
			float c_x  = c_x0  + delta_y *  (c_x1-c_x0);
			float c_y  = c_y0  + delta_y *  (c_y1-c_y0);
			
			// Out of bounds
			c_x = clamp(c_x, 0.00001, iW-0.00001);
			c_y = clamp(c_y, 0.00001, iH-0.00001);
				
			//------
			// Solve the 'bicubic' interpolation coefficients
			//------
			
			// determine grid cell and delta
			int i_x = int(c_x);
			int i_y = int(c_y);
			float d_x = c_x - i_x;
			float d_y = c_y - i_y;
			float d_x2 = d_x*d_x;
			float d_y2 = d_y*d_y;
			float d_x3 = d_x2 * d_x;
			float d_y3 = d_y2 * d_y;
			
			
			// Indices for the 4x4 kernel   idx_y_x
			int idx_1_1 = i_y*iW*C + i_x*C;
			int idx_2_1 = idx_1_1 + iW*C;
			int idx_0_1 = (i_y>0)    ? (idx_1_1 - iW*C) : idx_1_1;
			int idx_3_1 = (i_y<iW-1) ? (idx_2_1 + iW*C) : idx_2_1;
			int idx_0_2 = idx_0_1 + C;
			int idx_1_2 = idx_1_1 + C;
			int idx_2_2 = idx_2_1 + C;
			int idx_3_2 = idx_3_1 + C;
			int idx_0_0 = idx_0_1 - C;
			int idx_1_0 = idx_1_1 - C;
			int idx_2_0 = idx_2_1 - C;
			int idx_3_0 = idx_3_1 - C;
			int idx_0_3 = idx_0_1 - C;
			int idx_1_3 = idx_1_1 - C;
			int idx_2_3 = idx_2_1 - C;
			int idx_3_3 = idx_3_1 - C;
			
			// For every channel
			//  (four polynomials)   b=bottom  t=top  l=left r=right
			//                     Y1  Y2  (derivatives)   y1 y2  (values)
			//                 this notation matches math above.
			for (int ic=0; ic<C; ic++)
			{
				//----
				// Horizontal cubic interpolation
				//----
				
				// Read derivatives
				float Y01 = (i_x>0)    ? ( 0.5*((float)in[idx_0_2]-(float)in[idx_0_0]) ) : ((float)in[idx_0_2]-(float)in[idx_0_1]);
				float Y02 = (i_x<iW-1) ? ( 0.5*((float)in[idx_0_3]-(float)in[idx_0_1]) ) : ((float)in[idx_0_2]-(float)in[idx_0_1]);
				float Y11 = (i_x>0)    ? ( 0.5*((float)in[idx_1_2]-(float)in[idx_1_0]) ) : ((float)in[idx_1_2]-(float)in[idx_1_1]);
				float Y12 = (i_x<iW-1) ? ( 0.5*((float)in[idx_1_3]-(float)in[idx_1_1]) ) : ((float)in[idx_1_2]-(float)in[idx_1_1]);
				float Y21 = (i_x>0)    ? ( 0.5*((float)in[idx_2_2]-(float)in[idx_2_0]) ) : ((float)in[idx_2_2]-(float)in[idx_2_1]);
				float Y22 = (i_x<iW-1) ? ( 0.5*((float)in[idx_2_3]-(float)in[idx_2_1]) ) : ((float)in[idx_2_2]-(float)in[idx_2_1]);
				float Y31 = (i_x>0)    ? ( 0.5*((float)in[idx_3_2]-(float)in[idx_3_0]) ) : ((float)in[idx_3_2]-(float)in[idx_3_1]);
				float Y32 = (i_x<iW-1) ? ( 0.5*((float)in[idx_3_3]-(float)in[idx_3_1]) ) : ((float)in[idx_3_2]-(float)in[idx_3_1]);
				
				// Read values
				float y01 = (float)in[idx_0_1];
				float y02 = (float)in[idx_0_2];
				float y11 = (float)in[idx_1_1];
				float y12 = (float)in[idx_1_2];
				float y21 = (float)in[idx_2_1];
				float y22 = (float)in[idx_2_2];
				float y31 = (float)in[idx_3_1];
				float y32 = (float)in[idx_3_2];
				
				// Interpolation coefficients
				//   a = 2y1 - 2y2 + y1' + y2'
				//   b = - 3y1 + 3y2 - 2y1' - y2'
				//   c = y1'
				//   d = y1
				float a0 = 2.0*y01 - 2.0*y02 + Y01 + Y02;
				float a1 = 2.0*y11 - 2.0*y12 + Y11 + Y12;
				float a2 = 2.0*y21 - 2.0*y22 + Y21 + Y22;
				float a3 = 2.0*y31 - 2.0*y32 + Y31 + Y32;
				float b0 = -3.0*y01 + 3.0*y02 - 2.0*Y01 - Y02;
				float b1 = -3.0*y11 + 3.0*y12 - 2.0*Y11 - Y12;
				float b2 = -3.0*y21 + 3.0*y22 - 2.0*Y21 - Y22;
				float b3 = -3.0*y31 + 3.0*y32 - 2.0*Y31 - Y32;
				float c0 = Y01;
				float c1 = Y11;
				float c2 = Y21;
				float c3 = Y31;
				float d0 = y01;
				float d1 = y11;
				float d2 = y21;
				float d3 = y31;
				
				// Cubic horizontal interpolation
				float y0 = a0*d_x3 + b0*d_x2 + c0*d_x + d0;
				float y1 = a1*d_x3 + b1*d_x2 + c1*d_x + d1;
				float y2 = a2*d_x3 + b2*d_x2 + c2*d_x + d2;
				float y3 = a3*d_x3 + b3*d_x2 + c3*d_x + d3;
				
				//----
				// Vertical bi-cubic interpolation
				//----
				
				// Get derivatives
				float Y1 = (i_y>0)    ? ( 0.5*(y2-y0) ) : ( y2-y1 );
				float Y2 = (i_y<iH-1) ? ( 0.5*(y3-y1) ) : ( y2-y1 );
				
				// Bi-cubic coefficients
				float a = 2.0*y1 - 2.0*y2 + Y1 + Y2;
				float b = -3.0*y1 + 3.0*y2 - 2.0*Y1 - Y2;
				float c = Y1;
				float d = y1;
				
				// Bi-cubic interpolation
				float y = a*d_y3 + b*d_y2 + c*d_y + d;
				
				//----
				// Store the output value
				//----
				out_y[ ic ] += y;
				
				// Next index (channel)
				idx_0_0++;	idx_0_1++;	idx_0_2++;	idx_0_3++;
				idx_1_0++;	idx_1_1++;	idx_1_2++;	idx_1_3++;
				idx_2_0++;	idx_2_1++;	idx_2_2++;	idx_2_3++;
				idx_3_0++;	idx_3_1++;	idx_3_2++;	idx_3_3++;
			}
		}
	}
	
	//----
	// Store the final value color
	//----
	for (int ic=0; ic<3; ic++) {
		float y = out_y[ic] / (AAY*AAX);
		//y = clamp(y, 0.0, 255.0);
		//uint8_t y8 = (uint8_t)y;
		//out[ idx_out ] = y8;
		int idx_out = rank_y*oW*C + rank_x*C + ic;
		out[ idx_out ] = y;
	}
}



//---------------------------
//  bicubic_aa_uint8_cuda
//    out    [B oH oW C]   output image   (float32)   (0.0 to 1.0)
//     in    [B oH oW C]   input image    (uint8)     (0 to 255)
//  corners  [B 4 2]       corner points  (float32)
//  aa_ker   [B 2]    anti-alaising kernel (int32)
//---------------------------
void bicubic_aa_uint8_cuda(
	torch::Tensor out,
	torch::Tensor in,
	torch::Tensor corners,
	torch::Tensor aa_ker)
{
	printf("BEGIN: bicubic_aa_uint8_cuda\n");
	
	// Pointer to the data
	float   *out_data     = out.data_ptr<float>();
	uint8_t *in_data      = in.data_ptr<uint8_t>();
	float *corners_data   = corners.data_ptr<float>();
	int     *aa_ker_data  = aa_ker.data_ptr<int>();
	
	// What are the shapes of the input tensors
	int oB = out.sizes()[0];
	int oH = out.sizes()[1];
	int oW = out.sizes()[2];
	int oC = out.sizes()[3];
	int iB = in.sizes()[0];
	int iH = in.sizes()[1];
	int iW = in.sizes()[2];
	int iC = in.sizes()[3];
	int cB = corners.sizes()[0];
	int c4 = corners.sizes()[1];
	int c2 = corners.sizes()[2];
	printf("oB %d oH %d oW %d oC %d\n", oB, oH, oW, oC);
	printf("iB %d iH %d iW %d iC %d\n", iB, iH, iW, iC);
	printf("cB %d c4 %d c2 %d\n",           cB, c4, c2);
	
	// Shape Assertions
	if (oB!=iB || oB!=cB) {
		printf("ERROR: bicubic_aa_uint8_cuda inconsistent batch sizes\n");
		exit(1);
	}
	if (iC!=oC) {
		printf("ERROR: bicubic_aa_uint8_cuda inconsistent channels\n");
		exit(1);
	}
	if (iC!=3 or iC!=4) {
		printf("ERROR: bicubic_aa_uint8_cuda channels not equal to 3 or 4\n");
		exit(1);
	}
	if (c4!=4 || c2!=2) {
		printf("ERROR: bicubic_aa_uint8_cuda corners must be [B 4 2]\n");
		exit(1);
	}
	
	// Assign the blocking schedule    dim3 is (x, y, z)
	dim3 threadsPerBlock(32, 16, 1); // 32x16 block in x-y plane, 4 deep in z
	dim3 numBlocks((oW+31)/32, (oH+15)/16, oB); 

	bicubic_aa_uint8_kernel<<<numBlocks, threadsPerBlock>>>(
		iB, iH, iW, oH, oW, iC,
		out_data,
		in_data,
		corners_data,
		aa_ker_data);

	printf("END: bicubic_aa_uint8_cuda\n");
}


//---------------------------
// gaussian_blur_trans_kernel
//   Performs horizontal gaussian blur and transposes 
//    the result.  Run this twice to get a full (separable)
//    horizontal/vertical gaussian blur.
//   Also divides by kernel integral, which handles
//    boundaries more accurately than edge padding.
//  input   [N H W 4]   RGBA float32
//  output  [N W H 4]   RGBA float32  (transposed!)
//  sigmas  [N]         blur kernel sigmas (0.5 to 2.0)
//    N H W             int32
//---------------------------

#define MAX_RADIUS 15 // Supports up to 15x15 kernels (Radius 7)
#define TILE_W 128   // Width of the processing tile

__global__ void gaussian_blur_h_trans_kernel(
	const float4* __restrict__ input,   // [N H W 4]
	float4* __restrict__	  output,   // [N W H 4]   Transposed!
	const float* sigmas,                // Per-image sigma [N]
	int N, int H, int W) 
{
	// Shared memory for the tile + halo
	// We only need to store RGB (float3) to save space, but float4 is easier for alignment
	extern __shared__ float4 tile[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.z;

	float sigma = sigmas[n];
	int radius = (int)ceilf(3.0f * sigma);
	if (radius > MAX_RADIUS) radius = MAX_RADIUS;

	// 1. Load Tile into Shared Memory with Halo
	int l_idx = threadIdx.x + radius; // Local index in shared memory
	
	if (y < H && n < N) {
		// Load center pixel
		int g_idx = n * (H * W) + y * W + x;
		tile[l_idx] = (x < W) ? input[g_idx] : make_float4(0,0,0,0);

		// Load left halo
		if (threadIdx.x < radius) {
			int left_x = x - radius;
			tile[threadIdx.x] = (left_x >= 0) ? input[g_idx - radius] : make_float4(0,0,0,0);
		}
		// Load right halo
		if (threadIdx.x >= blockDim.x - radius) {
			int right_x = x + radius;
			int right_l_idx = l_idx + radius;
			tile[right_l_idx] = (right_x < W) ? input[g_idx + radius] : make_float4(0,0,0,0);
		}
	}
	__syncthreads();

	if (x >= W || y >= H || n >= N) return;

	// 2. Compute Weighted Average with Renormalization
	float4 sum = make_float4(0,0,0,0);
	float weight_sum = 0.0f;

	for (int i = -radius; i <= radius; i++) {
		int neighbor_x = x + i;
		
		// Only process if within image bounds (Renormalization Logic)
		if (neighbor_x >= 0 && neighbor_x < W) {
			float weight = expf(-(float)(i * i) / (2.0f * sigma * sigma));
			float4 sample = tile[l_idx + i];
			
			sum.x += sample.x * weight;
			sum.y += sample.y * weight;
			sum.z += sample.z * weight;
			sum.w += sample.w * weight;
			weight_sum += weight;
		}
	}

	// 3. Normalize and Write
	int out_idx = n * (W * H) + x * H + y;
	output[out_idx] = make_float4(
		sum.x / weight_sum,
		sum.y / weight_sum,
		sum.z / weight_sum,
		sum.w / weight_sum
	);
}

//---------------------------
// gaussian_blur_trans_kernel
//   Performs horizontal gaussian blur and transposes 
//    the result.  Run this twice to get a full (separable)
//    horizontal/vertical gaussian blur.
//   Also divides by kernel integral, which handles
//    boundaries more accurately than edge padding.
//  input   [N H W 4]   RGBA float32
//  output  [N W H 4]   RGBA float32  (transposed!)
//  sigmas  [N]         blur kernel sigmas (0.5 to 2.0)
//---------------------------
void gaussian_blur_h_trans_cuda(
	torch::Tensor input,	      // [N H W 4]
	torch::Tensor output,         // [N W H 4]   Transposed!
	torch::Tensor sigmas)         // Per-image sigma [N]
{
	printf("BEGIN: gaussian_blur_h_trans_cuda\n");

	// Obtain data pointers
	float4* input_data  = input.data_ptr<float4>();
	float4* output_data = output.data_ptr<float4>();
	float * sigmas_data = sigmas.data_ptr<float>();
	
	// Obtain the dimensions
	int oN = output.sizes()[0];
	int oH = output.sizes()[1];
	int oW = output.sizes()[2];
	int oC = output.sizes()[3];
	int iN = input.sizes()[0];
	int iW = input.sizes()[1];
	int iH = input.sizes()[2];
	int iC = input.sizes()[3];
	int sN = sigmas.sizes()[0];
	printf("output [%d %d %d %d]  input [%d %d %d %d]  sigmas [%d]\n",
		oN, oH, oW, oC, iN, iW, iH, iC, sN);

	// Check for size mismatches
	if (oN!=iN || oN!=sN) {
		printf("ERROR: gaussian_blur_h_trans_cuda inconsistent batch sizes\n");
		exit(1);
	}
	if (oH!=iH || oW!=iW) {
		printf("ERROR: gaussian_blur_h_trans_cuda height and width must be transposed\n");
		exit(1);
	}
	if (iC!=4) {
		printf("ERROR: gaussian_blur_h_trans_cuda height and width must be transposed\n");
		exit(1);
	}

	dim3 threadsPerBlock(256, 1, 1); // 32x16 block in x-y plane, 4 deep in z
	dim3 numBlocks((iW+255)/256, iH, iN);
	int sm_size = (256+2*MAX_RADIUS)*16;
	
	gaussian_blur_h_trans_kernel<<<numBlocks, threadsPerBlock, sm_size>>>(
		input_data,   // [N H W 4]
		output_data,   // [N W H 4]   Transposed!
		sigmas_data,                // Per-image sigma [N]
		iN, iH, iW); 

	printf("END: gaussian_blur_h_trans_cuda\n");
}



//-----------------
//  Avalanche mixup!   by   Appleby  SMhasher.
//
// Pseudorandom from [0, 2^32-1]
//   Extremely fast, but one known flaw,
//    random(0) = 0
//   Also, it is invertable, so not cryptographic
//   But excellent statistical properties!
//-----------------

__device__ uint32_t random(uint32_t h)
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

__device__ float uint_to_float(uint32_t u) {
    // 0x3f800000 is the bit representation of 1.0f
    // 0x007fffff masks the 23 bits for the mantissa
    unsigned int res = 0x3f800000 | (u >> 9);
    return __uint_as_float(res) - 1.0f;
}

// Pseudorandom float32 from [0, 1)
__device__ float random_f(uint32_t seed)
{
	return uint_to_float(random(seed));
}


//------------------
//  David's High Performance erf-1(x)
//  First three coef from Mclauren series
//  Remaining coef were fit empirically using Desmos
//   erf-1(x) = Ax + Bx^3 + Cx^5 + Dx^9 + Ex^15 + Fx^63 + Gx^255
//
//   erf-1(x) = (A + (B + (C + (D + (E + (F + Gx^192)x^48)x^6)x^4)x^2)x^2)x
//
//   A = 0.8862269254527579
//   B = 0.2320136665346544
//   C = 0.1275561753055979
//   D = 0.17
//   E = 0.3
//   F = 0.3
//   G = 0.38
//-------------------
__device__ float erf_inv_david(float x) {
	float x2   = x*x;
	float x4   = x2*x2;
	float x8   = x4*x4;
	float x16  = x8*x8;
	float x32  = x16*x16;
	float x64  = x32*x32;
	float x128 = x64*x64;
	float x6   = x4*x2;
	float x48  = x32*x16;
	float x192 = x128*x64;
	return   (0.8862269254527579 + (0.2320136665346544 + (0.1275561753055979 + (0.17 + (0.3 + (0.3 + 0.38*x192)*x48)*x6)*x4)*x2)*x2)*x;
}

// Pseudorandom standard normal
__device__ float random_n(uint32_t seed) {
	float x = random_f(seed);
	return 1.41421356237 * erf_inv_david(2.0*x - 1);
}



//--------------------------
// Photometric Kernel: Operates on float32 RGBA in range [0, 1].
// Performs: Naive Solarization, HSV-based Color Jitter, and Noise.
// Final pass performs Clamping and Mean/Std Normalization.
//
//  input     [N H W 4]  float32    input image (0 to 1)
//  output    [N H W 4]  float32    output image (0 to 1)
//  h_shifts  [N]        float32    hue shift
//  s_factors [N]        float32    saturation factor
//  v_fators  [N]        float32    value factor
//  sol_thresholds [N]   float32    threshold for solarization
//  noise_scales [N]     float32    noise scales to add
//  mean    [3]          float32    mean values          (normalization)
//  std     [3]          float32    standard deviations  (normalization)
//  N H W                int32
//  seed                            seed for normal distribution hashing
//--------------------------
__global__ void augment_photometric_kernel(
	const float4* __restrict__ input,
	float4* __restrict__ output,
	const float* h_shifts,       // Hue shift array [N]  (0 to 360.0)
	const float* s_factors,      // Saturation factor array [N]
	const float* v_factors,      // Value (Brightness) factor array [N]
	const float* sol_thresholds, // Solarization threshold array [N]
	const float* noise_scales,   // Noise scales for gaussian additive noise
	const float* mean,           // Pre-calculated normalization mean
	const float* std,            // Pre-calculated normalization std
	int N, int H, int W,
	uint32_t seed) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.z;

	if (x >= W || y >= H || n >= N) return;

	// Which seed for normal random numbers?
	uint32_t myseed = seed + n*(H*W*3) + y*(W*3) + x*3;

	size_t idx = (size_t)n * H * W + (size_t)y * W + x;

	// 1. Vectorized Load (float4)
	float4 pixel = input[idx];
	float r = pixel.x;
	float g = pixel.y;
	float b = pixel.z;

	// 2. Naive Solarization (Handles ringing/overshoot naturally)
	if (r > sol_thresholds[n]) r = 1.0f - r;
	if (g > sol_thresholds[n]) g = 1.0f - g;
	if (b > sol_thresholds[n]) b = 1.0f - b;

	// 3. HSV Jitter logic
	// We must clamp to [0, 1] ONLY for the HSV conversion to prevent NaNs
	float rc = clamp(r, 0.0, 1.0);
	float gc = clamp(r, 0.0, 1.0);
	float bc = clamp(r, 0.0, 1.0);

	// RGB to HSV (Simplified for jitter)
	float max_c = fmaxf(rc, fmaxf(gc, bc));
	float min_c = fminf(rc, fminf(gc, bc));
	float delta = max_c - min_c;

	float h = 0.0f;
	if (delta > 0.0f) {
		if (max_c == rc) h = 60.0f * (fmodf(((gc - bc) / delta), 6.0f));
		else if (max_c == gc) h = 60.0f * (((bc - rc) / delta) + 2.0f);
		else h = 60.0f * (((rc - gc) / delta) + 4.0f);
	}
	float s = (max_c == 0.0f) ? 0.0f : delta / max_c;
	float v = max_c;

	// Apply Jitter
	h = fmodf(h + h_shifts[n], 360.0f);
	if (h < 0.0f) h += 360.0f;
	s = s * s_factors[n];
	s = clamp(s, 0.0, 1.0);
	v = v * v_factors[n]; // Let V overshoot if needed for effect

	// HSV back to RGB
	float c = v * s;
	float x_h = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
	float m = v - c;

	if (h < 60)       { r = c; g = x_h; b = 0; }
	else if (h < 120) { r = x_h; g = c; b = 0; }
	else if (h < 180) { r = 0; g = c; b = x_h; }
	else if (h < 240) { r = 0; g = x_h; b = c; }
	else if (h < 300) { r = x_h; g = 0; b = c; }
	else              { r = c; g = 0; b = x_h; }
	r += m; g += m; b += m;

	// Apply Gaussian Noise
	float noise_scale = noise_scales[n];
	r = r + noise_scale * random_n(myseed);
	g = g + noise_scale * random_n(myseed+1);
	b = b + noise_scale * random_n(myseed+2);

	// 4. Final Normalization & Clamp
	// We clamp the final result to ensure model stability
	r = clamp(r, 0.0, 1.0);
	g = clamp(g, 0.0, 1.0);
	b = clamp(b, 0.0, 1.0);

	pixel.x = (r - mean[0]) / std[0];
	pixel.y = (g - mean[1]) / std[1];
	pixel.z = (b - mean[2]) / std[2];
	// pixel.w (Alpha) usually remains unnormalized or set to 1.0

	output[idx] = pixel;
}


//--------------------------
// Photometric Kernel: Operates on float32 RGBA in range [0, 1].
// Performs: Naive Solarization, HSV-based Color Jitter, and Noise.
// Final pass performs Clamping and Mean/Std Normalization.
//
//  input     [N H W 4]  float32    input image (0 to 1)
//  output    [N H W 4]  float32    output image (0 to 1)
//  h_shifts  [N]        float32    hue shift
//  s_factors [N]        float32    saturation factor
//  v_fators  [N]        float32    value factor
//  sol_thresholds [N]   float32    threshold for solarization
//  noise_scales [N]     float32    noise scales to add
//  mean                 float3     mean values          (normalization)
//  std                  float3     standard deviations  (normalization)
//  N H W                int32
//  seed                            seed for normal distribution hashing
//  
//  return    updated seed
//--------------------------
unsigned int augment_photometric_cuda(
	torch::Tensor input,
	torch::Tensor output,
	torch::Tensor h_shifts,       // Hue shift array [N]  (-180.0 to 180.0)
	torch::Tensor s_factors,      // Saturation factor array [N]
	torch::Tensor v_factors,      // Value (Brightness) factor array [N]
	torch::Tensor sol_thresholds, // Solarization threshold array [N]
	torch::Tensor noise_scales,   // Noise scales for gaussian additive noise
	torch::Tensor mean,           // Pre-calculated normalization mean
	torch::Tensor std,            // Pre-calculated normalization std
	unsigned int seed)
{
	printf("BEGIN augment_photometric_cuda\n");
	
	// Pointer into the data
	float4 *input_data          = input.data_ptr<float4>();
	float4 *output_data         = output.data_ptr<float4>();
	float  *h_shifts_data       = h_shifts.data_ptr<float>();
	float  *s_factors_data      = s_factors.data_ptr<float>();
	float  *v_factors_data      = v_factors.data_ptr<float>();
	float  *sol_thresholds_data = sol_thresholds.data_ptr<float>();
	float  *noise_scales_data   = noise_scales.data_ptr<float>();
	float  *mean_data           = mean.data_ptr<float>();
	float  *std_data            = std.data_ptr<float>();
	
	// Extract data shapes
	int iN  = input.sizes()[0];
	int iH  = input.sizes()[1];
	int iW  = input.sizes()[2];
	int i4  = input.sizes()[3];
	int oN  = output.sizes()[0];
	int oH  = output.sizes()[1];
	int oW  = output.sizes()[2];
	int o4  = output.sizes()[3];
	int hN  = h_shifts.sizes()[0];
	int sN  = s_factors.sizes()[0];
	int vN  = v_factors.sizes()[0];
	int soN = sol_thresholds.sizes()[0];
	int noN = noise_scales.sizes()[0];
	int meN = mean.sizes()[0];
	int stN = std.sizes()[0];
	printf("input (%d %d %d %d) output (%d %d %d %d) h %d s %d v %d so %d no %d me %d st %d\n",
		iN, iH, iW, i4,   oN, oH, oW, o4,   hN, sN, vN,  soN, noN, meN, stN);
	
	// Compare sizes
	if (iN!=oN || iH!=oH || iW!=oW || i4!=o4) {
		printf("ERROR augment_photometric_cuda input/output shapes mismatch\n");
		exit(1);
	}
	if (i4!=o4 || i4!=4) {
		printf("ERROR augment_photometric_cuda expected 4 channels\n");
		exit(1);
	}
	if (iN!=oN || iN!=hN || iN!=sN || iN!=vN || iN!=soN || iN!=noN || iN!=meN || iN!=stN) {
		printf("ERROR augment_photometric_cuda inconsistent batch sizes\n");
		exit(1);
	}

	// Run the kernel
	dim3 threadsPerBlock(32, 32, 1); // 32x16 block in x-y plane, 4 deep in z
	dim3 numBlocks((iW+31)/32, (iH+31)/32, iN);

	augment_photometric_kernel<<<threadsPerBlock, numBlocks>>>(
		input_data,
		output_data,
		h_shifts_data,       // Hue shift array [N]  (0 to 360.0)
		s_factors_data,      // Saturation factor array [N]
		v_factors_data,      // Value (Brightness) factor array [N]
		sol_thresholds_data, // Solarization threshold array [N]
		noise_scales_data,   // Noise scales for gaussian additive noise
		mean_data,           // Pre-calculated normalization mean
		std_data,            // Pre-calculated normalization std
		iN, iH, iW,
		seed);

	printf("END   augment_photometric_cuda\n");
	
	// Return updated seed
	return seed + iN*iH*iW*3; 
}



//--------------------------
// restore_uint8_features_kernel
//
//   Restores the DINO features 
//     from    uint8     frames
//     to      float32   tensor
//
//   input   [N frame]   (utf8)
//   output  [N H W C]
//
//   frame format:
//     header   minvals  maxvals  payload
//
//   header:    H (uint32)  W (uint32)  C (uint32)
//   minvals:   [N]  (float16)
//   maxvals:   [N]  (float16)
//   payload:   [H W C]  (uint8)  0 minval  255 maxval
//--------------------------

void restore_uint8_features_kernel(
	const uint8_t* __restrict__ input_data,
	float*         __restrict__ output_data,
	int N, int H, int W, int C, int F)
{
	// Get the rank
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.z;

	if (x >= W || y >= H || n >= N)
		return;

	// Byte offsets in the frame
	uint32_t b0 = 0;
	uint32_t b1 = 12;
	uint32_t b2 = b1 + 2*N;
	uint32_t b3 = b2 + 2*N;
	uint32_t b4 = b3 + H*W*C;
	
	// Pointer to the correct image offset
	uint8_t* input  = input_data + n*b4;
	float*   output = output_data + n*H*W*C;
	
	// Pointer to the correct frame offsets
	__half* minvals = reinterpret_cast<__half*>(   input + b1  );
	__half* maxvals = reinterpret_cast<__half*>(   input + b2  );
	uint8_t* indata = input + b3;

	//-----
	// Restore the data!   (from uint8 to float32)
	//-----
	uint32_t idx = y*W*C + x*C;
	for (c=0; c<C; c++) {
		float   minval = (float)minvals[n];
		float   maxval = (float)maxvals[n];
		float   val    = (float)indata[idx];
		val = minval + (maxval-minval)*(1.0/255.0)*val;
		outdata[idx] = val;
		idx++;
	}
}


//--------------------------
// restore_uint8_features_cuda
//
//   Restores the DINO features 
//     from    uint8     frames
//     to      float32   tensor
//
//   input   [N frame]   (utf8)
//   output  [N H W C]
//
//   frame format:
//     header   minvals  maxvals  payload
//
//   header:    H (uint32)  W (uint32)  C (uint32)
//   minvals:   [N]  (float16)
//   maxvals:   [N]  (float16)
//   payload:   [H W C]  (uint8)  0 minval  255 maxval
//--------------------------

void restore_uint8_features_cuda(
	torch::Tensor input,
	torch::Tensor output)
{
	printf("BEGIN restore_uint8_features_cuda\n");
	
	input_data  = input.data_ptr<float>();
	output_data = output.data_ptr<uint8>();
	
	// Read lengths
	iN = input.sizes()[0];
	iH = input.sizes()[1];
	iW = input.sizes()[2];
	iC = input.sizes()[3];
	oN = output.sizes()[0];
	oFrame = output.sizes()[1];
	
	// Calculate frame size
	uint32_t len_header = 12;
	uint32_t len_minvals = 2*iN;
	uint32_t len_maxvals = 2*iN;
	uint32_t len_payload = iH*iW*iC;
	uint32_t len_frame = len_header + len_minvals + len_maxvals + len_payload;
	
	// Shape checking
	if (iN!=oN) {
		printf("ERROR restore_uint8_features_cuda batch size mismatch\n");
		exit(1);
	}
	if (oFrame!=len_frame) {
		printf("ERROR restore_uint8_features_cuda unexpected frame size\n");
		exit(1);
	}
	

	// Run the kernel
	dim3 threadsPerBlock(32, 32, 1); // 32x16 block in x-y plane, 4 deep in z
	dim3 numBlocks((iW+31)/32, (iH+31)/32, iN);

	restore_uint8_features_kernel<<<threadsPerBlock, numBlocks>>>(
		input_data,
		output_data,
		iN, iH, iW, iC, oFrame);


	printf("END   restore_uint8_features_cuda\n");
}


//--------------------------
//  Rand-augment round
//--------------------------

//void rand_agument_cuda()


