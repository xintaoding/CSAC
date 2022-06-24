#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <math.h>

#include "../../common/book.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include<device_launch_parameters.h>
//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)

#include <cstdio>
#include <cstdlib>
#include <device_functions.h>
#include <time.h>
#include<cuda_texture_types.h>
#include<texture_fetch_functions.h>

#include <curand_kernel.h>

#include <vector_types.h>//used for float2,3,4 etc. type
#include <thrust/device_vector.h>//used for thrust library
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

//#include <thrust/remove.h>
//#include <thrust/execution_policy.h>

//#include <random>
//#include <iomanip>
//#include <algorithm>
//#include <set>
//#include <math_helper.h>
//#include "./cutil_math.h"
//#include <thrust/extrema.h>

//#include <C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc\helper_cuda.h>
//some math function such as cross product,
//in some early edition before CUDA5.0 it is included in  <math_helper.h>
//#include <C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc\helper_string.h>
#define imin(a,b) (a<b?a:b)
#define N 13
#define ndof 9//the elements number of the matrix for svd decomposition
#define N10 10
#define essential_fd 5
#define N3 3//size for 3*3 matrix
#define P_CORE 0.2
#define thresh_sampson 2.0//Sampson Distance: Fundamental Matrix Estimation: A Study of Error Criteria
#define m_maxIterationsPerRow 40
#define epsilon 2.220446e-16

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
using namespace std;

/* open file (from "fp") count the number of data in the file
* Returns number of ints actually read, EOF on end of file, EOF-1 on error*/
int return_datanum(FILE *fp)
{
	int j, n = 0;
	float vals;
	while (!feof(fp)) {
		j = fscanf(fp, "%e", &vals);
		//用 法:int fscanf(FILE *stream, char *format,[argument...]);
		//返回值：整型，数值等于[argument...]的个数
		if (j == EOF)
			break;
		n += j;
	}

	return n;
}

/* open file (from "fp") count the number of data in the file
* Returns number of rows actually read, EOF on end of file, EOF-1 on error
*/
int return_lineno(FILE *fp)
{
	char ch;
	int lineno = 0;
	while (!feof(fp)) {
		if ((ch = fgetc(fp)) == '\n')
			++lineno;
	}
	return lineno;
}

/* reads (from "fp") "nvals" ints into "vals".
* Returns number of ints actually read, EOF on end of file, EOF-1 on error
*/
int readNInts(FILE *fp, int *vals, int nvals, int matlab_flag)
{
	register int i;
	int n, j;
	float x;

	for (i = n = 0; i<nvals; ++i) {
		j = fscanf(fp, "%e", &x);
		if (matlab_flag)
			*(vals + i) = (int)x - 1;
		else
			*(vals + i) = (int)x;
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}

	return n;
}

/* reads (from "fp") "nvals" doubles into "vals".
* Returns number of doubles actually read, EOF on end of file, EOF-1 on error
*/
static int readNfloats(FILE *fp, float *vals, int nvals, int matlab_flag)
{
	register int i;
	int n, j;

	for (i = n = 0; i<nvals; ++i) {
		j = fscanf(fp, "%f", vals + i);
		if (matlab_flag)
			*(vals + i) -= 1.0;
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}
	//for(i=0;i<nvals;++i)
	//	  printf("%lf",*(vals+i));//辅助输出语句
	return n;
}

static int readNDoubles(FILE *fp, double *vals, int nvals, int matlab_flag)
{
	register int i;
	int n, j;

	for (i = n = 0; i<nvals; ++i) {
		j = fscanf(fp, "%lf", vals + i);
		if (matlab_flag)
			*(vals + i) -= 1.0;
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}
	//for(i=0;i<nvals;++i)
	//	  printf("%lf",*(vals+i));//辅助输出语句
	return n;
}

//==============================================
__constant__ int dim_m_I_cn24[7], ch_I0s1[2], ch_I0s2[2], ch_I1s1[2], ch_I1s2[2], I0interchange[2], I1interchange[2], dev_n_round[1];
//dim_m_I_cn24[0] is the number of matches pairs. 
//dim_m_I_cn24[1] is the number of columns of I0, dim_m_I_cn24[2] is the number of columns of I1. They are used for the pointer to index the pixel location linearly
//dim_m_I_cn24[3]=extention(c(n,2)), n is the number of matches pairs.
//Extend cn2 to 2^k for bitonic sort since original bitonic algorithm can only cope with vector with length 2^k.
//dim_m_I_cn24[4]=(int)(log(dim_m_I_cn24[3]) / log(2)) used for bitonic sort.
//dim_m_I_cn24[5] is the half of ext_cn2, used to control bitonic cycle.
//dim_m_I_cn24[6]=c(N,4) is the top c(N,4) small distances of matches pairs produced by c(n,2) sampling after bitonic sort, used to extract matches core.
//ch_I0s1 etc. are temporary variables used in bresenham algorithm

//dev_matches is the matches resulting in image2, which will used as share memory variable.
//dev_frames0x etc. are x and y frame coordinations resulting in image1 and 2 respectively.
//dev_cn2_error is the resulting distance error of cn2 matches.
//dev_ij_in_ind=ind(i,j) is the global index of elements in dev_cn2_error
//the correspondence of ind(i,j) and i, j is ind(i,j)=sum_(k=0)^(i-1)(n-k-1)+j-i-1
//dev_ij_in_ind has two columns using int2 type,
//in which the first column dev_ij_in_ind.x recorded i index in matches
//and the second column dev_ij_in_ind.y recorded j index in matches.
__global__ void extract_parallel_dist_cn2(float *frames0x, float *frames0y, float *frames1x, float *frames1y, float *Image0, float *Image1, float *dev_cn2_error, int2 *dev_ij_in_ind)
{
	unsigned int tid = blockIdx.x;
	for (int i = tid; i <dim_m_I_cn24[0] - 1; i += gridDim.x) {
		int coord_pi_I0_x, coord_pi_I0_y, coord_pi_I1_x, coord_pi_I1_y, I0_x, I0_y;//y is denoted as I0_y
		coord_pi_I0_x = round(frames0x[i]);/*frames0(1,pair_i_I0), x coordinate*/
		coord_pi_I0_y = round(frames0y[i]);/*frames0(2,pair_i_I0), y coordinate*/
		coord_pi_I1_x = round(frames1x[i]);/*frames1(1,pair_i_I1), x coordinate*/
		coord_pi_I1_y = round(frames1y[i]);/*frames1(2,pair_i_I1), y coordinate*/
		I0_x = coord_pi_I0_x;//x is denoted as I0_x
		I0_y = coord_pi_I0_y;//y is denoted as I0_y

		for (int j = threadIdx.x + 1; j <dim_m_I_cn24[0]; j += blockDim.x) {
			if (j > i) {
				int coord_pj_I0_x, coord_pj_I0_y, coord_pj_I1_x, coord_pj_I1_y, I1_x, I1_y;//y is denoted as I1_y
				int n = 0;
				double  temp = 0.0f;
				coord_pj_I0_x = round(frames0x[j]);/*frames0(1,pair_j_I0), x coordinate*/
				coord_pj_I0_y = round(frames0y[j]);/*frames0(2,pair_j_I0), y coordinate*/
				coord_pj_I1_x = round(frames1x[j]);/*frames1(1,pair_j_I1), x coordinate*/
				coord_pj_I1_y = round(frames1y[j]);/*frames1(2,pair_j_I1), y coordinate */
				I1_x = coord_pi_I1_x;//x is denoted as I1_x
				I1_y = coord_pi_I1_y;//y is denoted as I1_y

									 //                                    (pi_I1_x,pi_I1_y)
									 //                               ___---
									 //                      i  ___---
									 //(pi_I0_x, pi_I0_y)___---
									 //
									 //
									 //(pj_I0_x, pj_I0_y)---___
									 //            ---___  j
									 //                  ---___
									 //                        ---___
									 //                              (pj_I1_x, pj_I1_y)
				int d0x = abs(coord_pj_I0_x - coord_pi_I0_x);//x1-x0
				int d0y = abs(coord_pj_I0_y - coord_pi_I0_y);//y1-y0
				int dl0 = d0x*(d0x > d0y) + d0y*(d0x <= d0y);//dl0 = (d0x>d0y ? d0x : d0y);
				int d1x = abs(coord_pj_I1_x - coord_pi_I1_x);
				int d1y = abs(coord_pj_I1_y - coord_pi_I1_y);
				int dl1 = d1x*(d1x > d1y) + d1y*(d1x <= d1y); //dl1 = (d1x>d1y ? d1x : d1y);
				if (dl0 < 5 || dl1 < 5)
				{
					temp = 1000.0f;
					//continue;
				}
				else
				{
					int I0_dx = d0x;
					int I0_dy = d0y;
					int I1_dx = d1x;
					int I1_dy = d1y;
					int dl, ds;
					int I0p_temp, I1p_temp;
					float I0_cur, I1_cur, I0_next, I1_next, I0_temp, I1_temp, I0_cur_4_diff, I1_cur_4_diff;
					double coe;
					int frag;//, step_count = 0

					d0x = I0_dy * (I0_dy > I0_dx) + I0_dx * (I0_dy <= I0_dx);//if I0_dy > I0_dx, exchange dx and dy, let the line grow in y-direction
					d0y = I0_dx * (I0_dy > I0_dx) + I0_dy * (I0_dy <= I0_dx);
					int I0p = (d0y << 1) - d0x;//ek eq.(7)

					d1x = I1_dy * (I1_dy > I1_dx) + I1_dx * (I1_dy <= I1_dx);
					d1y = I1_dx * (I1_dy > I1_dx) + I1_dy * (I1_dy <= I1_dx);
					int I1p = (d1y << 1) - d1x;//ek eq.(7)

					if (dl0 > dl1)
					{
						dl = dl0;
						ds = dl1;
					}
					else
					{
						dl = dl1;//dl is the longer length between l0 and l1
						ds = dl0;//ds is the shorter length between l0 and l1
					}
					//l = (dl0 > dl1)*dl0 + (dl0 <= dl1)*dl1;
					int sl0 = dl0 < dl1;
					//coe = (dl0 - 1) / (dl1 - 1.0)*sl0 + (dl1 - 1) / (dl0 - 1.0) * (1 - sl0);
					if (sl0)
						coe = ((double)(dl0)) / ((double)(dl1));//eq. (5)
					else
						coe = ((double)(dl1)) / ((double)(dl0));
					//I0_temp = tex2D(Image1, I0_x - 1, I0_y - 1);
					//I1_temp = tex2D(Image2, I1_x - 1, I1_y - 1);
					I0_temp = *(Image0 + I0_y * dim_m_I_cn24[1] + I0_x);// I0 + i*NI0 + j
					I1_temp = *(Image1 + I1_y * dim_m_I_cn24[2] + I1_x);
					I0_cur_4_diff = I0_temp;
					I1_cur_4_diff = I1_temp;
					temp = 0.0f;
					//		temp += fabsf(I0_temp - I1_temp);// (I0_temp - I1_temp)*(I0_temp >= I1_temp) + (I1_temp - I0_temp)*(I0_temp < I1_temp);
					//I0interchange_m = I0interchange[I0_dy > I0_dx];
					dl0 = I0interchange[I0_dy > I0_dx];//x1>x0,temp variable of I0interchange_m, dl0 is used for saving memory of GPU
					dl1 = I1interchange[I1_dy > I1_dx];//y1>y0,temp variable of I1interchange_m
					I0_dx = ch_I0s1[coord_pj_I0_x > coord_pi_I0_x];//x1>x0,temp variable of I0s1
					I0_dy = ch_I0s2[coord_pj_I0_y > coord_pi_I0_y];//y1>y0,temp variable of I0s2
					I1_dx = ch_I1s1[coord_pj_I1_x > coord_pi_I1_x];//temp variable of I1s1
					I1_dy = ch_I1s2[coord_pj_I1_y > coord_pi_I1_y];//temp variable of I1s2
					int et0 = 1;//assistant variable
					int et1 = 1;
					int I0_x0 = I0_x;
					int I0_y0 = I0_y;
					for (n = 1; n < dl; n++)
					{
						if (et0 + 1 - sl0)//update variables in I0 using eq. (6)如果I0上的线段短
						{
							I0_cur = I0_temp;//eq. (6)
							I0_x0 += I0_dx*(1 - dl0);//I0_dx is the denotion gxk in eq. (6). dl0 is the denotion exk in eq. (6)
							I0_y0 += I0_dy *dl0;
							I0_y0 += I0_dy *(1 - dl0)*(I0p >= 0);//I0p is the denotion ek in eq. (6)
							I0_x0 += I0_dx *dl0 * (I0p >= 0);

							I0p_temp = I0p;
							I0p += ((d0y << 1) - (d0x << 1))*(I0p_temp >= 0);
							I0p += (d0y << 1)*(I0p_temp < 0);//update ek
							I0_next = *(Image0 + I0_y0 * dim_m_I_cn24[1] + I0_x0);
							//Image1 is read out in C system, ===========used for debug
							//I0_next = I0[(I0_y0-1)  * dim_m_I[1] + (I0_x0-1) ];//Image1 is read out in matlab system
							//I0_next = tex2D(Image1, I0_x0 - 1, I0_y0 - 1);
							//printf("n=%d, (I0_x,I0_y)=(%d,%d), I0_cur=%f  ", n, I0_x0, I0_y0, I0_next);

						}
						if (et1 + sl0)//如果I1上的线段长
						{//update variables in I1 using eq. (6)
							I1_cur = I1_temp;
							I1_x += I1_dx *(1 - dl1);
							I1_y += I1_dy *dl1;
							I1_y += I1_dy *(1 - dl1)*(I1p >= 0);
							I1_x += I1_dx *dl1 *(I1p >= 0);
							I1p_temp = I1p;
							I1p += ((d1y << 1) - (d1x << 1))*(I1p_temp >= 0);
							I1p += (d1y << 1)*(I1p_temp < 0);
							I1_next = Image1[I1_y *dim_m_I_cn24[2] + I1_x];
							//I1_next = tex2D(Image2, I1_x - 1, I1_y - 1);
						}
						//frag =  (int)(n*coe + 0.5) - (int)((n - 1)*coe + 0.5);
						frag = (int)(n*coe) - (int)((n - 1)*coe);//used for pixel growth
																 //l1: * * -> l1_elong * * *
																 //l2: * * *
																 //I(l1_elong(0))=I(l1(0));
																 //I(l1_elong[1])=I(l1([coe*1]-[coe*(1-1)]))=I(l1([2/3]-[0]))=I(l1(0-0))=I(l1(0));
																 //I(l1_elong[2])=I(l1([coe*2]-[coe*(2-1)]))=I(l1([2/3*2]-[2/3]))=I(l1(1-0))=I(l1(1));
																 //l1: * *-> l1_elong * * * * * * * * *
																 //l2: * * * * * * * * *
																 //I(l1_elong(0))=I(l1(0));
																 //frag(n=1)=[coe*1]-[coe*(1-1)]=[2/9]-[0]=0; 
																 //frag(n=2)=[coe*2]-[coe*(2-1)]=[2/9*2]-[2/9]=0;
																 //frag(n=3)=[coe*3]-[coe*(3-1)]=[2/9*3]-[2/9*2]=0;
																 //frag(n=4)=[coe*4]-[coe*(4-1)]=[2/9*4]-[2/9*3]=0;
																 //frag(n=5)=[coe*5]-[coe*(5-1)]=[2/9*5]-[2/9*4]=1;skip 
																 //frag(n=6)=[coe*6]-[coe*(6-1)]=[2/9*6]-[2/9*5]=0; 
																 //frag(n=7)=[coe*7]-[coe*(7-1)]=[2/9*7]-[2/9*6]=1-1=0;
																 //frag(n=8)=[coe*8]-[coe*(8-1)]=[2/9*8]-[2/9*7]=1-1=0;
						if (frag)
						{
							et0 = sl0;
							et1 = 1 - sl0;
						}
						else
						{
							et0 = 1 - sl0;
							et1 = sl0;
						}
						if (sl0 > 0)
						{
							I0_temp = I0_cur*(1 - frag) + I0_next*frag;
							//	temp += fabsf(I0_temp - I1_next);
							I1_temp = I1_next;
						}
						else
						{
							I1_temp = I1_cur*(1 - frag) + I1_next*(frag);
							//	temp += fabsf(I0_next - I1_temp);
							I0_temp = I0_next;
						}
						if (frag)
						{
							temp += fabsf(fabsf(I0_cur_4_diff - I0_next) - fabsf(I1_cur_4_diff - I1_next));
							//	step_count++;
							//	if (dim_m_I_cn24[0]>500)
							//		if (temp / step_count > 0.02)//break unnecessary error calculation
							//			break;
							I1_cur_4_diff = I1_next;
							I0_cur_4_diff = I0_next;
						}

					}/*for (n = 1; n < dl; n++)*/
					 //					temp = temp / dl;
					temp = temp / ds;
				}//else			{
				 //================write back temp to distance vector dev_cn2_error
				I1_y = 0;
				for (tid = 0; tid < i; tid++)
					I1_y += dim_m_I_cn24[0] - 1 - tid;
				I1_y += j - i - 1;
				//atomicExch(dev_cn2_error + I1_y, temp);
				*(dev_cn2_error + I1_y) = temp;
				dev_ij_in_ind[I1_y].x = i;//write back index
				dev_ij_in_ind[I1_y].y = j;
			}//if(j>i){
		}/*for (j = i + 1; j <L; j++) {*/
		 //		__syncthreads();//syncthrize in block
		 //		__threadfence();//syncthrize in grid
	}/*for (i = tid; i <L-1;i+= gridDim.x * blockDim.x) {*/
}/*Do the job*/

 //==================Bitonic sort to choose matches core
__global__ void Bitonic_sort_ex_mc(float *dev_cn2_error, unsigned int *dev_cn2_global_ind, int i, int j)
{
	int coord_pi_I1_x, coord_pi_I1_y, I1_x, I1_y;//y is denoted as I1_y
	unsigned int tid;
	//dim_m_I_cn24[3]=ext_cn2,which is the size of vector to sort. 
	//dim_m_I_cn24[4] = (int)(log(dim_m_I_cn24[3]) / log(2));dim_m_I_cn24[5]=c(N,2)/2
	//dim_m_I_cn24[5] is the half of ext_cn2, used to control bitonic cycle.
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	//	for (tid = blockIdx.x * blockDim.x + threadIdx.x; tid < dim_m_I_cn24[3]; tid += gridDim.x * blockDim.x)
	while (tid < dim_m_I_cn24[5])
	{
		coord_pi_I1_x = (tid / (1 << (j - 1)))*(1 << j) + (tid % (1 << (j - 1)));
		coord_pi_I1_y = (coord_pi_I1_x / (1 << i)) % 2;
		I1_x = (coord_pi_I1_y == 0) ? coord_pi_I1_x : (coord_pi_I1_x + (1 << (j - 1)));
		I1_y = (coord_pi_I1_y == 0) ? (coord_pi_I1_x + (1 << (j - 1))) : coord_pi_I1_x;
		if (dev_cn2_error[I1_y] < dev_cn2_error[I1_x])
		{//write down index to save produce time
		 //Compile kernel code for Compute 2.0 and above only
#if __CUDA_ARCH__ >=200
			atomicExch(&dev_cn2_global_ind[I1_x], atomicExch(&dev_cn2_global_ind[I1_y], dev_cn2_global_ind[I1_x])); //Device functions on the GPU
																													//				atomicMin(&dev_cn2[I1_x], atomicMax(&dev_cn2[I1_y], dev_cn2[I1_x]));
			atomicExch(&dev_cn2_error[I1_x], atomicExch(&dev_cn2_error[I1_y], dev_cn2_error[I1_x]));
#endif


		}
		tid += gridDim.x * blockDim.x;
		//	__syncthreads();
	}//while (tid < dim_m_I_cn24[5])
	__syncthreads();
}/*Do the job*/


 /*
 * K1, K2 are 3x3 intrinsics matrices
 * points is matrix of size |N| x 4
 * Assume K = [k11 k12 k13
 *              0  k22 k23
 *              0   0   1]
 */
void calibratePoints(double *K1, double *K2, float *frames0x, float *frames0y, float *frames1x, float *frames1y, float *nrmlz_frames0x, float *nrmlz_frames0y, float *nrmlz_frames1x, float *nrmlz_frames1y, int n_matches) {
	double *k1;
	k1 = K1;
	double inv1_k11 = (1.0 / k1[0]); // 1 / k11
	double inv1_k12 = (-k1[1] / (k1[0] * k1[4])); // -k12 / (k11*k22)
												  // (-k13*k22 + k12*k23) / (k11*k22)
	double inv1_k13 = ((-k1[2] * k1[4] + k1[1] * k1[5]) / (k1[0] * k1[4]));
	double inv1_k22 = (1.0 / k1[4]); // 1 / k22
	double inv1_k23 = (-k1[5] / k1[4]); // -k23 / k22

	double *k2;
	k2 = K2;
	double inv2_k11 = (1.0 / k2[0]);
	double inv2_k12 = (-k2[1] / (k2[0] * k2[4]));
	double inv2_k13 = ((-k2[2] * k2[4] + k2[1] * k2[5]) / (k2[0] * k2[4]));
	double inv2_k22 = (1.0 / k2[4]);
	double inv2_k23 = (-k2[5] / k2[4]);

	for (int i = 0; i < n_matches; i++) {
		(*nrmlz_frames0x++) = inv1_k11 * frames0x[i] + inv1_k12 * frames0y[i] + inv1_k13;
		(*nrmlz_frames0y++) = inv1_k22 * frames0y[i] + inv1_k23;
		(*nrmlz_frames1x++) = inv2_k11 * frames1x[i] + inv2_k12 * frames1y[i] + inv2_k13;
		(*nrmlz_frames1y++) = inv2_k22 * frames1y[i] + inv2_k23;
	}
}


 //If RANSAC is implemented on a fundamental matrix, which is conducted by epipolar geometry model, 
 //all the coordinates of the matches should be normalized in the interval [-sqrt(2), sqrt(2)]
 //Input: 
 //n: the number of correspondences
 //framesx, framesy: the x- and y-coordinates of correspondences
 //Output:
 //nrmlz_framesx, nrmlz_framesy: the normalized coordinates
 //T1: normal matrix
__host__ __device__ void normal_samples_fund(int n, float *framesx, float *framesy, float *nrmlz_framesx, float *nrmlz_framesy, double *T1)
{
	int i;
	double muI[2] = { 0,0 };
	double rho1 = 0.0;
	for (i = 0; i < n; i++)
	{
		muI[0] += framesx[i];
		muI[1] += framesy[i];
	}
	muI[0] /= double(n);
	muI[1] /= double(n);
	for (i = 0; i < n; i++)
	{
		nrmlz_framesx[i] = framesx[i] - muI[0];// center x-coordinations of the points
		nrmlz_framesy[i] = framesy[i] - muI[1];// center y-coordinations of the points
		rho1 += sqrt(nrmlz_framesx[i] * nrmlz_framesx[i] + nrmlz_framesy[i] * nrmlz_framesy[i]);
	}
	rho1 /= n;
	rho1 = 1.414213562373095 / rho1;//scale factor
	*T1 = rho1;//Row dominant
	*(T1+1) = 0.0;
	*(T1+2) = -rho1*muI[0];// muIx
	*(T1 + 3) = 0.0;
	*(T1 + 4) = rho1;
	*(T1 + 5) = -rho1*muI[1];// muIy
	*(T1 + 6) = 0.0;
	*(T1 + 7) = 0.0;
	*(T1 + 8) = 1.0;
	for (i = 0; i < n; i++)
	{
		nrmlz_framesx[i] = *T1 * framesx[i] + *(T1 + 2);//normalization
		nrmlz_framesy[i] = *(T1 + 4) * framesy[i] + *(T1 + 5);
	}
	//	cout << "muI:" << muI[0]<< muI[1] << endl;
	//	return muI[0];
}

__host__ __device__ static double PYTHAG(double a, double b)
{
	double at = fabs(a), bt = fabs(b), ct, result;
	if (at > bt) { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}

//SVD for thin matrix A=USV'
//A is input and output of U in size of m rows and n columns
//diag is the output of S matrix, stored in memory using a vector
//v is the output of V in size of n*n
//rv1 with the same length of sigular value vector, and almost equals 0
__host__ __device__ int svd(double *A, double *diag, double *v, double *rv1, int m, int n)
//For an arbitrary dimensional matrix m by n, you need to replace ndof and isnp in the following test with m and n, respectively.
//And add "int m, int n" in the function declair which is shown as follows:
//__device__ int svd(double A[ndof][ndof], int m, int n, double *diag, double v[isnp][isnp])
{
	int i, j, k, l;
	double f, h, s;
	double anorm = 0.0, g = 0.0, scale = 0.0;

	for (i = 0; i < n; i++)
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{
			for (k = i; k < m; k++)//array[i][j]=*(array +i*n +j)
								   //scale += fabs(A[k][i]);//A[k][i]=*(A+k*n+i)
				scale += fabs(*(A + k*n + i));
			if (scale)
			{
				for (k = i; k < m; k++)
				{
					*(A + k*n + i) = (*(A + k*n + i) / scale);
					s += (*(A + k*n + i) * *(A + k*n + i));
				}
				f = *(A + i*n + i);
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				*(A + i*n + i) = f - g;
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += *(A + k*n + i) * *(A + k*n + j);
						f = s / h;
						for (k = i; k < m; k++)
							*(A + k*n + j) += f * *(A + k*n + i);
					}
				}
				for (k = i; k < m; k++)
					*(A + k*n + i) = *(A + k*n + i) * scale;
			}
		}
		diag[i] = scale * g;

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1)
		{
			for (k = l; k < n; k++)
				scale += fabs(*(A + i*n + k));
			if (scale)
			{
				for (k = l; k < n; k++)
				{
					*(A + i*n + k) = *(A + i*n + k) / scale;
					s += *(A + i*n + k) * *(A + i*n + k);
				}
				f = *(A + i*n + l);
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				*(A + i*n + l) = f - g;
				for (k = l; k < n; k++)
					rv1[k] = *(A + i*n + k) / h;
				if (i != m - 1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += *(A + j*n + k) * *(A + i*n + k);
						for (k = l; k < n; k++)
							*(A + j*n + k) += s * rv1[k];
					}
				}
				for (k = l; k < n; k++)
					*(A + i*n + k) = *(A + i*n + k) * scale;
			}
		}
		anorm = MAX(anorm, (fabs(diag[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < n - 1)
		{
			if (g)
			{
				for (j = l; j < n; j++)
					*(v + j*n + i) = (*(A + i*n + j) / *(A + i*n + l)) / g;
				/* double division to avoid underflow */
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < n; k++)
						s += (*(A + i*n + k) * *(v + k*n + j));
					for (k = l; k < n; k++)
						*(v + k*n + j) += s * *(v + k*n + i);
				}
			}
			for (j = l; j < n; j++)
				*(v + i*n + j) = *(v + j*n + i) = 0.0;
		}
		*(v + i*n + i) = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = diag[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				*(A + i*n + j) = 0.0;
		if (g)
		{
			g = 1.0 / g;
			if (i != n - 1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += *(A + k*n + i) * *(A + k*n + j);
					f = s / *(A + i*n + i) * g;
					for (k = i; k < m; k++)
						*(A + k*n + j) += (f * *(A + k*n + i));
				}
			}
			for (j = i; j < m; j++)
				*(A + j*n + i) = *(A + j*n + i) * g;
		}
		else
		{
			for (j = i; j < m; j++)
				*(A + j*n + i) = 0.0;
		}
		++(*(A + i*n + i));
	}
	int flag, its, jj, nm;
	double c, x, y, z;
	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--)
	{                             /* loop over singular values */
		for (its = 0; its < 30; its++)
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(diag[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = diag[i];
						h = PYTHAG(f, g);
						diag[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++)
						{
							y = *(A + j*n + nm);
							z = *(A + j*n + i);
							*(A + j*n + nm) = y * c + z * s;
							*(A + j*n + i) = z * c - y * s;
						}
					}
				}
			}
			z = diag[k];
			if (l == k)
			{                  /* convergence */
				if (z < 0.0)
				{              /* make singular value nonnegative */
					diag[k] = -z;
					for (j = 0; j < n; j++)
						*(v + j*n + k) = (-*(v + j*n + k));
				}
				break;
			}
			if (its >= 30) {
				free((void*)rv1);
				printf("No convergence after 30,000! iterations \n");
				return(0);
			}

			/* shift from bottom 2 x 2 minor */
			x = diag[l];
			nm = k - 1;
			y = diag[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++)
			{
				i = j + 1;
				g = rv1[i];
				y = diag[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++)
				{
					x = *(v + jj*n + j);
					z = *(v + jj*n + i);
					*(v + jj*n + j) = x * c + z * s;
					*(v + jj*n + i) = z * c - x * s;
				}
				z = PYTHAG(f, h);
				diag[j] = z;
				if (z)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
					y = *(A + jj*n + j);
					z = *(A + jj*n + i);
					*(A + jj*n + j) = y * c + z * s;
					*(A + jj*n + i) = z * c - y * s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			diag[k] = x;
		}
	}
	return(1);
}


__host__ __device__ void makePolynomial(double* A, double* B, double* p)
{
	// calculates polynomial p in x, so that det(xA + (1-x)B) = 0
	// where A,B are [3][3] and p is [4] arrays
	// ** CHANGES B to A-B ***
	// so finally det(A + (x-1) B) = 0 

	*p = -((*(B + 2))*(*(B + 4))*(*(B + 6))) + (*(B + 1))*(*(B + 5))*(*(B + 6)) + (*(B + 2))*(*(B + 3))*(*(B + 7)) -
		(*B)*(*(B + 5))*(*(B + 7)) - (*(B + 1))*(*(B + 3))*(*(B + 8)) + (*B)*(*(B + 4))*(*(B + 8));
	*(p + 1) = -((*(A + 8))*(*(B + 1))*(*(B + 3))) + (*(A + 7))*(*(B + 2))*(*(B + 3)) + (*(A + 8))*(*B)*(*(B + 4)) -
		(*(A + 6))*(*(B + 2))*(*(B + 4)) - (*(A + 7))*(*B)*(*(B + 5)) + (*(A + 6))*(*(B + 1))*(*(B + 5)) +
		(*(A + 5))*(*(B + 1))*(*(B + 6)) - (*(A + 4))*(*(B + 2))*(*(B + 6)) - (*(A + 2))*(*(B + 4))*(*(B + 6)) +
		3 * (*(B + 2))*(*(B + 4))*(*(B + 6)) + (*(A + 1))*(*(B + 5))*(*(B + 6)) - 3 * (*(B + 1))*(*(B + 5))*(*(B + 6)) -
		(*(A + 5))*(*B)*(*(B + 7)) + (*(A + 3))*(*(B + 2))*(*(B + 7)) + (*(A + 2))*(*(B + 3))*(*(B + 7)) -
		3 * (*(B + 2))*(*(B + 3))*(*(B + 7)) - (*A)*(*(B + 5))*(*(B + 7)) + 3 * (*B)*(*(B + 5))*(*(B + 7)) +
		((*(A + 4))*(*B) - (*(A + 3))*(*(B + 1)) - (*(A + 1))*(*(B + 3)) + 3 * (*(B + 1))*(*(B + 3)) + (*A)*(*(B + 4)) -
			3 * (*B)*(*(B + 4)))*(*(B + 8));

	*(p + 2) = -((*(A + 3))*(*(A + 8))*(*(B + 1))) + (*(A + 3))*(*(A + 7))*(*(B + 2)) +
		(*(A + 2))*(*(A + 7))*(*(B + 3)) - (*(A + 1))*(*(A + 8))*(*(B + 3)) + 2 * (*(A + 8))*(*(B + 1))*(*(B + 3)) -
		2 * (*(A + 7))*(*(B + 2))*(*(B + 3)) - (*(A + 2))*(*(A + 6))*(*(B + 4)) + (*A)*(*(A + 8))*(*(B + 4)) -
		2 * (*(A + 8))*(*B)*(*(B + 4)) + 2 * (*(A + 6))*(*(B + 2))*(*(B + 4)) + (*(A + 1))*(*(A + 6))*(*(B + 5)) -
		(*A)*(*(A + 7))*(*(B + 5)) + 2 * (*(A + 7))*(*B)*(*(B + 5)) - 2 * (*(A + 6))*(*(B + 1))*(*(B + 5)) +
		2 * (*(A + 2))*(*(B + 4))*(*(B + 6)) - 3 * (*(B + 2))*(*(B + 4))*(*(B + 6)) - 2 * (*(A + 1))*(*(B + 5))*(*(B + 6)) +
		3 * (*(B + 1))*(*(B + 5))*(*(B + 6)) + (*(A + 2))*(*(A + 3))*(*(B + 7)) - 2 * (*(A + 3))*(*(B + 2))*(*(B + 7)) -
		2 * (*(A + 2))*(*(B + 3))*(*(B + 7)) + 3 * (*(B + 2))*(*(B + 3))*(*(B + 7)) + 2 * (*A)*(*(B + 5))*(*(B + 7)) -
		3 * (*B)*(*(B + 5))*(*(B + 7)) + (*(A + 5))*
		(-((*(A + 7))*(*B)) + (*(A + 6))*(*(B + 1)) + (*(A + 1))*(*(B + 6)) - 2 * (*(B + 1))*(*(B + 6)) -
		(*A)*(*(B + 7)) + 2 * (*B)*(*(B + 7))) +
			(-((*(A + 1))*(*(A + 3))) + 2 * (*(A + 3))*(*(B + 1)) + 2 * (*(A + 1))*(*(B + 3)) - 3 * (*(B + 1))*(*(B + 3)) -
				2 * (*A)*(*(B + 4)) + 3 * (*B)*(*(B + 4)))*(*(B + 8)) +
				(*(A + 4))*((*(A + 8))*(*B) - (*(A + 6))*(*(B + 2)) - (*(A + 2))*(*(B + 6)) + 2 * (*(B + 2))*(*(B + 6)) +
		(*A)*(*(B + 8)) - 2 * (*B)*(*(B + 8)));

	for (unsigned int i = 0; i < 9; ++i)
	{
		B[i] = A[i] - B[i];
	}

	*(p + 3) = -((*(B + 2))*(*(B + 4))*(*(B + 6))) + (*(B + 1))*(*(B + 5))*(*(B + 6)) + (*(B + 2))*(*(B + 3))*(*(B + 7)) -
		(*B)*(*(B + 5))*(*(B + 7)) - (*(B + 1))*(*(B + 3))*(*(B + 8)) + (*B)*(*(B + 4))*(*(B + 8));
} // end makePolynomial

__host__ __device__ double real(double2 xx)
{
	return(xx.x);
}
__host__ __device__ double imag(double2 xx)
{
	return(xx.y);
}

__host__ __device__ double2 cucpow(double2 input, double param)
{
	double thita, r;
	double2 output;
	thita = atan(input.y / (input.x + 0.000000000000000000000000000001));
	r = sqrt(input.x*input.x + input.y*input.y);
	r = pow(r, param);
	output.x = r*cos(param*thita);
	output.y = r*sin(param*thita);
	return(output);
}


__host__ __device__ int cubic_roots(double poly[4], double p3roots[N3])
{
	double2 tem, temp2, temp3, factor_w = { -0.5, 0.866025403784439 }, factor_w2{ -0.5, -0.866025403784439 }, x1, x2, x3;
	double temp;
	int n_sol;
	temp = -*(poly + 1) / (*poly) / 3.0;
	//transform general form to Cardano form: x^3+p*x+q=0, where p is stored in p3roots[0], q stored in p3roots[1]
	p3roots[0] = *(poly + 2) / (*poly) - *(poly + 1) / (*poly) * *(poly + 1) / (*poly) / 3.0;//p
	p3roots[1] = *(poly + 3) / (*poly) - *(poly + 1)**(poly + 2) / (*poly) / (*poly) / 3.0 + 2.0**(poly + 1) / (*poly)**(poly + 1) / (*poly)**(poly + 1) / (*poly) / 27.0;//q
	p3roots[2] = -p3roots[1] * 0.5;//-q/2
	p3roots[0] = p3roots[1] * p3roots[1] * 0.25 + p3roots[0] * p3roots[0] * p3roots[0] / 27.0;//(q/2)^2+(p/3)^3
	if (p3roots[0] > 0)//(q/2)^2+(p/3)^3 is a real number
	{
		tem = { sqrt(p3roots[0]),0 };//sqrt( (q/2)^2 + (p/3)^3 )
		temp2 = { p3roots[2] + real(tem),0 };//the first term of x1 before cubic root:-q/2 + sqrt( (q/2)^2 + (p/3)^3 )
		temp3 = { p3roots[2] - real(tem),0 };//the second term of x1 before cubic root: -q/2 - sqrt( (q/2)^2 + (p/3)^3 )
	}
	else//complex number
	{
		tem = { 0, sqrt(-p3roots[0]) };//temp2=*rv1 + i*temp2.y
		temp2 = { p3roots[2], imag(tem) };
		temp3 = { p3roots[2], -imag(tem) };//temp2=*rv1 - i*temp2.y
	}
	//pow cannot work on negative number
	temp2 = cucpow(temp2, 0.33333333333333333333333333333333333333);//cubic root of temp2 cubic_root( -q/2+sqrt(q^2/4+p^3/27) )
	temp3 = cucpow(temp3, 0.33333333333333333333333333333333333333);//cubic root of temp2 cubic_root( -q/2+sqrt(q^2/4+p^3/27) )

	x1 = { temp + real(temp2) + real(temp3),imag(temp2) + imag(temp3) };//x1
	x2 = { temp + real(factor_w)*real(temp2) - imag(factor_w)*imag(temp2),  real(factor_w)*imag(temp2) + imag(factor_w)*real(temp2) };
	//x2
	x2 = { real(x2) + real(factor_w2)*real(temp3) - imag(factor_w2)*imag(temp3), imag(x2) + real(factor_w2)*imag(temp3) + imag(factor_w2)*real(temp3) };
	x3 = { temp + real(factor_w2)*real(temp2) - imag(factor_w2)*imag(temp2),  real(factor_w2)*imag(temp2) + imag(factor_w2)*real(temp2) };
	//x3 
	x3 = { real(x3) + real(factor_w)*real(temp3) - imag(factor_w)*imag(temp3), imag(x3) + real(factor_w)*imag(temp3) + imag(factor_w)*real(temp3) };

	n_sol = 0;
	if (imag(x1) < 0.00000001&imag(x1) > -0.00000001)
	{
		n_sol++; p3roots[0] = real(x1);
	}//x1,x2,x3 stored in A from A+1 to A+3 depending on the head count at *A
	if (imag(x2) < 0.00000001&imag(x2) > -0.00000001)
	{
		n_sol++; p3roots[1] = real(x2);
	}
	if (imag(x3) < 0.00000001&imag(x3) > -0.00000001)
	{
		n_sol++; p3roots[2] = real(x3);
	}
	return(n_sol);
}

//run for host
void initial_A_fund_host(double *A, double2 *X1, double2 *X2, int m)
{
	int i;
	for (i = 0; i < m; i++)
	{
		*(A + i*ndof) = X1[i].x * X2[i].x; //A[i][0]
		*(A + i*ndof + 1) = X1[i].y * X2[i].x;//A[i][1]
		*(A + i*ndof + 2) = X2[i].x;
		*(A + i*ndof + 3) = X1[i].x * X2[i].y;
		*(A + i*ndof + 4) = X1[i].y * X2[i].y;
		*(A + i*ndof + 5) = X2[i].y;
		*(A + i*ndof + 6) = X1[i].x;
		*(A + i*ndof + 7) = X1[i].y;
		*(A + i*ndof + 8) = 1.0;
	}

}

__device__ void get_nrand(int *mask, int mask_num, int n_matches, int s_d_check)
{
	int s_d, i, j;
	curandState localState;
	curand_init(clock64(), threadIdx.x, 0, &localState);
	//	mask[0] = 0; mask[1] = 1; mask[2] = 2; mask[3] = 3;
	for (i = 0; i < mask_num; i++)
	{
		while (1)
		{
			s_d = 0;
			for (i = 0; i < mask_num; i++)
			{
				*(mask + i) = curand_uniform(&localState) * (n_matches - 1);//x=rand(), results in 0<=x<RAND_MAX
				if (*(mask + i) == n_matches)
					printf("============================================data out!!!!!!!!!!!!!, need to revise 'n_matches' to 'n_matches-1'\n");
			}
			for (i = 0; i < mask_num - 1; i++)
				for (j = i + 1; j < mask_num; j++)
					if (*(mask + i) != *(mask + j))
						s_d++;
			if (s_d == s_d_check)
				break;
		}
	}
}

//Let the minimum singular value be zero
__host__ __device__ void min_diag_0(double *diag3)
{
	double temp;
	int i, mask = 0;
	temp = *diag3;
	for (i = 0; i < N3; i++)
		if (*(diag3 + i) < temp)
		{
			mask = i;
			temp = *(diag3 + i);
		}
	*(diag3 + mask) = 0.0;
}

//F_star=U*diag3*V', U, diag3, V are all in size of 3*3
__host__ __device__ void usvt(double *A1, double *diag3, double *v3, double F_star[N3][N3])
{
	F_star[0][0] = *A1**diag3**v3 + *(A1 + 1)**(diag3 + 1)**(v3 + 1) + *(A1 + 2)**(diag3 + 2)**(v3 + 2);
	F_star[0][1] = *A1**diag3**(v3 + 3) + *(A1 + 1)**(diag3 + 1)**(v3 + 4) + *(A1 + 2)**(diag3 + 2)**(v3 + 5);
	F_star[0][2] = *A1**diag3**(v3 + 6) + *(A1 + 1)**(diag3 + 1)**(v3 + 7) + *(A1 + 2)**(diag3 + 2)**(v3 + 8);
	F_star[1][0] = *(A1 + 3)**diag3**v3 + *(A1 + 4)**(diag3 + 1)**(v3 + 1) + *(A1 + 5)**(diag3 + 2)**(v3 + 2);
	F_star[1][1] = *(A1 + 3)**diag3**(v3 + 3) + *(A1 + 4)**(diag3 + 1)**(v3 + 4) + *(A1 + 5)**(diag3 + 2)**(v3 + 5);
	F_star[1][2] = *(A1 + 3)**diag3**(v3 + 6) + *(A1 + 4)**(diag3 + 1)**(v3 + 7) + *(A1 + 5)**(diag3 + 2)**(v3 + 8);
	F_star[2][0] = *(A1 + 6)**diag3**v3 + *(A1 + 7)**(diag3 + 1)**(v3 + 1) + *(A1 + 8)**(diag3 + 2)**(v3 + 2);
	F_star[2][1] = *(A1 + 6)**diag3**(v3 + 3) + *(A1 + 7)**(diag3 + 1)**(v3 + 4) + *(A1 + 8)**(diag3 + 2)**(v3 + 5);
	F_star[2][2] = *(A1 + 6)**diag3**(v3 + 6) + *(A1 + 7)**(diag3 + 1)**(v3 + 7) + *(A1 + 8)**(diag3 + 2)**(v3 + 8);
}

//count inliers using sampson distances
__host__ __device__ int consensus_check(int n_matches, double F_star[N3][N3], float *frames1x, float *frames1y, float *frames2x, float *frames2y, double inlier_thresh)
{
	int k, inlier_count = 0;
	double Fx1_temp0, Fx1_temp1, Fx1_temp2, x2tFx1_temp, Ftx2_temp0, Ftx2_temp1;
	for (k = 0; k < n_matches; k++)
	{
		Fx1_temp0 = F_star[0][0] * *(frames1x + k) + F_star[0][1] * *(frames1y + k) + F_star[0][2];
		Fx1_temp1 = F_star[1][0] * *(frames1x + k) + F_star[1][1] * *(frames1y + k) + F_star[1][2];
		Fx1_temp2 = F_star[2][0] * *(frames1x + k) + F_star[2][1] * *(frames1y + k) + F_star[2][2];
		x2tFx1_temp = *(frames2x + k)*Fx1_temp0 + *(frames2y + k)*Fx1_temp1 + Fx1_temp2;
		Ftx2_temp0 = F_star[0][0] * *(frames2x + k) + F_star[1][0] * *(frames2y + k) + F_star[2][0];
		Ftx2_temp1 = F_star[0][1] * *(frames2x + k) + F_star[1][1] * *(frames2y + k) + F_star[2][1];
		x2tFx1_temp = x2tFx1_temp*x2tFx1_temp / (Fx1_temp0*Fx1_temp0 + Fx1_temp1*Fx1_temp1 + Ftx2_temp0*Ftx2_temp0 + Ftx2_temp1*Ftx2_temp1);
		if (x2tFx1_temp < inlier_thresh)
		{
			inlier_count++;
		}
	}
	return(inlier_count);
}


//Denormalize to normalization plan
__host__ __device__ void denormalization(double F_star[N3][N3], double T0[ndof], double T1[ndof], double F[N3][N3])
{
	int i, j, k;
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			F[i][j] = 0;
	for (i = 0; i < N3; i++)//T1'*F_star
		for (j = 0; j < N3; j++)
			for (k = 0; k < N3; k++)
				F[i][j] += T1[k*N3+i] * F_star[k][j];
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			F_star[i][j] = F[i][j];
	for (i = 0; i < N3; i++)//T1'*F_star*T0
		for (j = 0; j < N3; j++)
		{
			F[i][j] = 0;
			for (k = 0; k < N3; k++)
				F[i][j] += F_star[i][k] * T0[k*N3 + j];
		}
}

__host__ __device__ void eliminateUpperTriangular(double *a, int m, int n)
{
	double temp;
	for (int r = 0; r < m; r++) {
		double pivot = a[r*n + r];
		int row_with_pivot = r;

		// find the maximum pivot value among r-th column
		for (int k = r + 1; k < m; k++)
			if (fabs(pivot) < fabs(a[k*n + r])) {
				pivot = a[k*n + r];
				row_with_pivot = k;
			}

		// if pivot value is 0 continue
		if (fabs(pivot) < DBL_EPSILON)
			continue;

		// swap row with maximum pivot value with current row
		for (int c = r; c < n; c++)
		{
			temp = a[r*n + c];
			a[r*n + c] = a[row_with_pivot*n + c];
			a[row_with_pivot*n + c] = temp;
		}
		//std::swap(a[row_with_pivot*n + c], a[r*n + c]);

		// eliminate other rows
		for (int j = r + 1; j < m; j++) {
			const int row_idx1 = j*n, row_idx2 = r*n;
			const auto fac = a[row_idx1 + r] / pivot;
			a[row_idx1 + r] = 0.0; // zero eliminated element
			for (int c = r + 1; c < n; c++)
				a[row_idx1 + c] -= fac * a[row_idx2 + c];
		}
	}
	//return true;
}

//Initialize the essential equations and then solve the equations using LU chase method
//Input: pts(corresondence points: (dev_nrmlz_frames0x,dev_nrmlz_frames0y)<------>(dev_nrmlz_frames1x, dev_nrmlz_frames1y)
//sample: cm_ind[hit_ind} (5 integer array indicate the row index of pts)
//Output: ee (4 null tensors in shape of 4*9 )
__host__ __device__ void null_space(float *dev_nrmlz_frames0x, float *dev_nrmlz_frames0y, float *dev_nrmlz_frames1x, float *dev_nrmlz_frames1y, int *cm_ind, int *hit_ind, double ee[36])
{
	double coefficients[45], *coefficients_, x1, y1, x2, y2;
	int i, j, k;
	coefficients_ = coefficients;
	for (i = 0; i < 5; i++)
	{
		///x1 = dev_nrmlz_frames0x[hit_ind[i]];//extract coordinates based on index
		///y1 = dev_nrmlz_frames0y[hit_ind[i]];//hid_ind points to cm_ind and cm_ind points to frame
		///x2 = dev_nrmlz_frames1x[hit_ind[i]];//extract coordinates based on index
		///y2 = dev_nrmlz_frames1y[hit_ind[i]];
		x1 = dev_nrmlz_frames0x[cm_ind[hit_ind[i]]];//extract coordinates based on index
		y1 = dev_nrmlz_frames0y[cm_ind[hit_ind[i]]];//hid_ind points to cm_ind and cm_ind points to frame
		x2 = dev_nrmlz_frames1x[cm_ind[hit_ind[i]]];//extract coordinates based on index
		y2 = dev_nrmlz_frames1y[cm_ind[hit_ind[i]]];

		(*coefficients_++) = x2 * x1;
		(*coefficients_++) = x2 * y1;
		(*coefficients_++) = x2;
		(*coefficients_++) = y2 * x1;
		(*coefficients_++) = y2 * y1;
		(*coefficients_++) = y2;
		(*coefficients_++) = x1;
		(*coefficients_++) = y1;
		(*coefficients_++) = 1.0;
	}
	///printf("essential equations:\n");
	///for (i = 0; i < 5; i++)
	///{
	///for (j = 0; j < 9; j++)
	///printf("%f ", coefficients[i * 9 + j]);
	///printf("\n");
	///}
	int num_cols = 9, num_e_mat = 4;
	eliminateUpperTriangular(coefficients, 5, num_cols);
	// eliminate linear equations

	///printf("essential equations after elimination:\n");
	///for (i = 0; i < 5; i++)
	///{
	///for (j = 0; j < 9; j++)
	///printf("%f ", coefficients[i * 9 + j]);
	///printf("\n");
	///}
	for (i = 0; i < num_e_mat; i++)
		for (j = 5; j < num_cols; j++)
			ee[num_cols * i + j] = (i + 5 == j) ? 1.0 : 0.0;
	// use back-substitution
	for (int e = 0; e < num_e_mat; e++) {
		const int curr_e = num_cols * e;
		// start from the last row
		for (i = 4; i >= 0; i--) {
			const int row_i = i * num_cols;
			double acc = 0;
			for (j = i + 1; j < num_cols; j++)
				acc -= coefficients[row_i + j] * ee[curr_e + j];
			ee[curr_e + i] = acc / coefficients[row_i + i];
			// due to numerical errors return 0 solutions
		}
	}
}

/*
* Multiply two polynomials of degree one with unknowns x y z
* @p = (p1 x + p2 y + p3 z + p4) [p1 p2 p3 p4]
* @q = (q1 x + q2 y + q3 z + q4) [q1 q2 q3 a4]
* @result is a new polynomial in x^2 xy y^2 xz yz z^2 x y z 1 of size 10
*/
__host__ __device__ void multPolysDegOne(double * p, double * q, double *poly10)
{
	*poly10 = p[0] * q[0];
	*(poly10 + 1) = p[0] * q[1] + p[1] * q[0];
	*(poly10 + 2) = p[1] * q[1];
	*(poly10 + 3) = p[0] * q[2] + p[2] * q[0];
	*(poly10 + 4) = p[1] * q[2] + p[2] * q[1];
	*(poly10 + 5) = p[2] * q[2];
	*(poly10 + 6) = p[0] * q[3] + p[3] * q[0];
	*(poly10 + 7) = p[1] * q[3] + p[3] * q[1];
	*(poly10 + 8) = p[2] * q[3] + p[3] * q[2];
	*(poly10 + 9) = p[3] * q[3];
}

/*
* Multiply two polynomials with unknowns x y z
* @p is of size 10 and @q is of size 4
* @p = (p1 x^2 + p2 xy + p3 y^2 + p4 xz + p5 yz + p6 z^2 + p7 x + p8 y + p9 z + p10)
* @q = (q1 x + q2 y + q3 z + a4) [q1 q2 q3 q4]
* @result is a new polynomial of size 20
* x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
*/
__host__ __device__ void multPolysDegOneAndTwo(double *p, double *q, double *poly20)
{
	poly20[0] = p[0] * q[0];
	poly20[1] = p[0] * q[1] + p[1] * q[0];
	poly20[2] = p[1] * q[1] + p[2] * q[0];
	poly20[3] = p[2] * q[1];
	poly20[4] = p[0] * q[2] + p[3] * q[0];
	poly20[5] = p[1] * q[2] + p[3] * q[1] + p[4] * q[0];
	poly20[6] = p[2] * q[2] + p[4] * q[1];
	poly20[7] = p[3] * q[2] + p[5] * q[0];
	poly20[8] = p[4] * q[2] + p[5] * q[1];
	poly20[9] = p[5] * q[2];
	poly20[10] = p[0] * q[3] + p[6] * q[0];
	poly20[11] = p[1] * q[3] + p[6] * q[1] + p[7] * q[0];
	poly20[12] = p[2] * q[3] + p[7] * q[1];
	poly20[13] = p[3] * q[3] + p[6] * q[2] + p[8] * q[0];
	poly20[14] = p[4] * q[3] + p[7] * q[2] + p[8] * q[1];
	poly20[15] = p[5] * q[3] + p[8] * q[2];
	poly20[16] = p[6] * q[3] + p[9] * q[0];
	poly20[17] = p[7] * q[3] + p[9] * q[1];
	poly20[18] = p[8] * q[3] + p[9] * q[2];
	poly20[19] = p[9] * q[3];
}

__host__ __device__ void build10_20eqs(double  null_space_mat[N3][N3][4], double constraint_mat[10][20])
{
	double eet[3][3][10], poly10_0[10], poly10_1[10], poly10_2[10], trace[10];
	int i, j, k;
	for (k = 0; k < 10; k++)
		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++)
				// compute EE Transpose
				// Shorthand for multiplying the Essential matrix with its transpose.
			{
				multPolysDegOne(null_space_mat[i][0], null_space_mat[j][0], poly10_0);
				multPolysDegOne(null_space_mat[i][1], null_space_mat[j][1], poly10_1);
				multPolysDegOne(null_space_mat[i][2], null_space_mat[j][2], poly10_2);
				eet[i][j][k] = 2.0 * (poly10_0[k] + poly10_1[k] + poly10_2[k]);
			}
	for (k = 0; k < 10; k++)
		trace[k] = eet[0][0][k] + eet[1][1][k] + eet[2][2][k];

	double poly20_0[20], poly20_1[20], poly20_2[20], poly20_3[20];

	// Trace constraint
	for (k = 0; k<20; k++)
		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++)
			{
				multPolysDegOneAndTwo(eet[i][0], null_space_mat[0][j], poly20_0);
				multPolysDegOneAndTwo(eet[i][1], null_space_mat[1][j], poly20_1);
				multPolysDegOneAndTwo(eet[i][2], null_space_mat[2][j], poly20_2);
				multPolysDegOneAndTwo(trace, null_space_mat[i][j], poly20_3);
				constraint_mat[3 * i + j][k] = poly20_0[k] + poly20_1[k] + poly20_2[k] - 0.5*poly20_3[k];
			}

	// Rank = zero determinant constraint
	multPolysDegOne(null_space_mat[0][1], null_space_mat[1][2], poly10_0);
	multPolysDegOne(null_space_mat[0][2], null_space_mat[1][1], poly10_1);
	for (k = 0; k < 10; k++)
		poly10_0[k] = poly10_0[k] - poly10_1[k];
	multPolysDegOneAndTwo(poly10_0, null_space_mat[2][0], poly20_0);

	multPolysDegOne(null_space_mat[0][2], null_space_mat[1][0], poly10_0);
	multPolysDegOne(null_space_mat[0][0], null_space_mat[1][2], poly10_1);
	for (k = 0; k < 10; k++)
		poly10_0[k] = poly10_0[k] - poly10_1[k];
	multPolysDegOneAndTwo(poly10_0, null_space_mat[2][1], poly20_1);
	multPolysDegOne(null_space_mat[0][0], null_space_mat[1][1], poly10_0);
	multPolysDegOne(null_space_mat[0][1], null_space_mat[1][0], poly10_1);
	for (k = 0; k < 10; k++)
		poly10_0[k] = poly10_0[k] - poly10_1[k];
	multPolysDegOneAndTwo(poly10_0, null_space_mat[2][2], poly20_2);
	for (k = 0; k < 20; k++)
		constraint_mat[9][k] = poly20_0[k] + poly20_1[k] + poly20_2[k];
}


//QR decomposition.
__host__ __device__ int QR_decomp(double A[N10][N10], double Q[N10][N10], double R[N10][N10], int pivot[N10])
{
	double norm_square[N10], temp, normsquare_vect, vect[N10], vect_temp[N10], H1[N10][N10], H2[N10][N10], H_temp[N10][N10];
	int i, j, k, l, ind, ind_temp, ord_i;
	for (i = 0; i < N10; i++)
	{
		pivot[i] = i;//pivot initialization
		norm_square[i] = 0.0;
		for (j = 0; j < N10; j++)
		{
			norm_square[i] += A[j][i] * A[j][i];
			R[j][i] = A[j][i];
		}
	}
	for (i = 0; i < N10; i++)
	{//processing min(rows, columns) columns for a rectangle matrix
		temp = norm_square[i];
		ind = i;//initialization
		for (j = i + 1; j < N10; j++)
			if (norm_square[j] > temp)
			{
				ind = j;//the column index with a largest norm
				temp = norm_square[j];
			}
		if (ind != i)
		{
			for (j = 0; j < N10; j++)
			{//exchange pivot dominated columns
				temp = A[j][i];
				A[j][i] = A[j][ind];
				A[j][ind] = temp;
				temp = R[j][i];
				R[j][i] = R[j][ind];
				R[j][ind] = temp;

			}
			//update pivot
			ind_temp = pivot[i];
			pivot[i] = pivot[ind];
			pivot[ind] = ind_temp;
			//update norm
			temp = norm_square[i];
			norm_square[i] = norm_square[ind];
			norm_square[ind] = temp;
		}
		//householder transformation matrix
		//H_temp is in size of (N10-i)*(N10-i)
		temp = 0.0;
		for (j = i; j < N10; j++)
		{
			vect[j] = A[j][i];//u is initialized with the i-th column of A: u=A(i)
			temp += vect[j] * vect[j];
		}
		temp = sqrt(temp);
		if (vect[i]>0)
			vect[i] = vect[i] + temp;
		else
			vect[i] = vect[i] - temp;
		normsquare_vect = 0.0;
		//v=u/||u|| is a column vector
		for (j = i; j < N10; j++)
			//norm vect
			normsquare_vect += vect[j] * vect[j];
		///printf("normsquare_vect: %f\n", normsquare_vect);
		if (normsquare_vect> 1e-10)//update R under the nonzeros condition
		{
			for (j = i; j < N10; j++)//v=u/||u|| is a column vector
				vect_temp[j] = vect[j] / sqrt(normsquare_vect);
			//Householder reflection: Hi=I-2*v*v^T
			//Hi=1 0 0  0  0 //H with 0, 1
			//   0 1 0  0  0 //H with 0, 1
			//   0 0 h1 h2 h3//H with I-2*v*v^T
			//   0 0 h4 h5 h6
			//   0 0 h7 h8 h9
			for (k = 0; k < N10; k++)
				for (j = 0; j < N10; j++)
					if (k < i || j<i)//H with 0, 1
						if (k == j)
							H2[k][j] = 1.0;
						else
							H2[k][j] = 0.0;
					else//H with I-2*v*v^T
						if (j == k)
							H2[k][j] = 1 - 2 * vect_temp[k] * vect_temp[k];
						else
							H2[k][j] = -2 * vect_temp[k] * vect_temp[j];
			if (i > 0)
			{//H=H1*H2
				for (k = 0; k < N10; k++)
					for (j = 0; j < N10; j++)
					{
						H_temp[k][j] = 0.0;
						for (l = 0; l < N10; l++)
							H_temp[k][j] += H1[k][l] * H2[l][j];
					}
				for (k = 0; k < N10; k++)//update H1
					for (j = 0; j < N10; j++)
						H1[k][j] = H_temp[k][j];
			}
			else//assign H1 with H2 at the beginning
				for (k = 0; k < N10; k++)
					for (j = 0; j < N10; j++)
						H1[k][j] = H2[k][j];

			//householder processing on R
			for (j = i; j<N10; j++)
			{
				for (k = i; k<N10; k++)
				{
					temp = 0.0;
					for (l = i; l < N10; l++)
						temp += vect[j] * vect[l] * A[l][k];
					R[j][k] -= 2.0 / (normsquare_vect)*temp;
					//printf("%f ", R[j][k]);
				}
				//printf("\n");
			}
			for (j = i + 1; j < N10; j++)
				norm_square[j] -= R[i][j] * R[i][j];
			for (j = i; j < N10; j++)
				for (k = i; k < N10; k++)
					A[j][k] = R[j][k];
		}
	}

	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
		{
			Q[i][j] = H_temp[i][j];
			if (i > j)
				R[i][j] = 0.0;
		}
	return 0;
}

__host__ __device__ void Grobner_base(double constraint_mat[N10][20], double Q[N10][N10])
{
	double A[N10][N10], B[N10][N10], b[N10], b_temp[N10], R[N10][N10], temp;
	int i, j, t, Pivot[N10];
	for (i = 0; i < N10; i++)
		for (j = 0; j < 20; j++)
			if (j < N10)
				A[i][j] = constraint_mat[i][j];
			else
				B[i][j - N10] = constraint_mat[i][j];
	QR_decomp(A, Q, R, Pivot);

	for (t = 0; t < N10; t++)
	{
		for (i = 0; i < N10; i++)
			b[i] = B[i][t];
		for (i = 0; i < N10; i++)
		{
			b_temp[i] = 0;
			for (j = 0; j < N10; j++)
				b_temp[i] += Q[j][i] * b[j];
		}
		//run back to iteratively solve upper-triangular equations
		for (i = N10 - 1; i >= 0; i--)
		{
			temp = b_temp[i];
			for (j = i + 1; j < N10; j++)
				temp -= R[i][j] * A[j][t];
			A[i][t] = temp / R[i][i];
		}
	}
	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
			Q[Pivot[i]][j] = A[i][j];

}

__host__ __device__ void matrix_multi(double *input_A, int mc, int na, double *input_B, int nc, double *output_C)
{
	for (int i = 0; i<mc; i++)
		for (int j = 0; j < nc; j++)
		{
			*(output_C + i*nc + j) = 0;
			for (int k = 0; k < na; k++)
				*(output_C + i*nc + j) += *(input_A + i*nc + k) * *(input_B + k*nc + j);
		}
}


__host__ __device__ double computeNormOfT(double *std_data, int n_rows, int n_cols)
{
	double temp = 0;
	for (int i = 0; i < n_rows; i++)
		for (int j = 0; j < n_cols; j++)
			temp += fabs(*(std_data + i*n_cols + j));
	return temp;
}

/**Look for single small sub-diagonal element and returns its index */
__host__ __device__ int findSmallSubdiagEntry(int iu, double *m_matT, int n_cols)
{
	//using std::abs;
	int res = iu;
	double s;
	while (res > 0)
	{
		s = fabs(*(m_matT + (res - 1)*n_cols + res - 1)) + fabs(*(m_matT + res*n_cols + res));
		if (fabs(*(m_matT + res*n_cols + res - 1)) <= epsilon * s)
			break;
		res--;
	}
	return res;
}

// specialization for reals
//如果r没有接收到值，则r相关的代码为无效代码
__host__ __device__ void makeGivens(double p, double q, double *rot)
{
	double m_c, m_s, t, u;// , *r = 0;
	if (q == 0)
	{
		m_c = p<0 ? -1.0 : 1.0;
		m_s = 0.0;
		//if (r)
		//*r = fabs(p);
	}
	else if (p == 0)
	{
		m_c = 0.0;
		m_s = q<0 ? 1.0 : -1.0;
		//if (r)
		//*r = fabs(q);
	}
	else if (fabs(p) > fabs(q))
	{
		t = q / p;
		u = sqrt(1.0 + t*t);
		if (p<0)
			u = -u;
		m_c = 1.0 / u;
		m_s = -t * m_c;
		//if (r)
		//*r = p * u;
	}
	else
	{
		t = p / q;
		u = sqrt(1.0 + t*t);
		if (q<0)
			u = -u;
		m_s = -1.0 / u;
		m_c = -t * m_s;
		//if (r)
		//*r = q * u;
	}
	*rot = m_c;
	*(rot + 1) = m_s;
	//return *r;
}

__host__ __device__ void splitOffTwoRows(double *m_matT, double *hessen_Q, int size, int iu, double *exshift)
{
	int j;
	// The eigenvalues of the 2x2 matrix [a b; c d] are 
	// trace +/- sqrt(discr/4) where discr = tr^2 - 4*det, tr = a + d, det = ad - bc
	double z, rot[2], temp[2], p = 0.5 * (*(m_matT + (iu - 1)*size + iu - 1) - *(m_matT + iu*size + iu));
	double q = p * p + *(m_matT + iu*size + iu - 1) * *(m_matT + (iu - 1)*size + iu);   // q = tr^2 / 4 - det = discr/4
	*(m_matT + iu*size + iu) += *exshift;
	*(m_matT + (iu - 1)*size + iu - 1) += *exshift;

	if (q >= 0) // Two real eigenvalues
	{
		z = sqrt(fabs(q));
		if (p >= 0)
			makeGivens(p + z, *(m_matT + iu*size + iu - 1), rot);
		else
			makeGivens(p - z, *(m_matT + iu*size + iu - 1), rot);
		//J=[rot(0)  rot(1)*
		//   -rot(1) rot(0)*]
		//adjoint(J)*[m_matT(iu-1,iu-1:size);
		//            m_matT(iu,iu-1:size)]
		for (j = iu - 1; j < size; j++)
		{
			temp[0] = *rot**(m_matT + (iu - 1)*size + j) - *(rot + 1)**(m_matT + iu*size + j);
			temp[1] = *(rot + 1)**(m_matT + (iu - 1)*size + j) + *rot**(m_matT + iu*size + j);
			*(m_matT + (iu - 1)*size + j) = temp[0];
			*(m_matT + iu*size + j) = temp[1];
		}
		///printf("m_matT:\n");
		///for (int i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(m_matT + i*size + j));
		///printf("\n");
		///}

		for (j = 0; j < iu + 1; j++)
		{
			temp[0] = *(m_matT + j*size + iu - 1)**rot - *(m_matT + j*size + iu)**(rot + 1);
			temp[1] = *(m_matT + j*size + iu - 1)**(rot + 1) + *(m_matT + j*size + iu)**rot;
			*(m_matT + j*size + iu - 1) = temp[0];
			*(m_matT + j*size + iu) = temp[1];
		}
		///printf("m_matT:\n");
		///for (int i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(m_matT + i*size + j));
		///printf("\n");
		///}


		*(m_matT + iu*size + iu - 1) = 0.0;

		for (j = 0; j<size; j++)
		{
			temp[0] = *(hessen_Q + j*size + iu - 1)**rot - *(hessen_Q + j*size + iu)**(rot + 1);
			temp[1] = *(hessen_Q + j*size + iu - 1)**(rot + 1) + *(hessen_Q + j*size + iu)**rot;
			*(hessen_Q + j*size + iu - 1) = temp[0];
			*(hessen_Q + j*size + iu) = temp[1];
		}
		///printf("hessen_Q:\n");
		///for (int i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(hessen_Q + i*size + j));
		///printf("\n");
		///}
	}

	if (iu > 1)
		*(m_matT + (iu - 1)*size + iu - 2) = 0.0;
}

//the shift of Francis
__host__ __device__ void computeShift(double *std_data, int n_cols, int iu, int iter, double *exshift, double *shiftInfo)
{
	int i;
	double s;
	*shiftInfo = *(std_data + iu*n_cols + iu);
	*(shiftInfo + 1) = *(std_data + (iu - 1)*n_cols + iu - 1);
	*(shiftInfo + 2) = *(std_data + iu*n_cols + iu - 1) * *(std_data + (iu - 1)*n_cols + iu);
	// Wilkinson's original ad hoc shift
	if (iter == 10)
	{
		*exshift += *shiftInfo;
		for (i = 0; i <= iu; ++i)
			*(std_data + i*n_cols + i) -= *shiftInfo;
		s = fabs(*(std_data + iu*n_cols + iu - 1)) + fabs(*(std_data + (iu - 1)*n_cols + iu - 2));
		*shiftInfo = 0.75 * s;
		*(shiftInfo + 1) = 0.75 * s;
		*(shiftInfo + 2) = -0.4375 * s * s;
	}

	// MATLAB's new ad hoc shift
	if (iter == 30)
	{
		s = *(shiftInfo + 1) - *shiftInfo / 2.0;
		s = s * s + *(shiftInfo + 2);
		if (s > 0)
		{
			s = sqrt(s);
			if (*(shiftInfo + 1) < *shiftInfo)
				s = -s;
			s = s + (*(shiftInfo + 1) - *shiftInfo) / 2.0;
			s = *shiftInfo - *(shiftInfo + 2) / s;
			*exshift += s;
			for (i = 0; i <= iu; ++i)
				*(std_data + i*n_cols + i) -= s;
			*shiftInfo = 0.964;
			*(shiftInfo + 1) = 0.964;
			*(shiftInfo + 2) = 0.964;
		}
	}

}

__host__ __device__ int initFrancisQRStep(double *m_matT, int n_cols, int il, int iu, double *shiftInfo, double *firstHouseholderVector)
{
	double Tmm, r, s, *v, lhs, rhs;// alias to save typing
	v = firstHouseholderVector;
	int im;
	for (im = iu - 2; im >= il; --im)
	{
		Tmm = *(m_matT + im*n_cols + im);
		r = *shiftInfo - Tmm;
		s = *(shiftInfo + 1) - Tmm;
		*v = (r * s - *(shiftInfo + 2)) / *(m_matT + (im + 1)*n_cols + im) + *(m_matT + im*n_cols + im + 1);
		*(v + 1) = *(m_matT + (im + 1)*n_cols + im + 1) - Tmm - r - s;
		*(v + 2) = *(m_matT + (im + 2)*n_cols + im + 1);
		if (im == il) {
			break;
		}
		lhs = *(m_matT + im*n_cols + im - 1) * (fabs(*(v + 1)) + fabs(*(v + 2)));
		rhs = *v * (fabs(*(m_matT + (im - 1)*n_cols + im - 1)) + fabs(Tmm) + fabs(*(m_matT + (im + 1)*n_cols + im + 1)));
		if (fabs(lhs) < epsilon * rhs)
			break;
	}
	return im;
}

__host__ __device__ void makeHouseholder(double *v, int lenv, double *essential, double *tau, double *beta)
{
	int i;
	double tailSqNorm = 0, c0, tol = 2.25e-100;
	for (i = 1; i < lenv; i++)
		tailSqNorm += *(v + i)**(v + i);
	c0 = *v;
	if (tailSqNorm <= tol)
	{
		*tau = 0.0;
		*beta = c0;
		for (i = 1; i<lenv; i++)
			*(essential + i - 1) = 0.0;
	}
	else
	{
		*beta = sqrt(c0*c0 + tailSqNorm);
		if (c0 >= 0)
			*beta = -*beta;
		for (i = 1; i<lenv; i++)
			*(essential + i - 1) = *(v + i) / (c0 - *beta);
		*tau = (*beta - c0) / *beta;
	}
}

//H3*block(m_matT), H3 is in size of size_H3*size_H3
//block(m_matT) is a block of m_matT begin at the row of row_begin and the column of col_begin, the block is in size of size_H3*ncols
//the parameter "size" is the size of m_matT
//temp is a temporary variable in size of size_H3*size to temporarily store the results of H3*block(m_matT)
__host__ __device__ void applyHouseholderOnTheLeft(double *H3, int size_H3, double *m_matT, int size, double *temp, int row_begin, int col_begin, int ncols)
{
	int i, j, l;
	for (i = 0; i<size_H3; i++)//applyHouseholderOnTheLeft
		for (j = 0; j < ncols; j++)
		{
			*(temp + i*size + j) = 0.0;
			for (l = 0; l<size_H3; l++)
				*(temp + i*size + j) += *(H3 + i * size_H3 + l)**(m_matT + (row_begin + l)*size + col_begin + j);
		}
	for (i = 0; i < size_H3; i++)
		for (j = 0; j < ncols; j++)
			*(m_matT + (row_begin + i)*size + col_begin + j) = *(temp + i*size + j);
}

__host__ __device__ void applyHouseholderOnTheRight(double *H3, int size_H3, double *m_matT, int size, double *temp, int row_begin, int col_begin, int nrows)
{//m_matT.block(0, iu-1, iu+1, 2).applyHouseholderOnTheRight(ess, tau, workspace);
 //applyHouseholderOnTheRight(H3, 2, m_matT, size, temp, 0, iu - 1, iu + 1);
	int i, j, l;
	for (i = 0; i < nrows; i++)
		for (j = 0; j < size_H3; j++)
		{
			*(temp + i * size_H3 + j) = 0.0;
			for (l = 0; l < size_H3; l++)
				*(temp + i * size_H3 + j) += *(m_matT + (row_begin + i)*size + col_begin + l)**(H3 + l * size_H3 + j);
		}
	for (i = 0; i < nrows; i++)
		for (j = 0; j < size_H3; j++)
			*(m_matT + (row_begin + i)*size + col_begin + j) = *(temp + i * size_H3 + j);
}

/**Perform a Francis QR step involving rows il:iu and columns im:iu. */
__host__ __device__ void performFrancisQRStep(double *m_matT, double *hessen_Q, int n_cols, int il, int im, int iu, double *firstHouseholderVector)
{
	if (im < il)
	{
		printf("There is an error happened in the function performFrancisQRStep where im>=il is not satisfied.\n");
		exit(0);
	}
	if (im > iu - 2)
	{
		printf("There is an error happened in the function performFrancisQRStep where im<=iu-2 is not satisfied.\n");
		exit(0);
	}

	int size = n_cols, k, i, j, l, m;
	double v[3], tau, beta, ess[2], *p, H3[9], temp[3 * N10];
	bool firstIteration;
	for (k = im; k <= iu - 2; ++k)
	{
		firstIteration = (k == im);

		if (firstIteration)
		{
			*v = *firstHouseholderVector;
			*(v + 1) = *(firstHouseholderVector + 1);
			*(v + 2) = *(firstHouseholderVector + 2);
		}
		else
		{
			*v = *(m_matT + k*n_cols + k - 1);
			*(v + 1) = *(m_matT + (k + 1)*n_cols + k - 1);
			*(v + 2) = *(m_matT + (k + 2)*n_cols + k - 1);
		}
		//
		makeHouseholder(v, 3, ess, &tau, &beta);
		///printf("v: %f %f %f\n", *v, *(v+1), *(v+2));
		///printf("essential: %f %f\n", *ess, *(ess+1));
		///printf("tau: %f, beta: %f\n", tau, beta);

		///printf("m_matT:\n");
		///for (i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(m_matT+i*n_cols + j));
		///printf("\n");
		///}

		if (beta != 0) // if v is not zero
		{
			if (firstIteration && k > il)
				*(m_matT + k*n_cols + k - 1) = -*(m_matT + k*n_cols + k - 1);
			else if (!firstIteration)
				*(m_matT + k*n_cols + k - 1) = beta;

			//Apply the elementary reflector H given by H = I - tau v v^* with v^T = [1 ess^T] from the left to a vector or matrix.
			//H is denoted as H3 as follows.
			*H3 = 1.0 - tau; *(H3 + 1) = -tau**ess; *(H3 + 2) = -tau**(ess + 1);
			*(H3 + 3) = -tau**ess; *(H3 + 4) = 1.0 - tau**ess**ess; *(H3 + 5) = -tau**ess**(ess + 1);
			*(H3 + 6) = -tau**(ess + 1); *(H3 + 7) = -tau**ess**(ess + 1); *(H3 + 8) = 1.0 - tau**(ess + 1)**(ess + 1);

			//m_matT.block(k, k, 3, size-k).applyHouseholderOnTheLeft(ess, tau, workspace);
			//draw the block matrix sub_m_matT at the position (k,k) of m_matT with the size of 3 rows by size-k columns
			//and return H*sub_m_matT
			applyHouseholderOnTheLeft(H3, 3, m_matT, size, temp, k, k, size - k);

			///printf("m_matT:\n");
			///for (i = 0; i < size; i++)
			///{
			///for (j = 0; j < size; j++)
			///printf("%f ", *(m_matT + i*n_cols + j));
			///printf("\n");
			///}

			//applyHouseholderOnTheRight
			if (iu < k + 3)
				m = iu + 1;
			else
				m = k + 4;
			applyHouseholderOnTheRight(H3, 3, m_matT, size, temp, 0, k, m);
			///printf("m_matT:\n");
			///for (i = 0; i < size; i++)
			///{
			///for (j = 0; j < size; j++)
			///printf("%f ", *(m_matT + i*n_cols + j));
			///printf("\n");
			///}

			//m_matU.block(0, k, size, 3).applyHouseholderOnTheRight(ess, tau, workspace);
			applyHouseholderOnTheRight(H3, 3, hessen_Q, size, temp, 0, k, size);

			///printf("m_matU:\n");
			///for (i = 0; i < size; i++)
			///{
			///for (j = 0; j < size; j++)
			///printf("%f ", *(hessen_Q + i*N10 + j));
			///printf("\n");
			///}
		}//if (beta != 0) // if v is not zero
	}//for (k = im; k <= iu - 2; ++k)

	double vv[2], ess1[1];
	*vv = *(m_matT + (iu - 1)*n_cols + iu - 2);
	*(vv + 1) = *(m_matT + iu * n_cols + iu - 2);
	makeHouseholder(vv, 2, ess1, &tau, &beta);

	*H3 = 1.0 - tau; *(H3 + 1) = -tau**ess1;
	*(H3 + 2) = -tau**ess1; *(H3 + 3) = 1.0 - tau**ess1**ess1;


	if (beta != 0)
	{
		*(m_matT + (iu - 1)*n_cols + iu - 2) = beta;
		applyHouseholderOnTheLeft(H3, 2, m_matT, size, temp, iu - 1, iu - 1, size - iu + 1);
		///printf("m_matT:\n");
		///for (i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(m_matT + i*n_cols + j));
		///printf("\n");
		///}

		applyHouseholderOnTheRight(H3, 2, m_matT, size, temp, 0, iu - 1, iu + 1);
		///printf("m_matT:\n");
		///for (i = 0; i < size; i++)
		///{
		///for (j = 0; j < size; j++)
		///printf("%f ", *(m_matT + i*n_cols + j));
		///printf("\n");
		///}


		applyHouseholderOnTheRight(H3, 2, hessen_Q, size, temp, 0, iu - 1, size);
	}
	///else
	///printf("\n===================\n");

	// clean up pollution due to round-off errors
	for (i = im + 2; i <= iu; ++i)
	{
		*(m_matT + i*n_cols + i - 2) = 0.0;
		if (i > im + 2)
			*(m_matT + i*n_cols + i - 3) = 0.0;
	}

	///printf("m_matT:\n");
	///for (i = 0; i < size; i++)
	///{
	///for (j = 0; j < size; j++)
	///printf("%f ", *(m_matT + i*n_cols + j));
	///printf("\n");
	///}
	///printf("m_matU:\n");
	///for (i = 0; i < size; i++)
	///{
	///for (j = 0; j < size; j++)
	///printf("%f ", *(hessen_Q + i*N10 + j));
	///printf("\n");
	///}
}

//compute real Schur decomposition
__host__ __device__ int RealSchur(double action_mat_eig[N10][N10], double std_data[N10*N10], double hessen_Q[N10*N10])
{
	//for k = 1:8//Matlab codes
	//u = temp(k + 1:end, k);
	//if u(1)>0
	//u(1) = u(1) + norm(u);
	//else
	//u(1) = u(1) - norm(u);
	//end
	//v = u / norm(u);
	//hous = eye(10 - k) - 2 * v*v';
	//u = eye(10);
	//u(k + 1:end, k + 1 : end) = hous;
	//if k == 1
	//P = u;
	//else
	//P = P*u;
	//end
	//temp = u*temp
	//temp = temp*u
	//%temp is the hesseign upper triangular matrix
	//end
	//Step 1. Reduce to Hessenberg form
	int i, j, k, l;
	double scale = 0, temp_scale, u[N10], P[N10*N10], temp[N10*N10];
	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
		{
			temp_scale = fabs(action_mat_eig[i][j]);
			if (temp_scale > scale)
				scale = temp_scale;
		}
	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
			*(std_data + i*N10 + j) = action_mat_eig[i][j] / scale;
	//Hessenberg form of matrix
	for (k = 0; k < N10 - 2; k++)
	{
		for (i = k + 1; i < N10; i++)
			u[i] = *(std_data + i*N10 + k);
		temp_scale = 0;
		for (i = k + 1; i < N10; i++)
			temp_scale += u[i] * u[i];
		temp_scale = sqrt(temp_scale);
		if (u[k + 1] > 0)
			u[k + 1] += temp_scale;
		else
			u[k + 1] -= temp_scale;
		temp_scale = 0;
		for (i = k + 1; i < N10; i++)
			temp_scale += u[i] * u[i];
		temp_scale = sqrt(temp_scale);
		for (i = k + 1; i < N10; i++)
			u[i] = u[i] / temp_scale;
		for (i = 0; i < N10; i++)
			for (j = 0; j < N10; j++)
				if (i < k + 1 || j < k + 1)
					if (i != j)
						*(P + i*N10 + j) = 0;
					else
						*(P + i*N10 + j) = 1;
				else
					if (i == j)
						*(P + i*N10 + j) = 1 - 2 * u[i] * u[j];
					else
						*(P + i*N10 + j) = -2 * u[i] * u[j];
		if (k == 0)
			for (i = 0; i < N10; i++)
				for (j = 0; j < N10; j++)
					*(hessen_Q + i*N10 + j) = *(P + i*N10 + j);
		else
		{//temp=hessen_Q*P, 
		 //matrix_multi(hessen_Q, N1, N2, P, N3, temp);
		 //the parameters of N1 and N2 are respectively the numbers of rows and columns of hessen_Q
		 //N3 is the number of columns of P
		 //temp is the output in size of the N1 rows and N3 columns 
			matrix_multi(hessen_Q, N10, N10, P, N10, temp);
			for (i = 0; i < N10; i++)
				for (j = 0; j < N10; j++)
					*(hessen_Q + i*N10 + j) = *(temp + i*N10 + j);
		}
		matrix_multi(P, N10, N10, std_data, N10, temp);
		matrix_multi(temp, N10, N10, P, N10, std_data);
	}//std_data is the hessenberg form of
	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
			if (i>j + 1)
				*(std_data + i*N10 + j) = 0.0;

	///printf("m_matT:\n");
	///for (i = 0; i < N10; i++)
	///{
	///for (j = 0; j < N10; j++)
	///printf("%f ", *(std_data + i*N10 + j));
	///printf("\n");
	///}
	///printf("hessen_Q:\n");
	///for (i = 0; i < N10; i++)
	///{
	///for (j = 0; j < N10; j++)
	///printf("%f ", *(hessen_Q + i*N10 + j));
	///printf("\n");
	///}
	///FILE *fp;
	//fp = fopen("matrixH.txt", "r");
	//i = readNDoubles(fp, std_data, 100, 0);
	//fclose(fp);
	//fp = fopen("matrixQ.txt", "r");
	//i = readNDoubles(fp, hessen_Q, 100, 0);
	//fclose(fp);

	//Step 2. Reduce to real Schur form  
	int maxIters, iu, iter = 0, totalIter = 0, il, im;
	iu = N10 - 1;
	double shiftInfo[3], exshift = 0, workspace[N10];
	//std_data=(tij), temp_scale=|t00|+|t01|+...+|t10|+|t11|+...+|tnn|
	temp_scale = computeNormOfT(std_data, N10, N10);
	maxIters = m_maxIterationsPerRow * N10;
	if (temp_scale != 0)
	{
		while (iu >= 0)
		{
			il = findSmallSubdiagEntry(iu, std_data, N10);
			// Check for convergence
			if (il == iu) // One root found
			{
				*(std_data + iu*N10 + iu) = *(std_data + iu*N10 + iu) + exshift;
				if (iu > 0)
					*(std_data + iu*N10 + iu - 1) = 0.0;
				iu--;
				iter = 0;
			}
			else if (il == iu - 1) // Two roots found
			{
				splitOffTwoRows(std_data, hessen_Q, N10, iu, &exshift);
				iu -= 2;
				iter = 0;
			}
			else // No convergence yet
			{
				// The firstHouseholderVector vector has to be initialized to something to get rid of a silly GCC warning (-O1 -Wall -DNDEBUG )
				double firstHouseholderVector[3];
				for (i = 0; i < 3; i++)
					*(firstHouseholderVector + i) = 0.0;
				computeShift(std_data, N10, iu, iter, &exshift, shiftInfo);
				iter = iter + 1;
				totalIter = totalIter + 1;
				if (totalIter > maxIters) break;
				im = initFrancisQRStep(std_data, N10, il, iu, shiftInfo, firstHouseholderVector);
				performFrancisQRStep(std_data, hessen_Q, N10, il, im, iu, firstHouseholderVector);
			}
		}
	}
	for (i = 0; i < N10; i++)
		for (j = 0; j < N10; j++)
			*(std_data + i*N10 + j) *= scale;

	///printf("m_matT:\n");
	///for (i = 0; i < N10; i++)
	///{
	///for (j = 0; j < N10; j++)
	///printf("%f ", *(std_data + i*N10 + j));
	///printf("\n");
	///}
	///printf("hessen_Q:\n");
	///for (i = 0; i < N10; i++)
	///{
	///for (j = 0; j < N10; j++)
	///printf("%f ", *(hessen_Q + i*N10 + j));
	///printf("\n");
	///}
	if (totalIter <= maxIters)
		return 1;
	else
		return 0;
}

__host__ __device__ void extract_eigenvalue(double *std_data, int size, int m_info, double2 *m_eivalues)
{
	double p, z, t0, t1, maxval, p0;
	int i;
	if (m_info == 1)
	{
		// Compute eigenvalues from matT
		i = 0;
		while (i < size)
		{
			if (i == size - 1 || *(std_data + (i + 1)*size + i) == 0)
			{
				*(m_eivalues + i) = { *(std_data + i*size + i), 0 };
				++i;
			}
			else
			{
				p = 0.5 * (*(std_data + i*size + i) - *(std_data + (i + 1)*size + i + 1));
				// Compute z = sqrt(abs(p * p + m_matT.coeff(i+1, i) * m_matT.coeff(i, i+1)));
				// without overflow
				{
					t0 = *(std_data + (i + 1)*size + i);
					t1 = *(std_data + i *size + i + 1);
					if (fabs(t0) > fabs(t1))
						maxval = fabs(t0);
					else
						maxval = fabs(t1);
					if (fabs(p) > maxval)
						maxval = fabs(p);

					t0 /= maxval;
					t1 /= maxval;
					p0 = p / maxval;
					z = maxval * sqrt(fabs(p0 * p0 + t0 * t1));
				}

				*(m_eivalues + i) = { *(std_data + (i + 1)*size + i + 1) + p, z };
				*(m_eivalues + i + 1) = { *(std_data + (i + 1)*size + i + 1) + p, -z };
				i += 2;
			}
		}
	}
	else
	{
		for (i = 0; i < size; i++)
			*(m_eivalues + i) = { 0.0, 0.0 };
	}
}

__host__ __device__ void cal_comp_vecs(double2 *m_eivalues, double2 *m_eivec, int size)
{
	int j, i;
	double precision = 2 * epsilon, temp, temp1;
	for (j = 0; j<size; ++j)
	{
		if ((m_eivalues[j].y< precision) | j + 1 == size)
		{
			// we have a real eigen value
			temp = 0;
			for (i = 0; i < size; i++)
			{
				temp += (*(m_eivec + i*size + j)).x*(*(m_eivec + i*size + j)).x;
			}
			temp = sqrt(temp);
			for (i = 0; i < size; i++)
				*(m_eivec + i*size + j) = { (*(m_eivec + i*size + j)).x / temp, 0.0 };
		}
		else
		{
			// we have a pair of complex eigen values
			temp = 0;
			for (i = 0; i < size; ++i)
			{
				temp += (*(m_eivec + i*size + j)).x*(*(m_eivec + i*size + j)).x + (*(m_eivec + i*size + j)).y + (*(m_eivec + i*size + j)).y;
				temp += (*(m_eivec + i*size + j + 1)).x*(*(m_eivec + i*size + j + 1)).x + (*(m_eivec + i*size + j + 1)).y + (*(m_eivec + i*size + j + 1)).y;
			}
			temp = sqrt(temp);
			for (i = 0; i < size; ++i)
			{
				temp1 = (*(m_eivec + i*size + j)).x;
				*(m_eivec + i*size + j) = { temp1 / temp, (*(m_eivec + i*size + j + 1)).x / temp };
				*(m_eivec + i*size + j + 1) = { temp1 / temp, -(*(m_eivec + i*size + j + 1)).x / temp };
			}
			++j;
		}
	}
}

__host__ __device__ void extract_eigenvector(double *m_matT, double *m_eivec, int size, int m_info, double2 *m_eivalues, double2 *c_m_eivec)
{
	int i, j, j0, j1, k, n, l, comp_vec = 1;
	double absnorm = 0.0, p, q, lastr, lastw, w, r, temp1, temp2, m_tmp[N10];
	double 	x, y, denom, lastra, lastsa, ra, sa, vr, vi, t;
	double2 cc;

	///FILE *fp;
	//fp = fopen("schur_mat.txt", "r");
	//i = readNDoubles(fp, m_matT, 100, 0);
	//fclose(fp);
	//fp = fopen("schur_eivec.txt", "r");
	//i = readNDoubles(fp, m_eivec, 100, 0);
	//fclose(fp);


	if (m_info == 1)
	{
		for (j = 0; j < size; ++j)
		{
			j0 = j - 1 > 0 ? j - 1 : 0;
			for (k = j0; k < size; k++)
				absnorm += fabs(*(m_matT + j*size + k));
		}
		// Backsubstitute to find vectors of upper triangular form
		if (absnorm == 0)
		{
			return;
		}

		for (n = size - 1; n >= 0; n--)
		{
			///printf("m_matT:\n");
			///for (i = 0; i < N10; i++)
			///{
			///for (j = 0; j < N10; j++)
			///printf("%f ", *(m_matT + i*N10 + j));
			///printf("\n");
			///}

			//if (n == 5|n==4)
			//printf("need test\n");
			p = m_eivalues[n].x;
			q = m_eivalues[n].y;

			// Scalar vector
			if (q == 0)
			{
				l = n;

				*(m_matT + n*size + n) = 1.0;
				for (i = n - 1; i >= 0; i--)
				{
					w = *(m_matT + i*size + i) - p;
					r = 0.0;
					for (k = l; k<n + 1; k++)
						r += *(m_matT + i*size + k)**(m_matT + k*size + n);

					if (m_eivalues[i].y < 0)
					{
						lastw = w;
						lastr = r;
					}
					else
					{
						l = i;
						if (m_eivalues[i].y == 0)
						{
							if (w != 0)
								*(m_matT + i*size + n) = -r / w;
							else
								*(m_matT + i*size + n) = -r / (epsilon * absnorm);
						}
						else // Solve real equations
						{
							x = *(m_matT + i*size + i + 1);
							y = *(m_matT + (i + 1)*size + i);
							denom = (m_eivalues[i].x - p) * (m_eivalues[i].x - p) + m_eivalues[i].y * m_eivalues[i].y;
							t = (x * lastr - lastw * r) / denom;
							*(m_matT + i*size + n) = t;
							if (abs(x) > abs(lastw))
								*(m_matT + (i + 1)*size + n) = (-r - w * t) / x;
							else
								*(m_matT + (i + 1)*size + n) = (-lastr - y * t) / lastw;
						}

						// Overflow control
						t = fabs(*(m_matT + i*size + n));
						if ((epsilon * t) * t > 1)
							for (k = i; k<size; k++)
								*(m_matT + n + k*size) /= t;
					}
				}
			}
			else if (q < 0 && n > 0) // Complex vector
			{
				l = n - 1;

				// Last vector component imaginary so matrix is triangular
				if (fabs(*(m_matT + n*size + n - 1)) > fabs(*(m_matT + (n - 1)*size + n)))
				{
					*(m_matT + (n - 1)*size + n - 1) = q / *(m_matT + n*size + n - 1);
					*(m_matT + (n - 1)*size + n) = -(*(m_matT + n*size + n) - p) / *(m_matT + n*size + n - 1);
				}
				else
				{
					temp1 = -*(m_matT + (n - 1)*size + n);
					temp2 = *(m_matT + (n - 1)*size + n - 1) - p;

					cc = { temp1*q / (temp2*temp2 + q*q),temp1*temp2 / (temp2*temp2 + q*q) };
					*(m_matT + (n - 1)*size + n - 1) = cc.x;
					*(m_matT + (n - 1)*size + n) = cc.y;
				}
				*(m_matT + n*size + n - 1) = 0.0;
				*(m_matT + n*size + n) = 1;
				for (i = n - 2; i >= 0; i--)
				{
					ra = 0;
					sa = 0;
					for (j = l; j < n + 1; j++)
					{
						ra += *(m_matT + i*size + j)**(m_matT + j*size + n - 1);
						sa += *(m_matT + i*size + j)**(m_matT + j*size + n);
					}
					w = *(m_matT + i*size + i) - p;

					if (m_eivalues[i].y < 0)
					{
						lastw = w;
						lastra = ra;
						lastsa = sa;
					}
					else
					{
						l = i;
						if (m_eivalues[i].y == 0)
						{
							cc = { -(ra*w + sa*q) / (w*w + q*q),(ra*q - sa*w) / (w*w + q*q) };
							*(m_matT + i*size + n - 1) = cc.x;
							*(m_matT + i*size + n) = cc.y;
						}
						else
						{
							// Solve complex equations
							x = *(m_matT + i*size + i + 1);
							y = *(m_matT + (i + 1)*size + i);
							vr = (m_eivalues[i].x - p) * (m_eivalues[i].x - p) + m_eivalues[i].y * m_eivalues[i].y - q * q;
							vi = (m_eivalues[i].x - p) * 2.0 * q;
							if ((vr == 0) && (vi == 0))
								vr = epsilon * absnorm * (fabs(w) + fabs(q) + fabs(x) + fabs(y) + fabs(lastw));

							temp1 = x*lastra - lastw*ra + q*sa;
							temp2 = x*lastsa - lastw*sa - q*ra;
							cc = { (temp1*vr + temp2*vi) / (vr*vr + vi*vi),(-temp1*vi + temp2*vr) / (vr*vr + vi*vi) };
							*(m_matT + i*size + n - 1) = cc.x;
							*(m_matT + i*size + n) = cc.y;
							if (abs(x) > (abs(lastw) + abs(q)))
							{
								*(m_matT + (i + 1)*size + n - 1) = (-ra - w * *(m_matT + i*size + n - 1) + q * *(m_matT + i*size + n)) / x;
								*(m_matT + (i + 1)*size + n) = (-sa - w * *(m_matT + i*size + n) - q * *(m_matT + i*size + n - 1)) / x;
							}
							else
							{
								temp1 = -lastra - y**(m_matT + i*size + n - 1);
								temp2 = -lastsa - y**(m_matT + i*size + n);
								cc = { (temp1*lastw + temp2*q) / (lastw*lastw + q*q),(-temp1*q + temp2*lastw) / (lastw*lastw + q*q) };
								*(m_matT + (i + 1)*size + n - 1) = cc.x;
								*(m_matT + (i + 1)*size + n) = cc.y;
							}
						}

						// Overflow control
						temp1 = fabs(*(m_matT + i*size + n - 1));
						temp2 = fabs(*(m_matT + i*size + n));
						t = temp1 > temp2 ? temp1 : temp2;
						if ((epsilon * t) * t > 1)
							for (j = i; j < size; j++)
								for (k = n - 1; k < n + 1; k++)
									*(m_matT + j*size + k) /= t;

					}
				}

				// We handled a pair of complex conjugate eigenvalues, so need to skip them both
				n--;
			}
			else
			{
				printf("Internal bug in EigenSolver (INF or NaN has not been detected)"); // this should not happen
			}
		}

		///printf("m_matT:\n");
		///for (i = 0; i < N10; i++)
		///{
		///for (j = 0; j < N10; j++)
		///printf("%f ", *(m_matT + i*N10 + j));
		///printf("\n");
		///}
		///printf("eigen vetors:\n");
		///for (i = 0; i < N10; i++)
		///{
		///for (j = 0; j < N10; j++)
		///printf("%f ", *(m_eivec + i*N10 + j));
		///printf("\n");
		///}

		// Back transformation to get eigenvectors of original matrix
		for (j = size - 1; j >= 0; j--)
		{
			for (i = 0; i<size; i++)
			{
				m_tmp[i] = 0.0;
				//for(k=size-j-1;k<size;k++)
				for (k = 0; k < j + 1; k++)
					m_tmp[i] += *(m_eivec + i*size + k)**(m_matT + k*size + j);
			}
			for (i = 0; i<size; i++)
				*(m_eivec + i*size + j) = m_tmp[i];
		}

		///printf("eigen vetors:\n");
		///fp = fopen("eigvec.txt", "w");
		///for (i = 0; i < N10; i++)
		///{
		///for (j = 0; j < N10; j++)
		///{
		///printf("%f ", *(m_eivec + i*N10 + j));
		///fprintf(fp, "%f ", *(m_eivec + i*N10 + j));
		///}
		///printf("\n");
		///fprintf(fp, "\n");
		///}
		///fclose(fp);

		if (comp_vec)
		{
			for (i = 0; i < size; i++)
				for (j = 0; j < size; j++)
					(*(c_m_eivec + i*size + j)) = { *(m_eivec + i*size + j),0.0 };
			cal_comp_vecs(m_eivalues, c_m_eivec, N10);
		}

	}//if (m_info == 1)
	else//if (m_info != 1)
		for (i = 0; i < size; i++)
			for (j = 0; j < size; j++)
				*(m_eivec + i*size + j) = 0.0;

}

__host__ __device__ int find_essential(float *nrmlz_frames0x, float *nrmlz_frames0y, float *nrmlz_frames1x, float *nrmlz_frames1y, int n_matches, int *cm_ind, int *hit_ind, double E[ndof])
{// (1) Extract 4 null vectors from linear equations of epipolar constraint
		double null_space_mat[N3][N3][4], constraint_mat[10][20], ee[36], temp; // 9*4
		int i, j, k=0;
		//Initialize the essential equations and then solve the equations using LU chase method
		//Input: pts(corresondence points: (dev_nrmlz_frames0x,dev_nrmlz_frames0y)<------>(dev_nrmlz_frames1x, dev_nrmlz_frames1y)
		//sample: cm_ind[hit_ind} (5 integer array indicate the row index of pts)
		//Output: ee (4 null tensors in shape of 4*9 )
		null_space(nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, cm_ind, hit_ind, ee);

		for (k = 0; k < 4; k++)
			for (j = 0; j < 3; j++)
				for (i = 0; i < 3; i++)
					null_space_mat[i][j][k] = ee[k*ndof + j*N3 + i];

		// (2) Use the rank constraint and the trace constraint to build ten third-order polynomial
		// equations in the three unknowns. The monomials are ordered in GrLex order and
		// represented in a 10×20 matrix, where each row corresponds to an equation and each column
		// corresponds to a monomial
		build10_20eqs(null_space_mat, constraint_mat);

		///FILE *fp;
		///fp = fopen("constraint_mat.txt", "w");
		///for (i = 0; i < 10; i++)
		///{
		///for (j = 0; j < 20; j++)
		///fprintf(fp, "%f ", constraint_mat[i][j]);
		///fprintf(fp, "\n");
		///}
		///fclose(fp);
		// (3) Compute the Gröbner basis. This turns out to be as simple as performing a
		// Gauss-Jordan elimination on the 10×20 matrix
		double X[N10][N10];
		Grobner_base(constraint_mat, X);

		// (4) Compute the 10×10 action matrix for multiplication by one of the un-knowns.
		// This is a simple matter of extracting the correct elements fromthe eliminated
		// 10×20 matrix and organising them to form the action matrix.
		double action_mat_eig[N10][N10];
		for (i = 0; i < N10; i++)
			for (j = 0; j < N10; j++)
				if (i < 3)
					action_mat_eig[i][j] = X[i][j];
				else
					if (i == 3 || i == 4)
						action_mat_eig[i][j] = X[i + 1][j];
					else
						if (i == 5)
							action_mat_eig[i][j] = X[i + 2][j];
						else
							action_mat_eig[i][j] = 0;
		action_mat_eig[6][0] = -1.0;
		action_mat_eig[7][1] = -1.0;
		action_mat_eig[8][3] = -1.0;
		action_mat_eig[9][6] = -1.0;

		// (5) Compute the left eigenvectors of the action matrix
		double std_data[N10*N10], hessen_Q[N10*N10], p, z, t0, t1, maxval, p0;
		double2 m_eivalues[N10], eig_vecs[N10*N10];
		int m_info;
		m_info = RealSchur(action_mat_eig, std_data, hessen_Q);

		extract_eigenvalue(std_data, N10, m_info, m_eivalues);
		extract_eigenvector(std_data, hessen_Q, N10, m_info, m_eivalues, eig_vecs);

		///printf("eigen vetors:\n");
		///FILE *fp;
		///fp = fopen("eigvec.txt", "w");
		///for (i = 0; i < N10; i++)
		///{
		///for (j = 0; j < N10; j++)
		///{
		//printf("%f %f ", (*(eig_vecs + i*N10 + j)).x, (*(eig_vecs + i*N10 + j)).y);
		///fprintf(fp, "%f ", (*(eig_vecs + i*N10 + j)).x, (*(eig_vecs + i*N10 + j)).y);
		///}
		///printf("\n");
		///fprintf(fp, "\n");
		///}
		///fclose(fp);

		///printf("eigen values:\n");
		///fp = fopen("eigval.txt", "w");
		///for (j = 0; j < N10; j++)
		///{
			///printf("%f %f ", (*(m_eivalues + j)).x, (*(m_eivalues + j)).y);
			///fprintf(fp, "%f ", (*(m_eivalues + j)).x, (*(m_eivalues + j)).y);
		///}
		///printf("\n");
		///fprintf(fp, "\n");
		///fclose(fp);

		//(6) Compute essential models
		k = 0;
		for (i = 0; i < N10; i++)
		{
			if (m_eivalues[i].y == 0)
			{
				for (j = 0; j<9; j++)
					E[k * 9 + j] = ee[j] * eig_vecs[60 + i].x + ee[j + 9] * eig_vecs[70 + i].x +
					ee[j + 18] * eig_vecs[80 + i].x + ee[j + 27] * eig_vecs[90 + i].x;
				k++;
			}
		}
		return k;
}

/*
* Do oriented constraint to verify if epipolar geometry is in front or behind the camera.
* Return: true if all points are in front of the camers w.r.t. tested epipolar geometry - satisfies constraint.
*         false - otherwise.
* x'^T F x = 0
* e' × x' ~+ Fx   <=>  λe' × x' = Fx, λ > 0
* e  × x ~+ x'^T F
*/
__host__ __device__ int isModelValid(double *F_, int n_E, int *cm_ind, int *hit_ind, int min_sample_size, float *dev_nrmlz_frames0x, float *dev_nrmlz_frames0y, float *dev_nrmlz_frames1x, float *dev_nrmlz_frames1y)
{
	double ep[3], ep_[3], *F, sign1;
	int i, j, k, val_tag = 1, pt, current_count = 0;
	//f0 f1 f2
	//f3 f4 f5
	//f6 f7 f8
	for (k = 0; k < n_E; k++)
	{
		val_tag = 1;
		F = F_ + k * 9;
		ep_[0] = *(F + 1)**(F + 8) - *(F + 2)**(F + 7);//cross(F_[0][0],F_[0][1],F_[0][2]),(F_[2][0],F_[2][1],F_[2][2])
		ep_[1] = -*F **(F + 8) + *(F + 2)**(F + 6);
		ep_[2] = *F**(F + 7) - *(F + 1)**(F + 6);
		for (i = 0; i < 3; i++)
			ep[i] = ep_[i];
		if (ep[0] <= DBL_EPSILON && ep[0] > -DBL_EPSILON &&
			ep[1] <= DBL_EPSILON && ep[1] > -DBL_EPSILON &&
			ep[2] <= DBL_EPSILON && ep[2] > -DBL_EPSILON)
		{
			ep[0] = *(F + 4)**(F + 8) - *(F + 5)**(F + 7);//cros(F.row(1),F.row(2))
			ep[1] = -*(F + 3) **(F + 8) + *(F + 5)**(F + 6);
			ep[2] = *(F + 3)**(F + 7) - *(F + 4)**(F + 6);
		}

		// without loss of generality, let the first point in sample be in front of the camera.
		pt = cm_ind[hit_ind[0]];
		// check only two first elements of vectors (e × x) and (x'^T F)
		// s1 = (x'^T F)[0] = x2 * F11 + y2 * F21 + 1 * F31
		// s2 = (e × x)[0] = e'_2 * 1 - e'_3 * y1
		// sign1 = s1 * s2
		sign1 = (F[0] * *(dev_nrmlz_frames1x + pt) + F[3] * *(dev_nrmlz_frames1y + pt) + F[6])*(ep[1] - ep[2] * *(dev_nrmlz_frames0y + pt));

		for (int i = 1; i < min_sample_size; i++) {
			pt = cm_ind[hit_ind[i]];
			// if signum of the first point and tested point differs
			// then two points are on different sides of the camera.
			if (sign1*(F[0] * *(dev_nrmlz_frames1x + pt) + F[3] * *(dev_nrmlz_frames1y + pt) + F[6])*(ep[1] - ep[2] * *(dev_nrmlz_frames0y + pt))<0)
				val_tag = 0;
		}
		if (val_tag)
		{//if the k-th model is a valid model, move current model point to the next with current_count++
		 //otherwise, don't move the valid model point.
			for (i = 0; i < 9; i++)//move model
				*(F_ + current_count * 9 + i) = *(F + i);
			current_count++;
		}
	}
	return current_count;
}


//RANSAC implementation
//(dev_frames0x[X[0]],dev_frames0y[X[0]]) is a sample (x,y)-coordinate in image1, (dev_frames1x[X[0]],dev_frames1y[X[0]]) is a sample (x,y)-coordinate in image2. They are a match pair.
//Output:result_ninlers is a vector with the length gridDim stored in Global memory, in which the i-th element is the maximum number of inliers in the i-th block.
//result_MSS is a vector with the length gridDim*7 involving MSS stored in Global memory, which is in form of (i*7, i*7+1, i*7+2, i*7+3,..., i*7+7) in the i-th block.

//Epipolar geometry model: exploring fundamental matrix
//we use 7-points to estimate fundatmental matrix
__global__ void RANSAC_fund(float *dev_nrmlz_frames0x, float *dev_nrmlz_frames0y, float *dev_nrmlz_frames1x, float *dev_nrmlz_frames1y, int n_matches, double inlier_thresh, int s_d_check, int *cm_ind, int card_core, int *result_ninliers, int *result_MSS)
{
	int hit_ind[essential_fd], i, j, n_E=1, temp_count = 0, k = 0, count = 0;

	double  E[N10*ndof], *f1, F_temp[N3][N3];

	//inlier_count is in size of blockDim.x, every thread read and write its threadId-th element.
	//The blockDim.x elements in inlier_count are the number of inliers produced by every 8-samples test
	//The blockDim.x*8 elements in inlier_count_7points are the interest of 8-samples index in dev_frames.
	extern __shared__ int inlier_count_5points[];
	unsigned int tid = threadIdx.x;
	unsigned int ind_count = tid;
	inlier_count_5points[ind_count] = 0;
	//thread 0 is in charge of the read and write elements: blockDim.x, blockDim.x + 1, ..., blockDim.x + 7
	//thread 1 is in charge of the read and write elements: blockDim.x+8, blockDim.x + 9, ..., blockDim.x + 15
	for (i = 0; i<essential_fd; i++)
		inlier_count_5points[blockDim.x + essential_fd * ind_count + i] = 0;

	//	if (blockIdx.x == 0)
	//	{
	//		printf("blockIdx.x=%d, tid=%d  inlier_count_Io4si=%d  inlier_count_Io4si[%d]=%d \n ", blockIdx.x, threadIdx.x, inlier_count_Io4si[ind_count], ind_count + blockDim.x, inlier_count_Io4si[ind_count + blockDim.x]);
	//		printf( "dev_n_round[0]=%d, dev_filtered_4samples_list[hit_ind[%d]].x=(%d, %d, %d, %d)\n", dev_n_round[0], tid, dev_filtered_4samples_list[hit_ind[tid]].x, dev_filtered_4samples_list[hit_ind[tid]].y, dev_filtered_4samples_list[hit_ind[tid]].z, dev_filtered_4samples_list[hit_ind[tid]].w);
	//	}
	__syncthreads();//syncthrize in block
					//	__threadfence();//syncthrize in grid
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	*result_ninliers = 0;
	//	printf("tid=%d, dev_n_round[0]=%d, gridDim.x=%d, blockDim.x=%d, gridDim.x * blockDim.x=%d\n", tid, dev_n_round[0], gridDim.x, blockDim.x, gridDim.x * blockDim.x);
	//dev_n_round is a global variable
	while (count < dev_n_round[0])
	{
		//produce 7 different intergers to direct sample, samples are drawn in core, and the consensus are checked in n_matches
		//hit_ind is the output that sampled index of the core
		//printf("threadIdx.x:%d, %d, %d\n", threadIdx.x, card_core, s_d_check);
		get_nrand(hit_ind, essential_fd, card_core, s_d_check);
		//hit_ind[0] = 4; hit_ind[1] = 5; hit_ind[2] = 1; hit_ind[3] = 2;
		//hit_ind[4] = 3;

		n_E = find_essential(dev_nrmlz_frames0x, dev_nrmlz_frames0y, dev_nrmlz_frames1x, dev_nrmlz_frames1y, n_matches, cm_ind, hit_ind, E);

		n_E = isModelValid(E, n_E, cm_ind, hit_ind, essential_fd, dev_nrmlz_frames0x, dev_nrmlz_frames0y, dev_nrmlz_frames1x, dev_nrmlz_frames1y);

		for (i = 0; i < n_E; i++)
		{
			f1 = E + i * 9;
			for (j = 0; j < N3; j++)
				for (k = 0; k<N3; k++)
					F_temp[j][k] = f1[j*N3 + k];

		//temp_count is the returned number of consensus points.
			temp_count = consensus_check(n_matches, F_temp, dev_nrmlz_frames0x, dev_nrmlz_frames0y, dev_nrmlz_frames1x, dev_nrmlz_frames1y, inlier_thresh);
			if (temp_count > inlier_count_5points[ind_count])
			{
				inlier_count_5points[ind_count] = temp_count;
				///printf("test in ransac_fund: %d\n", temp_count);
				for (j = 0; j < essential_fd; j++)//if the thread with id tid results a better results, update MSS
				{
					inlier_count_5points[blockDim.x + essential_fd * ind_count + j] = cm_ind[hit_ind[j]]; //========================cm_ind[i];for the fixed MSS test
					///printf("%d ", cm_ind[hit_ind[j]]);
				}
				///printf("\n");
				///printf("F_temp: %d\n", temp_count);
				///for (j = 0; j < N3; j++)
					///for (k = 0; k < N3; k++)
						///printf("%f ", F_temp[j][k]);
				//if (temp_count > 0.1*dim_m_I_cn24[0])//used for sudden death===========
					//atomicExch(&results[0].x, 1);//used for sudden death===============
			}
		}

		tid += gridDim.x * blockDim.x;
		count++;
	}
	__syncthreads();//syncthrize in block
	if (threadIdx.x == 0)//0th thread is used to write back results in golbal memory.
	{
		for (i = 0; i < blockDim.x; i++)//the 0-th thread collects the best results in shared memory
			if (inlier_count_5points[i] > *(result_ninliers + blockIdx.x))
			{
				//inlier_count[0] = inlier_count[i];
				//for (k = 0; k < isnp; k++)int *result_ninliers, int *result_MSS)
				//inlier_count_7points[k * blockDim.x] = inlier_count_7points[i + k * blockDim.x];
				*(result_ninliers + blockIdx.x) = inlier_count_5points[i];//write result back to global memory
				for (k = 0; k < essential_fd; k++)
					result_MSS[essential_fd * blockIdx.x + k] = inlier_count_5points[blockDim.x + essential_fd * i + k];
			}
		//printf("%d, %d\n", blockIdx.x, *(result_ninliers + blockIdx.x));

	}
}

//host_result_MSS: point to the head of an 7 points MSS
//output:
//CS_temp: if CS_temp[i]=1, the i-th correpondence is a true correspondence
//function value is the number of inliers evaluated with the model modeled by the MSS host_result_MSS
int assembleinliers(int *host_result_MSS, float *nrmlz_frames0x, float *nrmlz_frames0y, float *nrmlz_frames1x, float *nrmlz_frames1y, float *host_cn2_error_beforesort, double inlier_thresh, int n_matches, int *CS_temp)
{
	int hit_ind[] = { 0,1,2,3,4 }, i, j, k = 0, n_E, temp_count, inlier_count = 0;
	double  E[N10*ndof], *f1, F_temp[N3][N3], E_star[N3][N3];
	double Fx1_temp0, Fx1_temp1, Fx1_temp2, x2tFx1_temp, Ftx2_temp0, Ftx2_temp1;
	

	n_E = find_essential(nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, n_matches, host_result_MSS, hit_ind, E);

	//n_E = isModelValid(E, n_E, cm_ind, hit_ind, essential_fd, nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y);


	for (i = 0; i < n_E; i++)
	{
		f1 = E + i * 9;
		for (j = 0; j < N3; j++)
			for (k = 0; k<N3; k++)
				F_temp[j][k] = f1[j*N3 + k];


		//temp_count is the returned number of consensus points.
		//count the number of inliers using the resulting fundamental matrix F_temp
		temp_count = consensus_check(n_matches, F_temp, nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, inlier_thresh);
		if (temp_count > inlier_count)
		{
			inlier_count = temp_count;
			for (j = 0; j < N3; j++)
				for (k = 0; k < N3; k++)
					E_star[j][k] = F_temp[j][k];
		}
	}
	//prepare for CS_temp
	inlier_count = 0;
	for (k = 0; k < n_matches; k++)
	{
		Fx1_temp0 = E_star[0][0] * *(nrmlz_frames0x + k) + E_star[0][1] * *(nrmlz_frames0y + k) + E_star[0][2];
		Fx1_temp1 = E_star[1][0] * *(nrmlz_frames0x + k) + E_star[1][1] * *(nrmlz_frames0y + k) + E_star[1][2];
		Fx1_temp2 = E_star[2][0] * *(nrmlz_frames0x + k) + E_star[2][1] * *(nrmlz_frames0y + k) + E_star[2][2];
		x2tFx1_temp = *(nrmlz_frames1x + k)*Fx1_temp0 + *(nrmlz_frames1y + k)*Fx1_temp1 + Fx1_temp2;
		Ftx2_temp0 = E_star[0][0] * *(nrmlz_frames1x + k) + E_star[1][0] * *(nrmlz_frames1y + k) + E_star[2][0];
		Ftx2_temp1 = E_star[0][1] * *(nrmlz_frames1x + k) + E_star[1][1] * *(nrmlz_frames1y + k) + E_star[2][1];
		x2tFx1_temp = x2tFx1_temp*x2tFx1_temp / (Fx1_temp0*Fx1_temp0 + Fx1_temp1*Fx1_temp1 + Ftx2_temp0*Ftx2_temp0 + Ftx2_temp1*Ftx2_temp1);
		//if (fabs(x2tFx1_temp) < T_dist & *(candi_matches+k))
		//printf("di:%f\n", x2tFx1_temp);
		if (x2tFx1_temp < inlier_thresh)
		{
			*(CS_temp + k) = 1;
			//*proj_error_temp += x2tFx1_temp;
			inlier_count++;
			//printf("di:%f, %d ", x2tFx1_temp, k);
		}
		else
			*(CS_temp + k) = 0;
	}
	return(inlier_count);
}

//output: near_ind_img1(i,j) is the top-N_mindist node to the i-th node, j=0,1,...,N_mindist-1
void nearest_dist(float *framesx, float *framesy, int N_I_star, int *inliers_candidates, int N_mindist, int *near_ind_img1)
{
	int i, j, k, min_ind;
	double dx, dy, *inliers_dist, temp;
	inliers_dist = (double *)malloc(N_I_star *N_I_star * sizeof(double));
	for (i = 0; i<N_I_star; i++)
		for (j = i + 1; j < N_I_star; j++)
		{
			dx = *(framesx + *(inliers_candidates + i)) - *(framesx + *(inliers_candidates + j));
			dy = *(framesy + *(inliers_candidates + i)) - *(framesy + *(inliers_candidates + j));
			*(inliers_dist + i*N_I_star + j) = dx*dx + dy*dy;
			*(inliers_dist + j*N_I_star + i) = *(inliers_dist + i*N_I_star + j);
		}
	for (i = 0; i < N_I_star; i++)
		*(inliers_dist + i*N_I_star + i) = 0;

	for (i = 0; i<N_I_star; i++)
		for (k = 0; k<N_mindist; k++)
		{
			temp = 10000000;
			min_ind = 0;
			for (j = 0; j < N_I_star; j++)
				if (temp > *(inliers_dist + i*N_I_star + j)&i != j)
				{
					temp = *(inliers_dist + i*N_I_star + j);
					min_ind = j;
				}//inliers_dist is a distance matrix, inliers_dist(i,j) is the distance between the i-th node and the j-th node
			*(inliers_dist + i*N_I_star + min_ind) = 10000000;//after finding a node nearrest to the current node, then let the finding node have a fixed large distance
			*(near_ind_img1 + i*N_mindist + k) = min_ind;//so that it cannot be found as the nearrest node in the next nearrest search loop
		}
	free(inliers_dist);
}

//output: partially assigned, CS_temp changes the indexes of outliers to 0, otherwise keep 1
//CS_outlieres_temp: partially assigned, CS_outlieres_temp changes the indexes of outliers to 1, otherwise keep 0
int outliers_detec(float *frames0x, float *frames0y, float *frames1x, float *frames1y, int n_matches, int N_I_star,int *CS_temp, int *CS_outlieres_temp)
{
	int i, j, k, t, temp, N_mindist = 6, deg0,deg1,frames_ind0,frames_ind1;
	int *inliers_candidates, *near_ind_img1, *near_ind_img2;
	inliers_candidates = (int *)malloc(N_I_star * sizeof(int));
	near_ind_img1 = (int *)malloc(N_mindist  *N_I_star * sizeof(int));
	near_ind_img2 = (int *)malloc(N_mindist  *N_I_star * sizeof(int));
	k = 0;
	for (i = 0; i < n_matches; i++)
	{
		*(CS_outlieres_temp + i) = 0;
		if (*(CS_temp + i))
		{
			*(inliers_candidates + k) = i;
			k++;
		}
	}
	///for (i = 0; i < N_I_star; i++)
		///printf("%d %f %f %f %f \n",*(inliers_candidates + i), *(frames0x+*(inliers_candidates + i)),
			///*(frames0y + *(inliers_candidates + i)), *(frames1x + *(inliers_candidates + i)), *(frames1y + *(inliers_candidates + i)));
	nearest_dist(frames0x, frames0y, N_I_star, inliers_candidates, N_mindist, near_ind_img1);
	nearest_dist(frames1x, frames1y, N_I_star, inliers_candidates, N_mindist, near_ind_img2);
	temp = 0;
	for (i = 0; i < N_I_star; i++)
	{
		frames_ind1 = *(inliers_candidates + i);//the current i-th node
		k = 0; deg0 = 0; deg1 = 0;
		for (j = 0; j < N_mindist; j++)
		{
			frames_ind0 = *(inliers_candidates + *(near_ind_img1 + i*N_mindist + j));//point to the nearrest N_mindist nodes to the current i-th node
			if (*(frames0x+ frames_ind0) == *(frames0x + frames_ind1)&*(frames0y + frames_ind0) == *(frames0y + frames_ind1))
				deg0++;//count many to one correspondences
		}
		for (t = 0; t < N_mindist; t++)
		{
			frames_ind0 = *(inliers_candidates + *(near_ind_img2 + i*N_mindist + t));
			if (*(frames1x + frames_ind0) == *(frames1x + frames_ind1)&*(frames1y + frames_ind0) == *(frames1y + frames_ind1))
				deg1++;//count many to one correspondences
		}
		if (deg1 > deg0)
			deg0 = deg1;
		
		for(j=0;j<N_mindist;j++)
			for (t = 0; t < N_mindist; t++)
				if (*(near_ind_img1 + i*N_mindist + j) == *(near_ind_img2 + i*N_mindist + t))
				{
					k++;//count the same neighbors
					break;
				}
			
		
		if (k < N_mindist - 2|deg0>4)//outliers detected (k < N_mindist - 2|deg>4)
		{
			*(CS_temp + *(inliers_candidates + i)) = 0;
			*(CS_outlieres_temp + *(inliers_candidates + i)) = 1;
			temp++;
		}
	}
	free(inliers_candidates);
	free(near_ind_img1);
	free(near_ind_img2);
	return(temp);
}


//fitting a model F using the inliers in *CS, where *(CS+i)=1 indicates an inlier, ninliers is the total inliers
//Output: F
void reestimate(float *frames0x, float *frames0y, float *frames1x, float *frames1y, int n_matches, int ninliers, int *CS, double F[N3][N3])
{
	int i, j, k;
	double T0[ndof], T1[ndof];
	float *in_frames0x, *in_frames0y, *in_frames1x, *in_frames1y, *nrmlz_in_frames0x, *nrmlz_in_frames0y, *nrmlz_in_frames1x, *nrmlz_in_frames1y;
	// A=USv^T, v9 is a column of v corresponding to the minimum singular value of in S.
	double *A, diag[ndof], v[ndof*ndof], rv1[ndof], v9[ndof];
	//Fundamental matrix is always a rank-2 matrix, we decompose it to svd, and let a minimum singular value be 0, and compose
	double diag3[N3], v3[N3*N3], rv13[N3], F_star[N3][N3];

	double2 *X1_inliers, *X2_inliers;
	int *ind_inliers;
	ind_inliers = (int *)calloc(ninliers, sizeof(int));
	in_frames0x = (float*)malloc(ninliers * sizeof(float));
	in_frames0y = (float*)malloc(ninliers * sizeof(float));
	in_frames1x = (float*)malloc(ninliers * sizeof(float));
	in_frames1y = (float*)malloc(ninliers * sizeof(float));
	nrmlz_in_frames0x = (float*)malloc(ninliers * sizeof(float));
	nrmlz_in_frames0y = (float*)malloc(ninliers * sizeof(float));
	nrmlz_in_frames1x = (float*)malloc(ninliers * sizeof(float));
	nrmlz_in_frames1y = (float*)malloc(ninliers * sizeof(float));

	X1_inliers = (double2*)malloc(ninliers * sizeof(double2));
	X2_inliers = (double2*)malloc(ninliers * sizeof(double2));

	A = (double*)malloc(ninliers * ndof * sizeof(double));


	j = 0;
	for (i = 0; i < n_matches; i++)
	{
		if (*(CS + i))
		{
			*(ind_inliers + j) = i;
			*(in_frames0x + j) = *(frames0x + i);
			*(in_frames0y + j) = *(frames0y + i);
			*(in_frames1x + j) = *(frames1x + i);
			*(in_frames1y + j) = *(frames1y + i);
			j++;
		}
	}
	normal_samples_fund(ninliers, in_frames0x, in_frames0y, nrmlz_in_frames0x, nrmlz_in_frames0y, T0);
	normal_samples_fund(ninliers, in_frames1x, in_frames1y, nrmlz_in_frames1x, nrmlz_in_frames1y, T1);
	for (i = 0; i<ninliers; i++)
	{
		X1_inliers[i].x = nrmlz_in_frames0x[i];//extract coordinates based on index
		X1_inliers[i].y = nrmlz_in_frames0y[i];
		X2_inliers[i].x = nrmlz_in_frames1x[i];//extract coordinates based on index
		X2_inliers[i].y = nrmlz_in_frames1y[i];
		//printf("i=%d, X1.x=%f,X1.y=%f,X2.x=%f,X2.y=%f\n", i, X1[i].x, X1[i].y, X2[i].x, X2[i].y);
	}
	initial_A_fund_host(A, X1_inliers, X2_inliers, ninliers);

	//implement singular value decomposition with thin type.
	i = svd(A, diag, v, rv1, ninliers, ndof);

	*rv1 = 10000.0;
	for (i = 0; i<ndof; i++)//search for the index of the minimum singular value
		if (*(diag + i) < *rv1)
		{
			*rv1 = *(diag + i);
			k = i;//keep the column index to obtain u9
		}

	for (i = 0; i < ndof; i++)//let u9 be the column of v corresponding to the minimum singular value
		v9[i] = *(v + i*ndof + k);

	//	printf("u9: %f, %f, %f, %f, %f", *v9, *(v9 + 1), *(v9 + 2), *(v9 + 3), *(v9 + 4));

	for (j = 0; j < N3; j++)
		for (i = 0; i < N3; i++)
			//to save thread memory, use A to store our interested 3*3 matrix
			//different from corresponding step in ransac, F is the 
			//v0 v1 v2         f00 f01 f02
			//v3 v4 v5    =F=  f10 f11 f12(there is a transpose from u to fundamental matrix F)
			//v6 v7 v8         f20 f21 f22
			*(A + j*N3 + i) = v9[j*N3 + i];

	i = svd(A, diag3, v3, rv13, N3, N3);
	min_diag_0(diag3);

	//F_star = USV^T (A*diag3*v3^T)
	usvt(A, diag3, v3, F_star);
	//Denormalize to original plan
	denormalization(F_star, T0, T1, F);

	free(ind_inliers);
	free(in_frames0x);
	free(in_frames0y);
	free(in_frames1x);
	free(in_frames1y);
	free(nrmlz_in_frames0x);
	free(nrmlz_in_frames0y);
	free(nrmlz_in_frames1x);
	free(nrmlz_in_frames1y);
	free(X1_inliers);
	free(X2_inliers);
	free(A);
	//return(ninliers);//don't need to return the value
}

//estimate projection error	on all the frames, rather on *CS indicated inliers because of error using the reestimated model F
//output: CS, CS_outlieres, proj_error
double est_proj_error(float *frames0x, float *frames0y, float *frames1x, float *frames1y, int n_matches, int *CS, int *CS_outlieres, int *n_inliers, double F[N3][N3])
{
	int i, n_eliminated;
	double *proj_error, mean_proj_error = 0.0, rv1[3], x1tfx0, rv2[3];
	proj_error = (double *)malloc(n_matches * sizeof(double));
	*n_inliers = 0;

	for (i = 0; i < n_matches; i++)
	{
		*(CS_outlieres + i) = 0;
		*rv1 = *(frames1x + i)*F[0][0] + *(frames1y + i)*F[1][0] + F[2][0];//x1^t*F(:,0)
		*(rv1 + 1) = *(frames1x + i)*F[0][1] + *(frames1y + i)*F[1][1] + F[2][1];//x1^t*F(:,1)
		*(rv1 + 2) = *(frames1x + i)*F[0][2] + *(frames1y + i)*F[1][2] + F[2][2];//x1^t*F(:,2)
		x1tfx0 = *rv1**(frames0x + i) + *(rv1 + 1)**(frames0y + i) + *(rv1 + 2);//x1^t*F*x0
		*rv2 = *(frames0x + i)*F[0][0] + *(frames0y + i)*F[0][1] + F[0][2];
		*(rv2 + 1) = *(frames0x + i)*F[1][0] + *(frames0y + i)*F[1][1] + F[1][2];
		*(rv2 + 2) = x1tfx0*x1tfx0 / (*rv1**rv1 + *(rv1 + 1)**(rv1 + 1) + *rv2**rv2 + *(rv2 + 1)**(rv2 + 1));
		*(CS + i) = 0;
		if (*(rv2 + 2)<thresh_sampson)
		{
			*(proj_error + i) = *(rv2 + 2);
			*(CS + i) = 1;
			(*n_inliers)++;
			//printf("i:%d, *rv1:%f, 2: %f\n", i, *rv1, proj_error);
		}
	}
	for (i = 0; i < 2; i++)
	{
		n_eliminated = outliers_detec(frames0x, frames0y, frames1x, frames1y, n_matches, *n_inliers, CS, CS_outlieres);
		*n_inliers -= n_eliminated;
	}
	for (i = 0; i < n_matches; i++)
		if (*(CS + i))
			mean_proj_error += *(proj_error + i);
	mean_proj_error = mean_proj_error / *n_inliers;
	return(mean_proj_error);
}


int main(int argc, char ** argv) {
	FILE *fp;
	int *matches1, *matches2;
	/*matches is an array points to the matrix of matched pairs*/
	/*<matches(1,j),matches(2,j)> is a pair of match between image I0 and I1*/
	/*matches(1,j) corresponds to the j-th points in frame1*/
	/*matches(2,j) corresponds to the j-th points in frame2*/
	float *f1x, *f1y, *f2x, *f2y, *frames0x, *frames0y, *frames1x, *frames1y;
	double *calib_points, *K1, *K2;
	int i, j, temp = 0, n_matches, n_frames0, n_frames1;
	char filename[] = "test01.txt";//comment this statement if filename receives string from main
	//char *filename;//designed for receiving string from main
	//filename = argv[1];//designed for receiving string from main

	char K_1[] = "1.txt", K_2[] = "2.txt";
	char fn_matches[200] = ".\\reichstag\\matches_";
	strcat(fn_matches, filename);
	char fn_frames0[200] = ".\\reichstag\\frames1_";
	strcat(fn_frames0, filename);
	char fn_frames1[200] = ".\\reichstag\\frames2_";
	strcat(fn_frames1, filename);
	int n_I0, r_I0, c_I0, n_I1, r_I1, c_I1;
	/*n_I0 is the total number of pixels in I0, n_I0=r_I0*c_I0*/
	/*r_I0 is the number of rows of I0*/
	/*c_I0 is the number of column of I0*/
	char fn_I0[200] = ".\\reichstag\\I1_";
	strcat(fn_I0, filename);
	char fn_I1[200] = ".\\reichstag\\I2_";
	strcat(fn_I1, filename);
	char fn_K2[500] = ".\\reichstag\\K_";
	char fn_K1[500] = ".\\reichstag\\K_";
	//strcat(fn_K1, filename);//sun3d
	//strcpy(fn_K2, fn_K1);//sun3d
	K_1[0] = filename[4];//reichstag
	K_2[0] = filename[5];//reichstag
	strcat(fn_K1, K_1);//reichstag
	strcat(fn_K2, K_2);//reichstag


	char file_csac_in[200] = ".\\reichstag\\csac_results\\csac_inliers_";
	char file_in[200] = ".\\reichstag\\csac_results\\ind_inliers_";
	char outliersstr[200] = ".\\reichstag\\csac_results\\ind_outliers_";
	char file_F[200] = ".\\reichstag\\csac_results\\F_";
	char file_abstraction[200] = ".\\reichstag\\csac_results\\abs_";
	char error_fn[200] = ".\\reichstag\\csac_results\\err_";
	//==============================================
	/*diagnose the size/length of matches*/
	if ((fp = fopen(fn_matches, "r")) == NULL) {
		fprintf(stderr, "error opening file %s!\n", fn_matches);
		exit(1);
	}
	n_matches = return_datanum(fp) / 2;
	if (n_matches > 65000)
	{
		printf("there are too many corresponding matches, which exceed INT_MAX.\n");
		return(0);
	}
	//	printf("n=%d ", n_matches);
	//==============================================

	//==============================================
	/*Read out matches based its length*/
	matches1 = (int *)malloc(n_matches * sizeof(int));
	matches2 = (int *)malloc(n_matches * sizeof(int));
	rewind(fp);
	n_matches = readNInts(fp, matches1, n_matches, 0);
	//////=========================for test
	//1 means the n ints are imported from software matlab, data in matches file is start from 1.
	//when they are used in C, they should substract 1
	n_matches = readNInts(fp, matches2, n_matches, 0);
	fclose(fp);
	//==============================================

	//==============================================
	/*diagnose the size/length of fframes0*/
	if ((fp = fopen(fn_frames0, "r")) == NULL) {
		fprintf(stderr, "error opening file %s!\n", fn_frames0);
		exit(1);
	}
	n_frames0 = return_datanum(fp) / 2;
	//	printf("n=%d ", n_frames0);
	//==============================================

	//==============================================
	/*Read out frames0 based its length*/
	f1x = (float *)malloc(n_frames0 * sizeof(float));
	f1y = (float *)malloc(n_frames0 * sizeof(float));
	rewind(fp);
	n_frames0 = readNfloats(fp, f1x, n_frames0, 0);
	n_frames0 = readNfloats(fp, f1y, n_frames0, 0);
	//1 means the n ints are imported from software matlab, data in matches file is start from 1.
	//when they are used in C, the fourth parameter should be set 0
	fclose(fp);
	//==============================================

	//==============================================
	/*diagnose the size/length of fframes1*/
	if ((fp = fopen(fn_frames1, "r")) == NULL) {
		fprintf(stderr, "error opening file %s!\n", fn_frames1);
		exit(1);
	}
	n_frames1 = return_datanum(fp) / 2;
	//==============================================

	//==============================================
	/*Read out frames1 based its length*/
	f2x = (float *)malloc(n_frames1 * sizeof(float));
	f2y = (float *)malloc(n_frames1 * sizeof(float));
	rewind(fp);
	n_frames1 = readNfloats(fp, f2x, n_frames1, 0);
	n_frames1 = readNfloats(fp, f2y, n_frames1, 0);
	fclose(fp);

	K1 = (double *)malloc(ndof * sizeof(double));
	K2 = (double *)malloc(ndof * sizeof(double));
	fp = fopen(fn_K1, "r");
	i = readNDoubles(fp, K1, 9, 0);
	fclose(fp);
	fp = fopen(fn_K2, "r");
	i = readNDoubles(fp, K2, 9, 0);
	fclose(fp);


	//rearrange frames data.
	//Before rearrangement, (f1x[matches1[i]], f1y[matches1[i]]) corresponds to point (f2x[matches2[i]], f2y[matches2[i]])
	//After rearrangement, (frames0x[i], frames0y[i]) corresponds to point (frames1x[i], frames1y[i])
	/*( frames1x,frames1y )=( f2x(matches(2,j)),f2y(matches(2,j)) )*/
	/*( frames0x,frames0y )=( f1x(matches(1,j)),f1y(matches(1,j)) )*/
	frames1x = (float *)malloc(n_matches * sizeof(float));
	frames1y = (float *)malloc(n_matches * sizeof(float));
	for (i = 0; i < n_matches; i++)
	{
		*(frames1x + i) = *(f2x + *(matches2 + i));
		*(frames1y + i) = *(f2y + *(matches2 + i));
	}
	frames0x = (float *)malloc(n_matches * sizeof(float));
	frames0y = (float *)malloc(n_matches * sizeof(float));
	for (i = 0; i < n_matches; i++)
	{
		*(frames0x + i) = *(f1x + *(matches1 + i));
		*(frames0y + i) = *(f1y + *(matches1 + i));
	}
	free(f1x);
	free(f1y);
	free(f2x);
	free(f2y);
	//==============================================

	//==============================================
	/*diagnose the size of image I0*/
	if ((fp = fopen(fn_I0, "r")) == NULL) {
		fprintf(stderr, "error opening file %s!\n", fn_I0);
		exit(1);
	}
	n_I0 = return_datanum(fp);
	//	printf("n=%d ", n_I0);
	rewind(fp);
	r_I0 = return_lineno(fp);
	c_I0 = n_I0 / r_I0;
	//	printf("row number ofI0=%d\n", r_I0);
	//	printf("column number ofI0=%d\n", c_I0);
	//==============================================

	//==============================================
	/*Read out I0 based its length*/
	float *I0, *I1;
	I0 = (float *)malloc(n_I0 * sizeof(float));
	rewind(fp);
	n_I0 = readNfloats(fp, I0, n_I0, 0);
	fclose(fp);
	//==============================================

	//==============================================
	/*diagnose the size of image I1*/
	if ((fp = fopen(fn_I1, "r")) == NULL) {
		fprintf(stderr, "error opening file %s!\n", fn_I1);
		exit(1);
	}
	n_I1 = return_datanum(fp);
	//	printf("n=%d ", n_I1);
	rewind(fp);
	r_I1 = return_lineno(fp);
	c_I1 = n_I1 / r_I1;
	//==============================================

	//==============================================
	/*Read out I1 based its length*/
	I1 = (float *)malloc(n_I1 * sizeof(float));
	rewind(fp);
	n_I1 = readNfloats(fp, I1, n_I1, 0);
	fclose(fp);
	//==============================================

	//==============================================
	//extract a core of matches
	float *dev_frames0x, *dev_frames1x, *dev_frames0y, *dev_frames1y;
	//dev_frames0x is used for x-coordinates of frames produced in image1. The other three variables have similar sense.
	int cn2, cc4 = N*(N - 1) / 2 * (N - 2) / 3 * (N - 3) / 4;
	if (n_matches % 2)
		cn2 = (n_matches*(n_matches - 1) / 2);
	else
		cn2 = (n_matches / 2 * (n_matches - 1));
	float *host_cn2_error, *host_cn2_error_beforesort, *dev_cn2_error;
	//	host_cn2_error and dev_cn2_error are distance error vector/list/array produced by c(n,2) matches chosen in global matches paires array.
	//before error sorting dev_cn2_error is:
	//e(match0,match1)
	//e(match0,match2)
	//...
	//e(match0,match_(n_matches))
	//e(match1,match2)
	//...
	//e(match1,match_(n_matches))
	//...
	//...
	//e(match_(n_matches-1),match_(n_matches))
	//where matchi is the i-th pair in matches array,
	//matchi(1) and matchi(2) are the coordinates index of match point in frame1 corresponding to image1.
	//e(matchi,matchj) is the intensity distribution mean error between pixes on line1 and line2,
	//line1 is produced by the ith and jth match points on image1,
	//line2 is produced by the ith and jth match points on image2.
	unsigned int *dev_cn2_global_ind, *host_cn2_global_ind;
	//dev_cn2_global_ind and host_cn2_global_ind are indexes of errors before/after sorting corresponds to host_cn2_error and *dev_cn2_error
	//when sorting error, its corresponding index are sorted correspondly.
	//before index sorting dev_cn2_global_ind is:
	//0
	//1
	//2
	//...
	//ext_cn2

	//cc4 is a combinatorial number, which equals c(n,4)
	/*matches is a 2xn array storing one match per column*/
	/*<matches(1,j),matches(2,j)> is a pair of match between image I0 and I1*/
	/*matches(1,j) corresponds to the j-th points in frame1*/
	/*matches(2,j) corresponds to the j-th points in frame2*/
	//frames0 = [X ...
	//         Y ...]
	// The coordination system is in accord with screen coordinate system, such
	//as-------------------------->
	// | O                     X
	// |
	// |
	// | Y
	//   V
	//I0 and I1 are two images normalized in[0, 1]

	//=================Initialize GPU
	cudaGetDeviceCount(&i);
	if (i == 0) {	//there is no corresponding GPU used for application.
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	struct cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);
	int blocksPerGrid = device_prop.multiProcessorCount << 1;//imin(32, (ncore + threadsPerBlock - 1) / threadsPerBlock);
	int threadsPerBlock = 256;//based the number of variables used in CUDA __global__ function (total number of registers per block supported by 1080 is 64k)
	printf("blocksPerGrid=%d\n", blocksPerGrid);
	printf("blocksPerGrid=%d\n", threadsPerBlock);

	//============================constant memory initialization.
	//Extend cn2 to 2^k for bitonic sort since original bitonic algorithm can only cope with vector with length 2^k.
	int ext_cn2, half_cn2;
	int temp_ch_I0s1[2] = { -1,1 }, temp_ch_I0s2[2] = { -1,1 }, temp_ch_I1s1[2] = { -1,1 }, temp_ch_I1s2[2] = { -1,1 },
		temp_dim_m_I_cn24[7], temp_I0interchange[2] = { 0,1 }, temp_I1interchange[2] = { 0,1 };
	temp_dim_m_I_cn24[0] = n_matches;
	temp_dim_m_I_cn24[1] = c_I0;//columns number of I0
	temp_dim_m_I_cn24[2] = c_I1;//columns number of I1
	for (i = 0; i < 40; i++)//Initialization of ext_cn2, for parallel Bitonic sort, ext_cn2 should be 2^i, and greater than cn2
	{
		ext_cn2 = pow(2.0, i);
		if (ext_cn2 >= cn2)
			break;
	}
	half_cn2 = ext_cn2 >> 1;
	temp_dim_m_I_cn24[3] = ext_cn2;
	temp_dim_m_I_cn24[4] = (int)(log(ext_cn2) / log(2));//the function of the number of matches c(n,2), temp_dim_m_I_cn24[4] = (int)(log(cn2) / log(2)) which is used in bitonic sort.
	temp_dim_m_I_cn24[5] = half_cn2;//the half of ext_cn2, used to control bitonic cycle.
	temp_dim_m_I_cn24[6] = cc4;//c(n,4)
	HANDLE_ERROR(cudaMemcpyToSymbol(dim_m_I_cn24, temp_dim_m_I_cn24, sizeof(int) * 7));
	HANDLE_ERROR(cudaMemcpyToSymbol(ch_I0s1, temp_ch_I0s1, sizeof(int) * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(ch_I0s2, temp_ch_I0s2, sizeof(int) * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(ch_I1s1, temp_ch_I1s1, sizeof(int) * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(ch_I1s2, temp_ch_I1s2, sizeof(int) * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(I0interchange, temp_I0interchange, sizeof(int) * 2));
	HANDLE_ERROR(cudaMemcpyToSymbol(I1interchange, temp_I1interchange, sizeof(int) * 2));

	//=================Initialize variable
	host_cn2_error = (float *)malloc(ext_cn2 * sizeof(float));
	host_cn2_error_beforesort = (float *)malloc(ext_cn2 * sizeof(float));
	host_cn2_global_ind = (unsigned int *)malloc(ext_cn2 * sizeof(unsigned int));
	for (i = 0; i < ext_cn2; i++)
	{
		if (i < cn2)
			*(host_cn2_error + i) = 0;
		else
			*(host_cn2_error + i) = 10000.0;
		*(host_cn2_global_ind + i) = i;
	}

	//=================capture the start time
	//dev_ij_in_ind=ind(i,j) is the global index of elements in dev_cn2_error
	//the correspondence of ind(i,j) and i, j is ind(i,j)=sum_(k=0)^(i-1)(n-k-1)+j-i-1
	//dev_ij_in_ind has two columns using int2 type,
	//in which the first column dev_ij_in_ind.x recorded i index in matches
	//and the second column dev_ij_in_ind.y recorded j index in matches.
	int2 *dev_ij_in_ind, *host_ij_in_ind;
	host_ij_in_ind = (int2 *)malloc(ext_cn2 * sizeof(int2));

	cudaEvent_t start1, stop1, start2, stop2;//time counting
	float elapsedTime1, elapsedTime2;//time counting
	float *dev_I0, *dev_I1;
	//	clock_t start, finish;
	double duration;
	_LARGE_INTEGER time_start;  //开始时间  
	_LARGE_INTEGER time_over;   //结束时间  
	double dqFreq;      //计时器频率  
	LARGE_INTEGER f;    //计时器频率  
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;
	QueryPerformanceCounter(&time_start);   //计时开始  
											//	start = clock();
	HANDLE_ERROR(cudaEventCreate(&start1));//time counting
	HANDLE_ERROR(cudaEventCreate(&stop1));//time counting
	HANDLE_ERROR(cudaEventRecord(start1, 0));//time counting
	HANDLE_ERROR(cudaMalloc((void**)&dev_frames0x, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_frames1x, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_frames0y, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_frames1y, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_I0, r_I0 * c_I0 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_I1, r_I1 * c_I1 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_cn2_error, ext_cn2 * sizeof(float)));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_frames0x, n_frames0 * sizeof(float), cudaHostAllocDefault));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_frames1x, n_frames1 * sizeof(float), cudaHostAllocDefault));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_frames0y, n_frames0 * sizeof(float), cudaHostAllocDefault));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_frames1y, n_frames1 * sizeof(float), cudaHostAllocDefault));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_I0, r_I0 * c_I0 * sizeof(float), cudaHostAllocDefault));
	//	HANDLE_ERROR(cudaHostAlloc((void**)&dev_I1, r_I1 * c_I1 * sizeof(float), cudaHostAllocDefault));  
	HANDLE_ERROR(cudaMemcpy(dev_frames0x, frames0x, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_frames1x, frames1x, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_frames0y, frames0y, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_frames1y, frames1y, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_I0, I0, r_I0 * c_I0 * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_I1, I1, r_I1 * c_I1 * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_cn2_error, host_cn2_error, ext_cn2 * sizeof(float), cudaMemcpyHostToDevice));
	//	HANDLE_ERROR(cudaMemcpy(dev_I0, I0, r_I0 * c_I0 * sizeof(float), cudaMemcpyHostToDevice));
	//	HANDLE_ERROR(cudaMemcpy(dev_I1, I1, r_I1 * c_I1 * sizeof(float), cudaMemcpyHostToDevice));

	//dev_matches is the matches resulting in image2, which will used as share memory variable.
	//dev_frames0x etc. are x and y frame coordinations resulting in image1 and 2 respectively.
	//dev_cn2_error is the resulting distance error of cn2 matches.
	//dev_ij_in_ind=ind(i,j) is the retured global index of elements in dev_cn2_error
	//the correspondence of ind(i,j) and i, j is ind(i,j)=sum_(k=0)^(i-1)(n-k-1)+j-i-1
	//dev_ij_in_ind has two columns using int2 type,
	//in which the first column dev_ij_in_ind.x recorded i index in matches
	//and the second column dev_ij_in_ind.y recorded j index in matches.
	HANDLE_ERROR(cudaMalloc((void**)&dev_ij_in_ind, ext_cn2 * sizeof(int2)));
	extract_parallel_dist_cn2 << <blocksPerGrid, threadsPerBlock >> >(dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, dev_I0, dev_I1, dev_cn2_error, dev_ij_in_ind);
	HANDLE_ERROR(cudaMemcpy(host_cn2_error_beforesort, dev_cn2_error, ext_cn2 * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(host_cn2_error, dev_cn2_error, ext_cn2 * sizeof(float), cudaMemcpyDeviceToHost));
	//copy dev_ij_in_ind to host_ij_in_ind to prepare for interception core index in global matches array using host CPU.
	HANDLE_ERROR(cudaMemcpy(host_ij_in_ind, dev_ij_in_ind, ext_cn2 * sizeof(int2), cudaMemcpyDeviceToHost));

	//	fp = fopen("error_ind.txt", "w");
	//	for (i = 0; i < ext_cn2; i++)
	//		fprintf(fp,"%f %d %d\n", host_cn2_error_beforesort[i], host_ij_in_ind[i].x, host_ij_in_ind[i].y);
	//	fclose(fp);

	HANDLE_ERROR(cudaEventRecord(stop1, 0));//time counting
	HANDLE_ERROR(cudaEventSynchronize(stop1));//time counting
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime1, start1, stop1));//time counting
	printf("Time for calculate priority using two pairs sampling in matches: %3.5f ms\n", elapsedTime1);//time counting
	HANDLE_ERROR(cudaFree(dev_ij_in_ind));
	HANDLE_ERROR(cudaFree(dev_I0));
	HANDLE_ERROR(cudaFree(dev_I1));
	HANDLE_ERROR(cudaEventDestroy(start1));
	HANDLE_ERROR(cudaEventDestroy(stop1));
	//		for (i = 0; i < ext_cn2; i++)
	//			printf("host_cn2_error: %f %f", *(host_cn2_error + ext_cn2/20), *(host_cn2_error + ext_cn2 / 10));

	//==============================================Bitonic sort dist (dev_cn2_error) to cut top cn4 elements as matches core.
	int *host_cc4_core_global_ind;
	//host_cc4_core_global_ind is intercepted c(n,4) global index corresponds to top min error e(matchi,matchj).
	host_cc4_core_global_ind = (int *)malloc(cc4 * sizeof(unsigned int));
	printf("cn2:%d, ext_cn2:%d", cn2, ext_cn2);
	//	for (i = 0; i < cc4; i++)
	//		*(host_cc4_core_global_ind + i) = 0;
	HANDLE_ERROR(cudaEventCreate(&start2));//time counting
	HANDLE_ERROR(cudaEventCreate(&stop2));//time counting
	HANDLE_ERROR(cudaEventRecord(start2, 0));//time counting

	HANDLE_ERROR(cudaMalloc((void**)&dev_cn2_global_ind, ext_cn2 * sizeof(unsigned int)));
	HANDLE_ERROR(cudaMemcpy(dev_cn2_global_ind, host_cn2_global_ind, ext_cn2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
	//Bitonic sort to extract matches core
	//dev_cn2_error are both input and output, in which output sorted distance errors are rewrited in dev_cn2_error
	//dev_cn2_global_ind is adjoint index output after sorting.
	for (i = 1; i <= temp_dim_m_I_cn24[4]; i++)//dim_m_I_cn24[4]
	{
		for (j = i; j > 0; j--) {
			Bitonic_sort_ex_mc << <blocksPerGrid, threadsPerBlock >> > (dev_cn2_error, dev_cn2_global_ind, i, j);
		}
	}
	cudaDeviceSynchronize();

	HANDLE_ERROR(cudaMemcpy(host_cn2_error, dev_cn2_error, ext_cn2 * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(host_cc4_core_global_ind, dev_cn2_global_ind, cc4 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	// get stop time, and display the timing results
	HANDLE_ERROR(cudaEventRecord(stop2, 0));//time counting
	HANDLE_ERROR(cudaEventSynchronize(stop2));//time counting
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime2, start2, stop2));//time counting
	printf("Time for Bitonic sort of two pairs sampling using CUDA after extend cn2 to 2^k ext_cn2: %3.1f ms\n", elapsedTime2);//time counting

																															   //unbind texture variables
	HANDLE_ERROR(cudaEventDestroy(start2));
	HANDLE_ERROR(cudaEventDestroy(stop2));
	printf("host_cn2_error: %f %f", *(host_cn2_error + ext_cn2 / 20), *(host_cn2_error + ext_cn2 / 10));


	//======================================================Because of error, there are always many outliers are taken as inliers.
	//In order to avoid the mistakes, we explore inliers in matches with small errors.
	//prepar for candidates of the matches for exploration.
	int *host_candi_matches_global_ind, n_candidate, *candi_matches, card_candi = 0;
	n_candidate = cc4;
	printf("cn2:%d, ext_cn2:%d, n_candidate:%d\n", cn2, ext_cn2, n_candidate);
	host_candi_matches_global_ind = (int *)malloc(n_candidate * sizeof(unsigned int));
	candi_matches = (int *)malloc(n_matches * sizeof(unsigned int));
	HANDLE_ERROR(cudaMemcpy(host_candi_matches_global_ind, dev_cn2_global_ind, n_candidate * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_cn2_error));
	HANDLE_ERROR(cudaFree(dev_cn2_global_ind));

	thrust::host_vector<int> matches_indx(n_matches, 0);
	for (i = 0; i < n_candidate; i++)
	{
		matches_indx[host_ij_in_ind[*(host_candi_matches_global_ind + i)].x] += 1;
		matches_indx[host_ij_in_ind[*(host_candi_matches_global_ind + i)].y] += 1;
	}
	for (i = 0; i < n_matches; i++)
	{
		if (matches_indx[i]>1)//default set to explore core with not less than 2 occurrences
		{
			//printf("matches_indx[i]: %d", matches_indx[i]);
			card_candi++;
			*(candi_matches + i) = 1;
			if (card_candi > n_candidate)
				break;
		}
		else
			*(candi_matches + i) = 0;

	}
	printf("card_candi: %d ", card_candi);



	//================================================Produce c(n,4) combination list
	//Preparing: In order to produce c(n,4) combination list, we first need to transform global index of core to i,j index of core in matches.
	//core_ij_indx(i)=1 indicate the ith match pair is core match;
	thrust::host_vector<int> core_ij_indx(n_matches, 0);
	for (i = 0; i < cc4; i++)
	{
		//printf("*(host_ij_in_ind+*(host_cc4_core_global_ind + i))=%d ", host_ij_in_ind [*(host_cc4_core_global_ind + i)].x);
		//host_cc4_core_global_ind is intercepted c(n,4) global index corresponds to top min error e(matchi,matchj).
		core_ij_indx[host_ij_in_ind[*(host_cc4_core_global_ind + i)].x] += 1;
		core_ij_indx[host_ij_in_ind[*(host_cc4_core_global_ind + i)].y] += 1;
	}
	//count cardinal number of core: card(core_ij_indx)
	int card_core = 0, core_upper_lim = 300, occur_thresh=1;
	for (i = 0; i < n_matches; i++)
	{
		if (core_ij_indx[i]>occur_thresh)//default set to explore core with not less than 2 occurrences
		{
			//			printf("i: %d, core_ij_indx[i]: %d\n",i, core_ij_indx[i]);
			card_core++;
			if (card_core > core_upper_lim)
				break;
		}
	}
	//if (card_core < 100)//if the core is no more than 100, explore core with 1 occurrences
	//{
		//card_core = 0;
		//temp = 1;
		//for (i = 0; i < n_matches; i++)
			//if (core_ij_indx[i])
			//{
				//card_core++;
				//if (card_core > core_upper_lim)//set a limitation to the core cardinal to fit the memory provision.
					//break;
			//}
	//}
	printf("\card_core=%d\n", card_core);

	//thrust::host_vector<int> cm_ind(card_core);
	int *cm_ind, *dev_cm_ind;
	cm_ind = (int *)malloc(card_core * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**)&dev_cm_ind, card_core * sizeof(int)));
	j = 0;
	//	fp = fopen("core_list.txt", "w");
	if (temp)
	{
		for (i = 0; i < n_matches; i++)
			if (core_ij_indx[i])
			{
				cm_ind[j++] = i;
				if (j > core_upper_lim)//according to the core limitation
					break;
			}
	}
	else//explore core with default set
		for (i = 0; i < n_matches; i++)
			if (core_ij_indx[i] > occur_thresh)
			{
				cm_ind[j++] = i;
				if (j > core_upper_lim)//according to the core limitation
					break;
			}
	HANDLE_ERROR(cudaMemcpy(dev_cm_ind, cm_ind, card_core * sizeof(int), cudaMemcpyHostToDevice));

	blocksPerGrid = 40<<1; //device_prop.multiProcessorCount << 1;
	//	threadsPerBlock = device_prop.maxThreadsPerMultiProcessor;
	threadsPerBlock = 256;//based the number of variables used in CUDA __global__ function (total number of registers per block supported by 1080 is 64k)
	cudaEvent_t start5, stop5;	//time counting
	float elapsedTime5;//time counting
	HANDLE_ERROR(cudaEventCreate(&start5));//time counting
	HANDLE_ERROR(cudaEventCreate(&stop5));//time counting
	HANDLE_ERROR(cudaEventRecord(start5, 0));//time counting

	int s_d_check = 0, n_round = 100;//100 every GPU thread run n_round to return a result to host
	double q, eps = 1e-30, temp_double;
	n_round = card_core * P_CORE;
	temp_double = card_core / n_round  * (card_core - 1) / (n_round-1) * (card_core - 2) / (n_round - 2) * (card_core - 3) / (n_round - 3) * (card_core - 4) / (n_round - 4);
	n_round = temp_double * 2 / threadsPerBlock / blocksPerGrid;
	if (n_round < 1)
		n_round = 1;
	printf("\nEvery GPU thread run %d round to return a result to host\n", n_round);
	unsigned int T_iter = 10000, max_iter = 10, iter = 0;
	int temp_count = 0, N_I_star = 0, *dev_result_ninliers, *host_result_ninliers, result_ninliers, temp_results;
	float *nrmlz_frames0x, *nrmlz_frames0y, *nrmlz_frames1x, *nrmlz_frames1y, *dev_nrmlz_frames0x, *dev_nrmlz_frames1x, *dev_nrmlz_frames0y, *dev_nrmlz_frames1y;
	nrmlz_frames0x = (float *)malloc(n_matches * sizeof(float));
	nrmlz_frames0y = (float *)malloc(n_matches * sizeof(float));
	nrmlz_frames1x = (float *)malloc(n_matches * sizeof(float));
	nrmlz_frames1y = (float *)malloc(n_matches * sizeof(float));
	calibratePoints(K1, K2, frames0x, frames0y, frames1x, frames1y, nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, n_matches);

	double inlier_thresh;
	inlier_thresh = thresh_sampson / ((K1[0] + K1[4] + K2[0] + K2[4]) / 4.0);
	inlier_thresh *= inlier_thresh;


	int *dev_result_MSS, *host_result_MSS, results_MSS[essential_fd];
	HANDLE_ERROR(cudaMalloc((void**)&dev_result_ninliers, blocksPerGrid * sizeof(int)));//GPU write results back to global memory, since every block works independently in their shared memory.
	host_result_ninliers = (int *)malloc(blocksPerGrid * sizeof(int));
	for (i = 0; i<blocksPerGrid; i++)
		host_result_ninliers[i] = 0;
	HANDLE_ERROR(cudaMemcpy(dev_result_ninliers, host_result_ninliers, blocksPerGrid * sizeof(int), cudaMemcpyHostToDevice));

	//we design a variable with the length of the number of blocks to deliver results back to host memory
	HANDLE_ERROR(cudaMalloc((void**)&dev_result_MSS, blocksPerGrid * 6 * sizeof(int)));
	host_result_MSS = (int *)malloc(blocksPerGrid * 6 * sizeof(int));

	result_ninliers = 0;//we use it to collect the results in array host_result_MSS
	for (i = 0; i < essential_fd; i++)
		results_MSS[i] = 0;
	//For a minimal sample list with a length l_MSL, draw a subset with the length threadsPerBlock * blocksPerGrid * n_round and store to variable rand_ind

	HANDLE_ERROR(cudaMalloc((void**)&dev_nrmlz_frames0x, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_nrmlz_frames1x, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_nrmlz_frames0y, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_nrmlz_frames1y, n_matches * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_nrmlz_frames0x, nrmlz_frames0x, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_nrmlz_frames1x, nrmlz_frames1x, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_nrmlz_frames0y, nrmlz_frames0y, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_nrmlz_frames1y, nrmlz_frames1y, n_matches * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_n_round, &n_round, sizeof(int)));
	for (i = 0; i < essential_fd; i++)
		s_d_check += i;

	while (iter <= T_iter && iter <= max_iter)
	{	//Input: n_matches, dev_frames0x,...
		//Output:dev_results.x is a vector with the length gridDim, in which the i-th element is the maximum number of inliers in the i-th block.
		//dev_results.y is a vector with the length gridDim. The i-th element is the index in 4 samples list which implied maximum inliers in the i-th block.
		//__global__ void RANSAC_fund(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int l_list, int s_d_check, int *cm_ind, int card_core, int *result_ninliers, int *result_MSS)

		RANSAC_fund << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) * 6 >> > (dev_nrmlz_frames0x, dev_nrmlz_frames0y, dev_nrmlz_frames1x, dev_nrmlz_frames1y, n_matches, inlier_thresh, s_d_check, dev_cm_ind, card_core, dev_result_ninliers, dev_result_MSS);
		HANDLE_ERROR(cudaMemcpy(host_result_ninliers, dev_result_ninliers, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(host_result_MSS, dev_result_MSS, blocksPerGrid * 6 * sizeof(int), cudaMemcpyDeviceToHost));

		//Find optimal results so that results.x=max(host_results.x), results.y is corresponding to host_results.y
		result_ninliers = host_result_ninliers[0];//Initialization
												  //		printf("\nhost_result_MSS[j]\n");
		for (j = 0; j < essential_fd; j++)
		{
			results_MSS[j] = host_result_MSS[j];
			//			printf("%d ", host_result_MSS[j]);
		}

		//		results.x = host_results[0].x;
		//		results.y = host_results[0].y;
		//		printf("temp_results.x=%d  temp_results.y=%d\n", result_ninliers, results_MSS[0]);

		for (i = 1; i < blocksPerGrid; i++)
		{
			if (host_result_ninliers[i] > result_ninliers)
			{
				result_ninliers = host_result_ninliers[i];
				for (j = 0; j<essential_fd; j++)
					results_MSS[j] = host_result_MSS[i*essential_fd + j];
			}
		}
		printf("current host loop: %d, find %d inliers, corresponding optimal MSS: %d %d %d  %d %d\n", iter, result_ninliers, results_MSS[0], results_MSS[1], results_MSS[2], results_MSS[3], results_MSS[4]);
		temp_count = result_ninliers;
		if (temp_count > N_I_star)
		{
			//			rcs_star = rank_cs;
			N_I_star = temp_count;

			//update the number of iterations
			q = (double)(N_I_star) / (double)(n_matches);
			temp_double = 1 - q*q*q*q*q*q*q;
			//			printf("N_I_star: %d, n_matches:%d, q:%f, temp:%f\n", N_I_star, n_matches,q, temp_double);
			q = eps > temp_double ? eps : temp_double;
			temp_double = 1 - eps < q ? 1 - eps : q;
			//			printf("q>eps:%d, %5.16f, temp: %5.16f\n", q > eps, q, temp_double);
			T_iter = (unsigned int)(log(0.01) / log(temp_double));
			//			if (T_iter < 0)
			//				printf("T_iter=%d,log(epsilon)=%f,log(1 - q)=%f,T_iter=%ld\n", T_iter, log(epsilon), log(1 - q), round(log(epsilon) / log(1 - q)));
			printf("\nupdate the RANSAC results, total iter need: %u, n_inliers: %d, MSS: %d, %d, %d, %d, %d\n", T_iter, N_I_star, results_MSS[0], results_MSS[1], results_MSS[2], results_MSS[3], results_MSS[4]);
		}
		if (temp_count == n_matches)
			break;
		//the main loop contains every thread runs n_round RANSAC exploring to return results to host memory.
		iter = iter + threadsPerBlock*blocksPerGrid*n_round;
		printf("iter: %d\n", iter);
	}
	//=======================================================================================


	HANDLE_ERROR(cudaEventRecord(stop5, 0));//time counting
	HANDLE_ERROR(cudaEventSynchronize(stop5));//time counting
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime5, start5, stop5));//time counting
	printf("Time for RANSAC against core list: %3.1f ms\n", elapsedTime5);//time counting
	printf("\nTime for GPU processing totals to: %3.1f ms\n", elapsedTime1 + elapsedTime2 + elapsedTime5);//time counting

	HANDLE_ERROR(cudaFree(dev_frames0x));
	HANDLE_ERROR(cudaFree(dev_frames0y));
	HANDLE_ERROR(cudaFree(dev_frames1x));
	HANDLE_ERROR(cudaFree(dev_frames1y));
	HANDLE_ERROR(cudaFree(dev_result_MSS));
	HANDLE_ERROR(cudaFree(dev_result_ninliers));

	//	printf("\nresults.x=%d  results.y=%d\n", results.x, results.y);
	if (result_ninliers == 0)
	{
		printf("There is no inlier explored. \n");
		exit(0);
	}

	int rank_results = 0;
	//in some case, different MSS return a same number of inliers, and we need to explore every MSS with a same maximum number of inliers
	for (i = 0; i < blocksPerGrid; i++)
		//printf("results.x=%d  results.y=%d\n", host_results[i].x, host_results[i].y);
		if (host_result_ninliers[i] == N_I_star)//result_ninliers
			rank_results++;

	int *results_array = (int*)calloc(rank_results, sizeof(int));
	j = 0;
	for (i = 0; i < blocksPerGrid; i++)
	{
		//printf("results.x=%d  results.y=%d\n", host_results[i].x, host_results[i].y);
		if (host_result_ninliers[i] == N_I_star)//result_ninliers
		{
			*(results_array + j) = i;
			j++;
		}
	}

	double F[N3][N3], F_star[N3][N3],  proj_error = 0.0;
	int *CS, *CS_temp, *CS_outlieres, *CS_outlieres_temp;
	int ranki, n_temp = 0, ninliers = 0, k = 0, l, n_temp_init, n_eliminated, non_degenerated=0;

	F[0][0] = 0.0; F[0][1] = 0.0; F[0][2] = 0.0;
	F[1][0] = 0.0; F[1][1] = 0.0; F[1][2] = 0.0;
	F[2][0] = 0.0; F[2][1] = 0.0; F[2][2] = 0.0;

	CS = (int*)calloc(n_matches, sizeof(int));
	CS_temp = (int*)calloc(n_matches, sizeof(int));
	CS_outlieres = (int*)calloc(n_matches, sizeof(int));
	CS_outlieres_temp = (int*)calloc(n_matches, sizeof(int));

	///int *hmss;//for test
	///hmss = (int*)calloc(5, sizeof(int));//for test
	///*hmss = 422; *(hmss + 1) = 535; *(hmss + 2) = 441; *(hmss + 3)=330; *(hmss + 4) = 90;//for test
	strcat(file_csac_in, filename);
	temp_count = 0;

	printf("\nrank_results: %d\n", rank_results);
	for (ranki = 0; ranki < rank_results; ranki++)
	{//host_result_MSS: point to the head of an 7 points MSS
//output:
//CS_temp: if CS_temp[i]=1, the i-th correpondence is a true correspondence
//function value: n_temp is the number of inliers evaluated with the model modeled by the MSS host_result_MSS
		///n_temp = assembleinliers(hmss, nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, host_cn2_error_beforesort, inlier_thresh, n_matches, CS_temp);
		n_temp = assembleinliers(host_result_MSS + *(results_array + ranki)*essential_fd, nrmlz_frames0x, nrmlz_frames0y, nrmlz_frames1x, nrmlz_frames1y, host_cn2_error_beforesort, inlier_thresh, n_matches, CS_temp);
		n_temp_init = n_temp;

		if(n_temp_init>temp_count)
		{
			temp_count = n_temp_init;
			fp = fopen(file_csac_in, "w");
			for (i = 0; i < n_matches; i++)
				if (*(CS_temp + i))
					fprintf(fp, "%d ", i);
			fclose(fp);
		}


		//potential outliers elimination
		////////////////////////////////////////////////////////////////////////
		for (j = 0; j < n_matches; j++)
			*(CS_outlieres_temp + j) = 0;
		for (j = 0; j < 2; j++)//two runs to eliminate outliers more thoroughly
		{//output: CS_temp that eliminated outliers
		 //CS_outlieres_temp that indicate outilers
			n_eliminated = outliers_detec(frames0x, frames0y, frames1x, frames1y, n_matches, n_temp, CS_temp, CS_outlieres_temp);
			n_temp -= n_eliminated;
		}
		//fitting a model F using the inliers in *CS_temp, where *(CS_temp+i)=1 indicates an inlier, n_temp is the total inliers
		//Output: F
		if (n_temp > 7)
		{
			reestimate(frames0x, frames0y, frames1x, frames1y, n_matches, n_temp, CS_temp, F_star);
			non_degenerated = 1;
		}
		//////////////////////////////////////////////////////////////////////

		if (n_temp > ninliers & non_degenerated)
		{
			ninliers = n_temp;
			for (j = 0; j < N3; j++)
				for (k = 0; k < N3; k++)
				{
					F[j][k] = F_star[j][k];
					printf("%f ", F[j][k]);
				}
			for (j = 0; j < n_matches; j++)
			{
				*(CS_outlieres + j) = *(CS_outlieres_temp + j);
				*(CS + j) = *(CS_temp + j);
			}
		}
	}

	printf("\nexplored the number of inliers: %d\n", ninliers);
	strcat(file_F, filename);
	fp = fopen(file_F, "w");
	for (i = 0; i < N3; i++)
	{
		for (j = 0; j < N3; j++)
		{
			printf("%5.12f ", F[i][j]);
			fprintf(fp, "%5.12f  ", F[i][j]);
		}
		printf("\n");
		fprintf(fp, "\n");

	}
	fclose(fp);

	if (non_degenerated)
	{	//estimate the projection error 
		proj_error = est_proj_error(frames0x, frames0y, frames1x, frames1y, n_matches, CS, CS_outlieres, &N_I_star, F);

		printf("proj_error: %f", proj_error);

		strcat(outliersstr, filename);
		fp = fopen(outliersstr, "w");
		for (i = 0; i < n_matches; i++)
			if (*(CS_outlieres + i))
				fprintf(fp, "%d ", i);
		fclose(fp);

		k = 0;

		strcat(file_in, filename);
		fp = fopen(file_in, "w");
		for (i = 0; i < n_matches; i++)
			if (*(CS + i))
			{
				fprintf(fp, "%d ", i);
				k++;
			}
		fclose(fp);
		printf("\nBefore recount, the number of inliers satisfying CSAC constraints: %d\n", N_I_star);
		ninliers = k;
		printf("\nFinally explored the number of inliers satisfying CSAC constraints: %d\n", ninliers);

		QueryPerformanceCounter(&time_over);    //计时结束  
		duration = 1000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
		printf("\nThe time cost against the whole demo, including GPU and CPU processing time (ms): %3.2f\n", duration);

		strcat(file_abstraction, filename);
		fp = fopen(file_abstraction, "w");
		//total number of correspondences| card_core| ninliers before n-1/n inliers model fitting | ninliers| cn2| Bitonic sort| RANSAC against core list| GPU| the whole time cost| proj_error 
		fprintf(fp, "%d\t %d\t %d\t %d\t %f\t %f\t %f\t %f\t %f\t %f\t", n_matches, card_core, temp_count, ninliers, elapsedTime1, elapsedTime2, elapsedTime5, elapsedTime1 + elapsedTime2 + elapsedTime5, duration, proj_error);
		fclose(fp);
	}
	else
	{
		strcat(file_abstraction, filename);
		fp = fopen(file_abstraction, "w");
		//total number of correspondences| card_core| ninliers before n-1/n inliers model fitting | ninliers| cn2| Bitonic sort| RANSAC against core list| GPU| the whole time cost| proj_error 
		fprintf(fp, "%d\t %d\t %d\t %d\t %f\t %f\t %f\t %f\t %f\t %f\t",n_matches, card_core, temp_count, ninliers, elapsedTime1, elapsedTime2, elapsedTime5, elapsedTime1 + elapsedTime2 + elapsedTime5, elapsedTime1 + elapsedTime2 + elapsedTime5, 0);
		fclose(fp);
	}

	free(results_array);
	free(CS);
	free(matches1);
	free(matches2);
	free(I0);
	free(I1);
	free(host_cn2_error);
	free(host_cn2_error_beforesort);
	free(host_cn2_global_ind);
	free(host_ij_in_ind);
	free(host_cc4_core_global_ind);
	free(host_candi_matches_global_ind);
	free(candi_matches);
	free(cm_ind);
	free(nrmlz_frames0x);
	free(nrmlz_frames0y);
	free(nrmlz_frames1x);
	free(nrmlz_frames1y);
	free(host_result_ninliers);
	free(host_result_MSS);

}
