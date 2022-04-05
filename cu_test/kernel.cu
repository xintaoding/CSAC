#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <math.h>

#include "book.h"
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
#define isnp 8//size parameter in homography
#define nmss 4
#define P_CORE 0.2
#define N3 3//size for 3*3 matrix
#define T_noise_squared 8
//||x-Hx'||+||inv(H)*x-x'||<T_noise_squared,8 23.512742444991080
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
static int readNDoubles(FILE *fp, float *vals, int nvals, int matlab_flag)
{
	register int i;
	int n, j;

	for (i = n = 0; i<nvals; ++i) {
		j = fscanf(fp, "%ef", vals + i);
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
					int frag, step_count = 0;

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

 //CUDA produce c(n,4) combination list
 //prod_comb_list(int *c, int N, int cc4)
__global__ void prod_comb_list(int4 *dev_core_comb_list, int *cm_ind, int card_core)
{
	unsigned int k, l, temp;
	if (threadIdx.x < card_core - 2 - blockIdx.x)//block blockIdx.x in 0,...,card_core-4, thread threadIdx.x in 1,...,card_core-3
	{
		temp = 0;
		for (l = 1; l <= blockIdx.x; l++)
			for (k = 0; k <= card_core - l - 3; k++)
				temp += (card_core - l - 1 - k)*(card_core - l - 2 - k) >> 1;

		if (threadIdx.x >0)
		{
			for (k = 0; k <= threadIdx.x - 1; k++)
				temp += ((card_core - blockIdx.x - 2 - k)*(card_core - blockIdx.x - 3 - k)) >> 1;
		}
		//A classical combination production may be emplemented as follows serially using for loop.
		//		for (i = 0; i<N; i++)
		//			for (j = i + 1; j<N; j++)
		//				for (k = j + 1; k<N; k++)
		//					for (l = k + 1; l<N; l++)
		//					{
		//						*(a + t * 4) = i;
		//						*(a + t * 4 + 1) = j;
		//						*(a + t * 4 + 2) = k;
		//						*(a + t * 4 + 3) = l;
		//						t++;
		//					}
		for (k = threadIdx.x + blockIdx.x + 2; k < card_core - 1; k++)//loop variable k is used to arrange the third bit combination element, which need to greater than the second bit, i.e., >=blockIdx.x+2
			for (l = k + 1; l < card_core; l++)
			{
				dev_core_comb_list[temp].x = cm_ind[blockIdx.x];//blockIdx.x is employed in the first bit
				dev_core_comb_list[temp].y = cm_ind[threadIdx.x + blockIdx.x + 1];//the second bit is arranged combination element greater than the first bit, i.e., >=blockIdx.x+1
				dev_core_comb_list[temp].z = cm_ind[k];//the third bit is arranged by loop variable k
				dev_core_comb_list[temp++].w = cm_ind[l];
			}
	}
}

//Before ransac implementation, we need to filter out redundant case of combination, such as coalinear, illogical distribution.
//The illogical distribution corresponds to points spacial distribution. The following  graph an illigal distribution.
//  Image1:      2  *                                   Image2:           2 o
//        4  *    /                                              4 o       /
//               /                                                        /     
//              /     3  *                                     3 o       /        
//             /                                                        /     
//          1 *                                                    1 o /
__global__ void comb_list_filter(int4 *dev_core_comb_list, unsigned int c_card_core_4, float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int *dev_comb_ele_01judge)
{
	//flag = validateMSS_homography(X, s)
	// Check if the points are in pathological configurations. Compute the
	// covariance matrix and see if the determinant is too small(which implies
	// the point are collinear)
	int4 X;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < c_card_core_4)
	{
		X = dev_core_comb_list[tid];//[X.x X.y X.z X.w] is a case of c(n,4) combination.
									//filter common points in Image1
		if (dev_frames0x[X.x] == dev_frames0x[X.y] && dev_frames0y[X.x] == dev_frames0y[X.y])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames0x[X.x] == dev_frames0x[X.z] && dev_frames0y[X.x] == dev_frames0y[X.z])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames0x[X.x] == dev_frames0x[X.w] && dev_frames0y[X.x] == dev_frames0y[X.w])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames0x[X.y] == dev_frames0x[X.z] && dev_frames0y[X.y] == dev_frames0y[X.z])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames0x[X.y] == dev_frames0x[X.w] && dev_frames0y[X.y] == dev_frames0y[X.w])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames0x[X.z] == dev_frames0x[X.w] && dev_frames0y[X.z] == dev_frames0y[X.w])
			dev_comb_ele_01judge[tid] = 0;
		//filter common points in Image2
		if (dev_frames1x[X.x] == dev_frames1x[X.y] && dev_frames1y[X.x] == dev_frames1y[X.y])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames1x[X.x] == dev_frames1x[X.z] && dev_frames1y[X.x] == dev_frames1y[X.z])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames1x[X.x] == dev_frames1x[X.w] && dev_frames1y[X.x] == dev_frames1y[X.w])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames1x[X.y] == dev_frames1x[X.z] && dev_frames1y[X.y] == dev_frames1y[X.z])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames1x[X.y] == dev_frames1x[X.w] && dev_frames1y[X.y] == dev_frames1y[X.w])
			dev_comb_ele_01judge[tid] = 0;
		if (dev_frames1x[X.z] == dev_frames1x[X.w] && dev_frames1y[X.z] == dev_frames1y[X.w])
			dev_comb_ele_01judge[tid] = 0;

		//===============coalinear filter
		//coalinear filter corresponds to image1
		double mux = (double)(dev_frames0x[X.x] + dev_frames0x[X.y] + dev_frames0x[X.z] + dev_frames0x[X.w]) / 4.0;
		double muy = (double)(dev_frames0y[X.x] + dev_frames0y[X.y] + dev_frames0y[X.z] + dev_frames0y[X.w]) / 4.0;
		double C1 = (double)(dev_frames0x[X.x] - mux)*(dev_frames0x[X.x] - mux) + (dev_frames0x[X.y] - mux)*(dev_frames0x[X.y] - mux) + (dev_frames0x[X.z] - mux)*(dev_frames0x[X.z] - mux) + (dev_frames0x[X.w] - mux)*(dev_frames0x[X.w] - mux);
		double C2 = (double)(dev_frames0x[X.x] - mux)*(dev_frames0y[X.x] - muy) + (dev_frames0x[X.y] - mux)*(dev_frames0y[X.y] - muy) + (dev_frames0x[X.z] - mux)*(dev_frames0y[X.z] - muy) + (dev_frames0x[X.w] - mux)*(dev_frames0y[X.w] - muy);
		double C3 = (double)(dev_frames0y[X.x] - muy)*(dev_frames0y[X.x] - muy) + (dev_frames0y[X.y] - muy)*(dev_frames0y[X.y] - muy) + (dev_frames0y[X.z] - muy)*(dev_frames0y[X.z] - muy) + (dev_frames0y[X.w] - muy)*(dev_frames0y[X.w] - muy);
		double kappa, delta = (double)sqrt((C1 + C3)*(C1 + C3) + 4.0 * C2 * C2 - 4.0 * C1 * C3);
		kappa = fabs((C1 + C3 + delta) / (C1 + C3 - delta + 0.00000000001));
		if (kappa > 1e6)
		{
			//			if (tid == 46552091)
			//			printf("case0: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		//coalinear filter corresponds to image2
		mux = (dev_frames1x[X.x] + dev_frames1x[X.y] + dev_frames1x[X.z] + dev_frames1x[X.w]) / 4.0;
		muy = (dev_frames1y[X.x] + dev_frames1y[X.y] + dev_frames1y[X.z] + dev_frames1y[X.w]) / 4.0;
		C1 = (dev_frames1x[X.x] - mux)*(dev_frames1x[X.x] - mux) + (dev_frames1x[X.y] - mux)*(dev_frames1x[X.y] - mux) + (dev_frames1x[X.z] - mux)*(dev_frames1x[X.z] - mux) + (dev_frames1x[X.w] - mux)*(dev_frames1x[X.w] - mux);
		C2 = (dev_frames1x[X.x] - mux)*(dev_frames1y[X.x] - muy) + (dev_frames1x[X.y] - mux)*(dev_frames1y[X.y] - muy) + (dev_frames1x[X.z] - mux)*(dev_frames1y[X.z] - muy) + (dev_frames1x[X.w] - mux)*(dev_frames1y[X.w] - muy);
		C3 = (dev_frames1y[X.x] - muy)*(dev_frames1y[X.x] - muy) + (dev_frames1y[X.y] - muy)*(dev_frames1y[X.y] - muy) + (dev_frames1y[X.z] - muy)*(dev_frames1y[X.z] - muy) + (dev_frames1y[X.w] - muy)*(dev_frames1y[X.w] - muy);
		delta = sqrt((C1 + C3)*(C1 + C3) + 4.0 * C2*C2 - 4.0 * C1*C3);
		kappa = fabs((C1 + C3 + delta) / (C1 + C3 - delta + 0.00000000001)); //0.000000000001
																			 //		if (tid == 46552091)
																			 //			printf("\ncase1: tid=%d, kappa =%f, C1=%f,  C3=%f, delta=%f\n", tid, kappa, C1, C3, delta);

		if (kappa >1e6)//1000000000.0
		{
			//			if (tid == 46552091)
			//			printf("case1: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		//=============================illogical distribution filter
		//Start from point 1 there are three cases need to check: (1,2), (1,3), (1,4).
		//If points 1 and 2 are linked, 
		//we need to check the distribution of points 3 and 4  on image1 and image2 respectively, to check
		//whether they are distributed on the same sides or two different sides of the line produced by points 1 and 2.
		//Of course, we also need start from point 2, 3 and 4 to check the illogicality. However, the check is a dual question
		//If we link points 1 and 2 to check the side homogeneity between points 3 and 4, it is the same check between points 1 and 2 which links points 3 and 4.
		//All the undirected link are listed as follows: (1,2) (1,3) (1,4) (2,3) (2,4) (3,4).
		//Since check(link(1,2))=check(link(3,4));check(link(1,3))=check(link(2,4));check(link(1,4))=check(link(2,3)),
		//we only need to check three link cases: (1,2), (1,3), (1,4).
		//	int ind[3][3] = { 2 3 4; 3 2 2; 4 4 3 };
		//link case of (1,2)
		muy = dev_frames0y[X.y] - dev_frames0y[X.x];//image1
		mux = dev_frames0x[X.x] - dev_frames0x[X.y];
		C1 = muy*(dev_frames0x[X.z] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.z] - dev_frames0y[X.x]);
		C2 = muy*(dev_frames0x[X.w] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.w] - dev_frames0y[X.x]);
		muy = dev_frames1y[X.y] - dev_frames1y[X.x];//image2
		mux = dev_frames1x[X.x] - dev_frames1x[X.y];
		kappa = muy*(dev_frames1x[X.z] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.z] - dev_frames1y[X.x]);
		delta = muy*(dev_frames1x[X.w] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.w] - dev_frames1y[X.x]);
		if (C1 * kappa <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case2: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		if (C2 * delta <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case3: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		//link case of (1,3)
		muy = dev_frames0y[X.z] - dev_frames0y[X.x];//image1
		mux = dev_frames0x[X.x] - dev_frames0x[X.z];
		C1 = muy*(dev_frames0x[X.y] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.y] - dev_frames0y[X.x]);
		C2 = muy*(dev_frames0x[X.w] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.w] - dev_frames0y[X.x]);
		muy = dev_frames1y[X.z] - dev_frames1y[X.x];//image2
		mux = dev_frames1x[X.x] - dev_frames1x[X.z];
		kappa = muy*(dev_frames1x[X.y] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.y] - dev_frames1y[X.x]);
		delta = muy*(dev_frames1x[X.w] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.w] - dev_frames1y[X.x]);
		if (C1 * kappa <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case4: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		if (C2 * delta <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case5: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f,  C2 =%f, delta=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w], C2, delta);
			dev_comb_ele_01judge[tid] = 0;
		}
		//link case of (1,4)
		muy = dev_frames0y[X.w] - dev_frames0y[X.x];//image1
		mux = dev_frames0x[X.x] - dev_frames0x[X.w];
		C1 = muy*(dev_frames0x[X.y] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.y] - dev_frames0y[X.x]);
		C2 = muy*(dev_frames0x[X.z] - dev_frames0x[X.x]) + mux*(dev_frames0y[X.z] - dev_frames0y[X.x]);
		muy = dev_frames1y[X.w] - dev_frames1y[X.x];//image2
		mux = dev_frames1x[X.x] - dev_frames1x[X.w];
		kappa = muy*(dev_frames1x[X.y] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.y] - dev_frames1y[X.x]);
		delta = muy*(dev_frames1x[X.z] - dev_frames1x[X.x]) + mux*(dev_frames1y[X.z] - dev_frames1y[X.x]);
		if (C1 * kappa <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case6: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		if (C2 * delta <= 0)
		{
			//			if (tid == 46552091)
			//				printf("case7: tid=%d,dev_frames0x[X.x]=%f dev_frames0x[X.y]=%f dev_frames0x[X.z]=%f dev_frames0x[X.w]=%f, dev_frames0y[X.x]=%f dev_frames0y[X.y]=%f dev_frames0y[X.z]=%f dev_frames0y[X.w]=%f, dev_frames1x[X.x]=%f dev_frames1x[X.y]=%f dev_frames1x[X.z]=%f dev_frames1x[X.w]=%f, dev_frames1y[X.x]=%f dev_frames1y[X.y]=%f dev_frames1y[X.z]=%f dev_frames1y[X.w]=%f\n", tid, dev_frames0x[X.x], dev_frames0x[X.y], dev_frames0x[X.z], dev_frames0x[X.w], dev_frames0y[X.x], dev_frames0y[X.y], dev_frames0y[X.z], dev_frames0y[X.w], dev_frames1x[X.x], dev_frames1x[X.y], dev_frames1x[X.z], dev_frames1x[X.w], dev_frames1y[X.x], dev_frames1y[X.y], dev_frames1y[X.z], dev_frames1y[X.w]);
			dev_comb_ele_01judge[tid] = 0;
		}
		tid += gridDim.x * blockDim.x;
	}
}

//dev_comb_ele_01judge indicates refinement rule which filters away redundant using true or false indication 0 or 1.
//judge_list_prefixsum mainly used for parallel prefix sum.
//Let dev_comb_ele_01judge be: j0 ... j31  j32 ... j63 j64 ... j96 ......
//partial_s(0)=sum(j0 ... j31),
//partial_s(1)=sum(j32 ... j63),
//partial_s(i)=sum(j(32*i) ... j(32*(i+1))-1)
__global__ void judge_list_prefixsum(int *dev_comb_ele_01judge, int c_card_core_4, int *partial_s, int l_partial_s)
{
	unsigned int i, s, tid = blockIdx.x * blockDim.x + threadIdx.x, temp0, temp;

	while (tid < l_partial_s)
	{
		s = 0;
		temp0 = 32 * tid;
		for (i = 0; i < 32; i++)
		{
			temp = temp0 + i;
			if (temp<c_card_core_4)
				s += *(dev_comb_ele_01judge + temp);
		}
		partial_s[tid] = s;
		//		printf("tid=%d, %d  ", tid, partial_s[tid]);
		tid += gridDim.x * blockDim.x;
	}
	//	__syncthreads();//syncthrize in block
	//	__threadfence();//syncthrize in grid
}

//dev_core_comb_list is the list of core candidates:                           c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 
//dev_comb_ele_01judge indicates refinement rule which filters away redundant: 0  1  1  0  0  0  1  1  1  1   1   1   1   0   0   1   0   1   0
//dev_filtered_4samples_list is real core tested in RANSAC:                    c2 c3 c7 c8 c9 c10 c11 c12 c13 c16 c18
//s_prefixsum is a helpful vector used to indicate the begining of reserving index of all parallel thread when reduce:s_prefixsum(i)=sum(dev_comb_ele_01judge(0:32*(i+1)-1)
__global__ void comb_list_reduce(int4 *dev_core_comb_list, int *dev_comb_ele_01judge, int c_card_core_4, int *s_prefixsum, int l_s_prefixsum, int4 *dev_filtered_4samples_list)
{
	unsigned int i, k, temp0, temp, tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < l_s_prefixsum)
	{
		k = 0;
		temp0 = 32 * tid;
		for (i = 0; i < 32; i++)
		{
			temp = temp0 + i;
			if (temp<c_card_core_4 && *(dev_comb_ele_01judge + temp))
			{
				if (tid == 0)
				{
					dev_filtered_4samples_list[k] = dev_core_comb_list[32 * tid + i];
					//					printf("tid=%d, %d, %d %d %d    ", tid, k, dev_filtered_4samples_list[k].x, dev_filtered_4samples_list[k].y, dev_filtered_4samples_list[k].z, dev_filtered_4samples_list[k].w);
				}
				else
				{
					dev_filtered_4samples_list[s_prefixsum[tid - 1] + k] = dev_core_comb_list[32 * tid + i];
					//					printf("tid=%d, %d, %d %d %d    ", tid, k, dev_filtered_4samples_list[s_prefixsum[tid - 1] + k].x, dev_filtered_4samples_list[s_prefixsum[tid - 1] + k].y, dev_filtered_4samples_list[s_prefixsum[tid - 1] + k].z, dev_filtered_4samples_list[s_prefixsum[tid - 1] + k].w);
				}
				k++;
			}
		}
		tid += gridDim.x * blockDim.x;
	}
}

//normalize n data of (dev_framesx[X[i]],dev_framesx[X[i]]), (i=1,...,n ) and returned in array X1.
__host__ __device__ void normal_samples(int *X, int n, float *dev_framesx, float *dev_framesy, double T1[N3][N3], double2 *X1)
{
	int i, j;
	double muI[2] = { 0,0 };
	double rho1 = 0.0;
	for (i = 0; i < n; i++)
		muI[0] += dev_framesx[X[i]];
	muI[0] /= double(n);
	for (i = 0; i < n; i++)
		muI[1] += dev_framesy[X[i]];
	muI[1] /= double(n);
	for (i = 0; i<n; i++)
		X1[i].x = dev_framesx[X[i]] - muI[0];// center x-coordinations of the points
	for (i = 0; i<n; i++)
		X1[i].y = dev_framesy[X[i]] - muI[1];// center y-coordinations of the points
	for (i = 0; i<n; i++)
		rho1 += sqrt(X1[i].x * X1[i].x + X1[i].y * X1[i].y);
	rho1 /= n;
	rho1 = 1.414213562373095 / rho1;//scale factor
	for (j = 0; j < n; j++)
	{
		X1[j].x = rho1*X1[j].x;
		X1[j].y = rho1*X1[j].y;
	}
	T1[0][0] = rho1;//Row dominant
	T1[0][1] = 0.0;
	T1[0][2] = -rho1*muI[0];// muIx
	T1[1][0] = 0.0;
	T1[1][1] = rho1;
	T1[1][2] = -rho1*muI[1];// muIy
	T1[2][0] = 0.0;
	T1[2][1] = 0.0;
	T1[2][2] = 1.0;
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
//A is input and output of U
//diag is the output of S matrix
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
//A=U*S*V', m and n are the row and column dimensions of A respectively.
//Since A is a thin matrix, the returned U, which is rewrited in A, is a thin type. u9 is the supplement of U so that [A u9] is a square unitary matrix.
__host__ __device__ void homography(double *A, int m, int n, double *u9)
{
	int i, j;
	double diag[isnp], v[isnp*isnp];
	double rv1[isnp];

	i = svd(A, diag, v, rv1, m, n);//implement singular value decomposition with thin type.

								   //===================================Build left full unitary matrix from thin decomposition.
	for (i = 0; i < ndof; i++)
		u9[i] = 1.0;//Firstly, initialize it to 1.
	double temp = 0.0;
	for (j = 0; j < isnp; j++)//for every orthgonal vector, let u(:,9) substract their linear combinations using Schmidt orthogonalization.
	{
		temp = 0.0;
		for (i = 0; i < ndof; i++)//compute dot production of between alpha(u(:,9)) and beta(j)( u(:,j) ), j=1,2,...,8.
			temp += u9[i] * *(A + i*n + j);
		for (i = 0; i < ndof; i++)//implement substraction among every element.
			u9[i] -= temp* *(A + i*n + j);
	}

	temp = 0.0;
	for (i = 0; i < ndof; i++)//normalize u(:,9) to calculate 2-norm
		temp += u9[i] * u9[i];
	for (i = 0; i < ndof; i++)//implement normalization among all elements of alpha(u(:,9))
		u9[i] = u9[i] / sqrt(temp);

	//====================================================Output
	//	printf("\n");
	//	printf("Calculated full U(%d,%d) using Schmidt orthogonalization: \n", ndof, ndof);
	//	for (i = 0; i < ndof; i++)//A returned U m*n matrix of svd decomposition
	//	{
	//		for (j = 0; j < isnp; j++)
	//			printf("%f ", *(A + i*n + j));
	//	printf("%f ", u9[i]);
	//		printf("\n");
	//	}
	//	printf("Resulting diagnal matrix: \n");
	//	for (i = 0; i<isnp; i++)
	//	{
	//		printf("%d %f\n", i, diag[i]);
	//	}
	//	printf("\n");
	//	printf("Resulting v matrix, where A=USV': \n");
	//	for (i = 0; i<isnp; i++)
	//	{
	//		for (j = 0; j<isnp; j++)
	//			printf(" %f ", *(v+i*isnp + j));
	//		printf("\n");
	//	}
	//	printf("\n");
	//=================================================================
}
__device__ int dev_get_consensus_set(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, double H[N3][N3], double detH)
{
	int k, i, inlier_count = 0;
	double temp[N3], e1, e2;
	double2 Xd;
	for (k = 0; k < dim_m_I_cn24[0]; k++)
	{
		for (i = 0; i < N3; i++)//map frames0 in I0 into I1.
			temp[i] = H[i][0] * dev_frames0x[k] + H[i][1] * dev_frames0y[k] + H[i][2];
		Xd.x = temp[0] / temp[2];
		Xd.y = temp[1] / temp[2];
		//calculate the error between mapped frames0 in I1 and frames1 in I1.
		e1 = (dev_frames1x[k] - Xd.x)*(dev_frames1x[k] - Xd.x) + (dev_frames1y[k] - Xd.y)*(dev_frames1y[k] - Xd.y);
		//inv(H):H=[a b c;
		//		d e f;
		//		g h i];
		//		//1/det(H)*
		//		[ei-hf -bi+hc bf-ce; 		H[1][1]*H[2][2]-H[2][1]*H[1][2]   -H[0][1]*H[2][2]+H[2][1]*H[0][2]  H[0][1]*H[1][2]-H[0][2]*H[1][1]; 
		//		fg-id -cg+ia cd-af; 		H[1][2]*H[2][0]-H[2][2]*H[1][0]   -H[0][2]*H[2][0]+H[2][2]*H[0][0]  H[0][2]*H[1][0]-H[0][0]*H[1][2];
		//		dh-ge -ah+gb ae-bd]		H[1][0] * H[2][1] - H[2][0] * H[1][1] - H[0][0] * H[2][1] + H[2][0] * H[0][1]  H[0][0] * H[1][1] - H[0][1] * H[1][0]
		//map frames0 in I1 into I0.
		temp[0] = ((H[1][1] * H[2][2] - H[2][1] * H[1][2])* dev_frames1x[k] + (H[2][1] * H[0][2] - H[0][1] * H[2][2])* dev_frames1y[k] + H[0][1] * H[1][2] - H[0][2] * H[1][1]) / detH;
		temp[1] = ((H[1][2] * H[2][0] - H[2][2] * H[1][0])* dev_frames1x[k] + (H[2][2] * H[0][0] - H[0][2] * H[2][0])* dev_frames1y[k] + H[0][2] * H[1][0] - H[0][0] * H[1][2]) / detH;
		temp[2] = ((H[1][0] * H[2][1] - H[2][0] * H[1][1])* dev_frames1x[k] + (H[2][0] * H[0][1] - H[0][0] * H[2][1])* dev_frames1y[k] + H[0][0] * H[1][1] - H[0][1] * H[1][0]) / detH;
		Xd.x = temp[0] / temp[2];
		Xd.y = temp[1] / temp[2];
		//calculate the error between mapped frames0 in I0 and frames0 in I0.
		e2 = (dev_frames0x[k] - Xd.x)*(dev_frames0x[k] - Xd.x) + (dev_frames0y[k] - Xd.y)*(dev_frames0y[k] - Xd.y);
		if (e1 + e2 < T_noise_squared)
			inlier_count++;
	}
	return(inlier_count);
}
__host__ __device__ void initial_A(double *A, double2 *X1, double2 *X2, int n)
{
	int i, j;
	j = 0;
	for (i = 0; i < n; i++)
	{
		//In order to use thin SVD decompostion method bidiagnal decomposition, input matrix A in size of 8*9 is transformed into 9*8 in this application.
		*(A + j) = X1[i].x; //A[0][j] = X1[0][i]
		*(A + isnp + j) = 0;//A[1][j]
		*(A + 2 * isnp + j) = -X1[i].x * X2[i].x;//A[2][j] = -X1[0][i] * X2[0][i];
		*(A + 3 * isnp + j) = X1[i].y;//A[3][j] = X1[1][i];
		*(A + 4 * isnp + j) = 0;//A[4][j]
		*(A + 5 * isnp + j) = -X1[i].y * X2[i].x;//A[5][j] = -X1[1][i] * X2[0][i];
		*(A + 6 * isnp + j) = 1.0;//A[6][j]
		*(A + 7 * isnp + j) = 0.0;//A[7][j]
		*(A + 8 * isnp + j) = -X2[i].x;//A[8][j] = -X2[0][i];

		*(A + j + 1) = 0.0;//A[0][j + 1]
		*(A + isnp + j + 1) = X1[i].x;//A[1][j + 1] = X1[0][i];
		*(A + 2 * isnp + j + 1) = -X1[i].x * X2[i].y;//A[2][j + 1]  = -X1[0][i] * X2[1][i];
		*(A + 3 * isnp + j + 1) = 0.0;//A[3][j + 1]
		*(A + 4 * isnp + j + 1) = X1[i].y;//A[4][j + 1]  = X1[1][i];
		*(A + 5 * isnp + j + 1) = -X1[i].y * X2[i].y;//A[5][j + 1] = -X1[1][i] * X2[1][i];
		*(A + 6 * isnp + j + 1) = 0.0;//A[6][j + 1] 
		*(A + 7 * isnp + j + 1) = 1.0;//A[7][j + 1] 
		*(A + 8 * isnp + j + 1) = -X2[i].y;//A[8][j + 1] = -X2[1][i];
		j += 2;
	}

}

//de-normalize Homography
__host__ __device__ void denorm_H(double T2[N3][N3], double H[N3][N3], double T1[N3][N3])
{
	int i, j;
	double temp[N3][N3];
	T2[0][2] = -T2[0][2] / T2[0][0];//inv(T2)
	T2[1][2] = -T2[1][2] / T2[1][1];
	T2[0][0] = 1.0 / T2[0][0];
	T2[1][1] = 1.0 / T2[1][1];
	//		T2[0][1] = 0.0;
	T2[2][0] = 0.0;
	//		T2[1][0] = 0.0;
	T2[2][1] = 0.0;
	//		T2[2][2] = 1.0;
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			temp[i][j] = T2[i][0] * H[0][j] + T2[i][1] * H[1][j] + T2[i][2] * H[2][j];
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			H[i][j] = temp[i][0] * T1[0][j] + temp[i][1] * T1[1][j] + temp[i][2] * T1[2][j];
	temp[0][0] = 0;
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			temp[0][0] += H[i][j] * H[i][j];
	temp[0][0] = sqrt(temp[0][0]);
	for (i = 0; i < N3; i++)
		for (j = 0; j < N3; j++)
			H[i][j] = H[i][j] / temp[0][0];
}

//RANSAC implementation
//Input:dev_filtered_4samples_list,s_prefixsum,dev_frames0x,...
//dev_filtered_4samples_list[i] is a 4-sample minimum sampling with its length l_list,
//e.g.,dev_filtered_4samples_list[i] =[dev_filtered_4samples_list[i].x,dev_filtered_4samples_list[i].y,dev_filtered_4samples_list[i].z,dev_filtered_4samples_list[i].w]
//which denoted as X=[X[0],X[1],X[2],X[3]]
//(dev_frames0x[X[0]],dev_frames0y[X[0]]) is a sample (x,y)-coordinate in image1, (dev_frames1x[X[0]],dev_frames1y[X[0]]) is a sample (x,y)-coordinate in image2. They are a match pair.
//In this way, dev_filtered_4samples_list[i] indicates 4-sample pairs.
//Output:dev_results.x is a vector with the length gridDim, in which the i-th element is the maximum number of inliers in the i-th block.
//dev_results.y is a vector with the length gridDim. The i-th element is the index in 4 samples list which implied maximum inliers in the i-th block.
__global__ void RANSAC_greedy(int4 *dev_filtered_4samples_list, int l_list, float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int2 *results)
{
	int X[4], i, j, temp_count = 0;
	double T1[N3][N3], T2[N3][N3];
	double2 X1[4], X2[4];
	double detH;
	double A[ndof*isnp], u9[ndof];// u9 is the supplement of U so that[A u9] is a square unitary matrix after SVD decomposition.

	extern __shared__ int inlier_count_Io4si[];//inlier_count_Io4si.x is the number of inliers produced by every 4-samples in dev_filtered_4samples_list
											   //inlier_count_Io4si.y is the interest of 4-samples index in dev_filtered_4samples_list.
	unsigned int tid = threadIdx.x;
	unsigned int ind_count = tid;
	inlier_count_Io4si[ind_count] = 0;
	inlier_count_Io4si[ind_count + blockDim.x] = 0;
	//	printf("blockIdx.x=%d, tid=%d  inlier_count_Io4si=%d  inlier_count_Io4si[%d]=%d \n ", blockIdx.x, threadIdx.x, inlier_count_Io4si[ind_count], ind_count + blockDim.x, inlier_count_Io4si[ind_count + blockDim.x]);
	__syncthreads();//syncthrize in block
					//	__threadfence();//syncthrize in grid
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	results[0].x = 0;
	//	int Nloop = MIN(l_list / 2, blockDim.x*gridDim.x << 2);
	//	while (tid < l_list&&results[0].x==0)//used for sudden death=========
	while (tid < l_list)
	{
		X[0] = dev_filtered_4samples_list[tid].x;//X is a case of c(n,4) combination.
		X[1] = dev_filtered_4samples_list[tid].y;
		X[2] = dev_filtered_4samples_list[tid].z;
		X[3] = dev_filtered_4samples_list[tid].w;
		//printf("X=%d %d %d %d\n", X[0], X[1], X[2], X[3]);
		//normalization samples.
		//Input: X, dev_frames0x, dev_frames0y; Output: T1, X1;
		normal_samples(X, 4, dev_frames0x, dev_frames0y, T1, X1);
		normal_samples(X, 4, dev_frames1x, dev_frames1y, T2, X2);
		//Initialize Zuliani formulation of homogrphy transformation.
		initial_A(A, X1, X2, 4);
		//Homography between 4-samples after using svd decomposition of A.
		homography(A, ndof, isnp, u9);//V is returned in A, where A=USV'. u9 is the returned homography matrix in vector style.
		double H[N3][N3];
		//H=[a(0,isnp), a(3,isnp), a(6,isnp)
		//   a(1,isnp), a(4,isnp), a(7,isnp)
		//   a(2,isnp), a(5,isnp), a(8,isnp)]
		for (i = 0; i < N3; i++)
			H[i][0] = u9[i];
		for (i = 0; i < N3; i++)
			H[i][1] = u9[N3 + i];
		for (i = 0; i < N3; i++)
			H[i][2] = u9[(N3 << 1) + i];

		//de-normalize Homography
		denorm_H(T2, H, T1);

		detH = H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - H[2][0] * H[1][1] * H[0][2] - H[2][1] * H[1][2] * H[0][0] - H[2][2] * H[1][0] * H[0][1];
		if (detH > 1.0e14 || detH < -1.0e14 || (detH < 0.00000000000001 && detH > -0.00000000000001))
		{
			for (i = 0; i < N3; i++)
				for (j = 0; j < N3; j++)
					H[i][j] = 0.0;
			detH = 0.0;
			temp_count = 0;
		}
		if (detH)//temp_count is the returned number of consensus points.
			temp_count = dev_get_consensus_set(dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, H, detH);
		if (temp_count > inlier_count_Io4si[ind_count])
		{
			inlier_count_Io4si[ind_count] = temp_count;
			inlier_count_Io4si[ind_count + blockDim.x] = tid;
			//			printf("tid=%d, inlier_count_Io4si[%d]=%d, inlier_count_Io4si[%d]=%d\n", tid, ind_count, inlier_count_Io4si[ind_count], ind_count + blockDim.x, inlier_count_Io4si[ind_count + blockDim.x]);
			//			if (temp_count > 0.1*dim_m_I_cn24[0])//used for sudden death===========
			//				atomicExch(&results[0].x, 1);//used for sudden death===============
		}
		tid += gridDim.x * blockDim.x;
	}
	__syncthreads();//syncthrize in block
	if (threadIdx.x == 0)//0th thread is used to explore local optimal results in its block.
	{
		for (i = 1; i < blockDim.x; i++)
			if (inlier_count_Io4si[i] > inlier_count_Io4si[0])
			{
				inlier_count_Io4si[0] = inlier_count_Io4si[i];
				inlier_count_Io4si[blockDim.x] = inlier_count_Io4si[i + blockDim.x];
			}
		results[blockIdx.x].x = inlier_count_Io4si[0];
		results[blockIdx.x].y = inlier_count_Io4si[blockDim.x];
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
		//				printf("threadIdx.x:%d, %d, %d\n", threadIdx.x, mask_num, s_d_check);

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


__global__ void RANSAC(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int l_list, int s_d_check, int *cm_ind, int card_core, int *result_ninliers, int *result_MSS)
{
	int X[4], hit_ind[nmss], i, j, temp_count = 0, k = 0;
	double T1[N3][N3], T2[N3][N3];
	double2 X1[nmss], X2[nmss];
	double detH;
	double A[ndof*isnp], u9[ndof];// u9 is the supplement of U so that[A u9] is a square unitary matrix after SVD decomposition.

	extern __shared__ int inlier_count_4points[];//inlier_count_Io4si.x is the number of inliers produced by every 4-samples in dev_filtered_4samples_list
											   //inlier_count_Io4si.y is the interest of 4-samples index in dev_filtered_4samples_list.
	unsigned int tid = threadIdx.x;
	unsigned int ind_count = tid;
	inlier_count_4points[ind_count] = 0;
	//thread 0 is in charge of the read and write elements: blockDim.x, blockDim.x + 1, ..., blockDim.x + 3
	//thread 1 is in charge of the read and write elements: blockDim.x+4, blockDim.x + 5, ..., blockDim.x + 7
	for (i = 0; i<nmss; i++)
		inlier_count_4points[blockDim.x + nmss * ind_count + i] = 0;

	__syncthreads();//syncthrize in block
					//	__threadfence();//syncthrize in grid
	tid = blockIdx.x * blockDim.x + threadIdx.x;
	*result_ninliers = 0;
	//	int Nloop = MIN(l_list / 2, blockDim.x*gridDim.x << 2);
	//	while (tid < l_list&&results[0].x==0)//used for sudden death=========
	while (k < dev_n_round[0])
	{
		//produce 4 different intergers to direct sample, samples are drawn in core, and the consensus are checked in n_matches
		//hit_ind is the output that sampled index of the core
		get_nrand(hit_ind, nmss, card_core, s_d_check);

		for (i=0;i<nmss;i++)
			X[i] = cm_ind[hit_ind[i]];//X is a case of c(n,4) combination.

		//printf("X=%d %d %d %d\n", X[0], X[1], X[2], X[3]);
		//normalization samples.
		//Input: X, dev_frames0x, dev_frames0y; Output: T1, X1;
		normal_samples(X, nmss, dev_frames0x, dev_frames0y, T1, X1);
		normal_samples(X, nmss, dev_frames1x, dev_frames1y, T2, X2);
		//Initialize Zuliani formulation of homogrphy transformation.
		initial_A(A, X1, X2, nmss);
		//Homography between 4-samples after using svd decomposition of A.
		homography(A, ndof, isnp, u9);//V is returned in A, where A=USV'. u9 is corresponds to homography matrix.
		double H[N3][N3];
		//H=[a(0,isnp), a(3,isnp), a(6,isnp)
		//   a(1,isnp), a(4,isnp), a(7,isnp)
		//   a(2,isnp), a(5,isnp), a(8,isnp)]
		for (i = 0; i < N3; i++)
			H[i][0] = u9[i];
		for (i = 0; i < N3; i++)
			H[i][1] = u9[N3 + i];
		for (i = 0; i < N3; i++)
			H[i][2] = u9[(N3 << 1) + i];

		//de-normalize Homography
		denorm_H(T2, H, T1);

		detH = H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - H[2][0] * H[1][1] * H[0][2] - H[2][1] * H[1][2] * H[0][0] - H[2][2] * H[1][0] * H[0][1];
		if (detH > 1.0e14 || detH < -1.0e14 || (detH < 0.00000000000001 && detH > -0.00000000000001))
		{
			for (i = 0; i < N3; i++)
				for (j = 0; j < N3; j++)
					H[i][j] = 0.0;
			detH = 0.0;
			temp_count = 0;
		}
		if (detH)//temp_count is the returned number of consensus points.
			temp_count = dev_get_consensus_set(dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, H, detH);
		if (temp_count > inlier_count_4points[ind_count])
		{
			inlier_count_4points[ind_count] = temp_count;
			for(i=0;i<nmss;i++)
			inlier_count_4points[blockDim.x + nmss * ind_count + i] = cm_ind[hit_ind[i]];
			//			printf("tid=%d, inlier_count_Io4si[%d]=%d, inlier_count_Io4si[%d]=%d\n", tid, ind_count, inlier_count_Io4si[ind_count], ind_count + blockDim.x, inlier_count_Io4si[ind_count + blockDim.x]);
			//			if (temp_count > 0.1*dim_m_I_cn24[0])//used for sudden death===========
			//				atomicExch(&results[0].x, 1);//used for sudden death===============
		}
		tid += gridDim.x * blockDim.x;
		k++;
	}
	__syncthreads();//syncthrize in block
	if (threadIdx.x == 0)//0th thread is used to explore local optimal results in its block.
	{
		for (i = 1; i < blockDim.x; i++)
			if (inlier_count_4points[i] > *(result_ninliers + blockIdx.x))
			{
				*(result_ninliers + blockIdx.x) = inlier_count_4points[i];//write result back to global memory
				for (k = 0; k < nmss; k++)
					result_MSS[nmss * blockIdx.x + k] = inlier_count_4points[blockDim.x + nmss * i + k];
			}
	}
}


//exploring inliers using H and T_noise_squared
//Output:
//CS CS[i]=1 means the i-th match is an inlier pair (i=1,...,n_matches). CS[i]=0 means the i-th match is an outlier pair
//d2
__host__ int host_get_consensus_set(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int n_matches, double H[N3][N3], double detH, int *CS, double *d2)
{
	int k, i, inlier_count = 0, t = 0;
	double temp[N3], e1, e2;
	double2 Xd;
	for (k = 0; k < n_matches; k++)
	{
		for (i = 0; i < N3; i++)//map frames0 in I0 into I1.
			temp[i] = H[i][0] * dev_frames0x[k] + H[i][1] * dev_frames0y[k] + H[i][2];
		Xd.x = temp[0] / temp[2];
		Xd.y = temp[1] / temp[2];
		//calculate the error between mapped frames0 in I1 and frames1 in I1.
		e1 = (dev_frames1x[k] - Xd.x)*(dev_frames1x[k] - Xd.x) + (dev_frames1y[k] - Xd.y)*(dev_frames1y[k] - Xd.y);
		//inv(H):H=[a b c;
		//		d e f;
		//		g h i];
		//		//1/det(H)*
		//		[ei-hf -bi+hc bf-ce; 		H[1][1]*H[2][2]-H[2][1]*H[1][2]   -H[0][1]*H[2][2]+H[2][1]*H[0][2]  H[0][1]*H[1][2]-H[0][2]*H[1][1]; 
		//		fg-id -cg+ia cd-af; 		H[1][2]*H[2][0]-H[2][2]*H[1][0]   -H[0][2]*H[2][0]+H[2][2]*H[0][0]  H[0][2]*H[1][0]-H[0][0]*H[1][2];
		//		dh-ge -ah+gb ae-bd]		H[1][0] * H[2][1] - H[2][0] * H[1][1] - H[0][0] * H[2][1] + H[2][0] * H[0][1]  H[0][0] * H[1][1] - H[0][1] * H[1][0]
		//map frames0 in I1 into I0.
		temp[0] = ((H[1][1] * H[2][2] - H[2][1] * H[1][2])* dev_frames1x[k] + (H[2][1] * H[0][2] - H[0][1] * H[2][2])* dev_frames1y[k] + H[0][1] * H[1][2] - H[0][2] * H[1][1]) / detH;
		temp[1] = ((H[1][2] * H[2][0] - H[2][2] * H[1][0])* dev_frames1x[k] + (H[2][2] * H[0][0] - H[0][2] * H[2][0])* dev_frames1y[k] + H[0][2] * H[1][0] - H[0][0] * H[1][2]) / detH;
		temp[2] = ((H[1][0] * H[2][1] - H[2][0] * H[1][1])* dev_frames1x[k] + (H[2][0] * H[0][1] - H[0][0] * H[2][1])* dev_frames1y[k] + H[0][0] * H[1][1] - H[0][1] * H[1][0]) / detH;
		Xd.x = temp[0] / temp[2];
		Xd.y = temp[1] / temp[2];
		//calculate the error between mapped frames0 in I0 and frames0 in I0.
		e2 = (dev_frames0x[k] - Xd.x)*(dev_frames0x[k] - Xd.x) + (dev_frames0y[k] - Xd.y)*(dev_frames0y[k] - Xd.y);
		//		if (k == 409 || k == 541 || k == 542 || k == 672)
		//			printf("%f\n", e1 + e2);
		if (e1 + e2 < T_noise_squared)
		{
			inlier_count++;
			CS[k] += 1;//record inlier k-th match in CS.
			*d2 += sqrt(e1 + e2);//d=sigma_i=0^inlier_count sqrt((e1+e2)) is the total projection error
		}
	}
	*d2 /= (inlier_count << 1);//mean projection error.
	return(inlier_count);
}

__host__ void host_initial_A(double *A, double2 *X1, double2 *X2, int m, int n)
{
	int i, j;
	j = 0;
	for (i = 0; i < m; i++)
	{
		*(A + j*n) = X1[i].x; //A[j][0] = X1[0][i]
		*(A + j*n + 1) = 0;//A[j][1]
		*(A + j*n + 2) = -X1[i].x * X2[i].x;//A[j][2] = -X1[0][i] * X2[0][i];
		*(A + j*n + 3) = X1[i].y;//A[j][3] = X1[1][i];
		*(A + j*n + 4) = 0;//A[j][4]
		*(A + j*n + 5) = -X1[i].y * X2[i].x;//A[j][5] = -X1[1][i] * X2[0][i];
		*(A + j*n + 6) = 1.0;//A[j][6]
		*(A + j*n + 7) = 0.0;//A[j][7]
		*(A + j*n + 8) = -X2[i].x;//A[j][8] = -X2[0][i];

		*(A + (j + 1)*n) = 0.0;//A[j + 1][0]
		*(A + (j + 1)*n + 1) = X1[i].x;//A[j + 1][1] = X1[0][i];
		*(A + (j + 1)*n + 2) = -X1[i].x * X2[i].y;//A[j + 1][2]  = -X1[0][i] * X2[1][i];
		*(A + (j + 1)*n + 3) = 0.0;//A[j + 1][3]
		*(A + (j + 1)*n + 4) = X1[i].y;//A[j + 1][4]  = X1[1][i];
		*(A + (j + 1)*n + 5) = -X1[i].y * X2[i].y;//A[j + 1][5] = -X1[1][i] * X2[1][i];
		*(A + (j + 1)*n + 6) = 0.0;//A[j + 1][6] 
		*(A + (j + 1)*n + 7) = 1.0;//A[j + 1][7]
		*(A + (j + 1)*n + 8) = -X2[i].y;//A[j + 1][8] = -X2[1][i];
		j += 2;
	}

}

//Seek inliers in frames using Xi.
//Xi is a case of c(n,4) combination.
//Output:
//Let CS[i]=k, if k greater than 0, the i-th match is an inlier. Otherwise, if k equals 0, the i-th match is an outlier.
//proj_error is the mean error of projection.
__host__ int assembleinliers(int4 Xi, float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int n_matches, int *CS, double *proj_error)
{
	int X[4], i, j, temp_count = 0;
	double T1[N3][N3], T2[N3][N3], H[N3][N3];
	double2 X1[4], X2[4];
	double detH;
	double A[ndof*isnp], u9[ndof];// u9 is the supplement of U so that[A u9] is a square unitary matrix after SVD decomposition.
	X[0] = Xi.x;//X is a case of c(n,4) combination.
	X[1] = Xi.y;
	X[2] = Xi.z;
	X[3] = Xi.w;
	//normalization samples.
	//Input: X, dev_frames0x, dev_frames0y; Output: T1, X1;
	normal_samples(X, 4, dev_frames0x, dev_frames0y, T1, X1);
	normal_samples(X, 4, dev_frames1x, dev_frames1y, T2, X2);

	//Initialize Zuliani formulation of homogrphy transformation.
	initial_A(A, X1, X2, 4);

	//Homography between 4-samples after using svd decomposition of A.
	homography(A, ndof, isnp, u9);//V is returned in A, where A=USV'. The last column of A is corresponds to homography matrix.
	double temp[N3][N3];
	//H=[a(0,isnp), a(3,isnp), a(6,isnp)
	//   a(1,isnp), a(4,isnp), a(7,isnp)
	//   a(2,isnp), a(5,isnp), a(8,isnp)]
	for (i = 0; i < N3; i++)
		H[i][0] = u9[i];
	for (i = 0; i < N3; i++)
		H[i][1] = u9[N3 + i];
	for (i = 0; i < N3; i++)
		H[i][2] = u9[(N3 << 1) + i];

	//de-normalize Homography
	denorm_H(T2, H, T1);

	detH = H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - H[2][0] * H[1][1] * H[0][2] - H[2][1] * H[1][2] * H[0][0] - H[2][2] * H[1][0] * H[0][1];
	//temp_count is the number of inliers after inliers exploring using H and T_noise_squared
	temp_count = host_get_consensus_set(dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, n_matches, H, detH, CS, proj_error);

	return(temp_count);
}


//Input:frames0x, frames0y, frames1x, frames1y, n_matches, ninliers, CS
//Output: CS_star, H, &proj_error
int reestimate(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int n_matches, int  ninliers, int *CS, double H[N3][N3], double *proj_error)
{
	int i, j;
	double T1[N3][N3], T2[N3][N3];
	double2 *X1_inliers, *X2_inliers;
	int *ind_inliers;
	ind_inliers = (int *)calloc(ninliers, sizeof(int));
	X1_inliers = (double2*)malloc(ninliers * sizeof(double2));
	X2_inliers = (double2*)malloc(ninliers * sizeof(double2));

	j = 0;
	for (i = 0; i < n_matches; i++)
	{
		if (*(CS + i))
			*(ind_inliers + j++) = i;
		*(CS + i) = 0;
	}

	normal_samples(ind_inliers, ninliers, dev_frames0x, dev_frames0y, T1, X1_inliers);
	normal_samples(ind_inliers, ninliers, dev_frames1x, dev_frames1y, T2, X2_inliers);

	double *A_hat = (double*)malloc(ndof *(ninliers << 1) * sizeof(double));
	//Initialize Zuliani formulation of homogrphy transformation.
	host_initial_A(A_hat, X1_inliers, X2_inliers, ninliers, ndof);
	double detH;

	//Homography between n-samples against get consensus set.
	double diag[ndof], v[ndof*ndof];
	double rv1[ndof];
	i = svd(A_hat, diag, v, rv1, ninliers << 1, ndof);
	//implement singular value decomposition with thin type.

	//choose singular vector corresponds to minimum sigular value.
	j = 0;
	for (i = 1; i < ndof; i++)
		if (fabsf(diag[i]) < fabsf(diag[j]))
			j = i;
	//H=[v(0,j), v(3,j), v(6,j)
	//   v(1,j), v(4,j), v(7,j)
	//   v(2,j), v(5,j), v(8,j)]
	for (i = 0; i < N3; i++)
		H[i][0] = *(v + i*ndof + j);
	for (i = 0; i < N3; i++)
		H[i][1] = *(v + (N3 + i)*ndof + j);
	for (i = 0; i < N3; i++)
		H[i][2] = *(v + ((N3 << 1) + i)*ndof + j);

	//de-normalize Homography
	denorm_H(T2, H, T1);//result in fitting homography using the current inliers.
						//Result in inliers and projection error using the fitting homography.
	detH = H[0][0] * H[1][1] * H[2][2] + H[0][1] * H[1][2] * H[2][0] + H[0][2] * H[1][0] * H[2][1] - H[2][0] * H[1][1] * H[0][2] - H[2][1] * H[1][2] * H[0][0] - H[2][2] * H[1][0] * H[0][1];
	*proj_error = 0.0;
	int inliers_count;
	inliers_count = host_get_consensus_set(dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, n_matches, H, detH, CS, proj_error);

	free(ind_inliers);
	free(X1_inliers);
	free(X2_inliers);
	free(A_hat);

	return(inliers_count);
}


int main(void) {
	FILE *fp;
	int *matches1, *matches2;
	/*matches is an array points to the matrix of matched pairs*/
	/*<matches(1,j),matches(2,j)> is a pair of match between image I0 and I1*/
	/*matches(1,j) corresponds to the j-th points in frame1*/
	/*matches(2,j) corresponds to the j-th points in frame2*/
	float *f1x, *f1y, *f2x, *f2y, *frames0x, *frames0y, *frames1x, *frames1y;
	int i, j, temp = 0, n_matches, n_frames0, n_frames1;
	char filename[] = "adam.txt";
	char fn_matches[200] = ".\\EVD\\matches_";
	strcat(fn_matches, filename);
	char fn_frames0[200] = ".\\EVD\\frames1_";
	strcat(fn_frames0, filename);
	char fn_frames1[200] = ".\\EVD\\frames2_";
	strcat(fn_frames1, filename);
	int n_I0, r_I0, c_I0, n_I1, r_I1, c_I1;
	/*n_I0 is the total number of pixels in I0, n_I0=r_I0*c_I0*/
	/*r_I0 is the number of rows of I0*/
	/*c_I0 is the number of column of I0*/
	char fn_I0[200] = ".\\EVD\\I1_";
	strcat(fn_I0, filename);
	char fn_I1[200] = ".\\EVD\\I2_";
	strcat(fn_I1, filename);
	char Hstr[200] = ".\\EVD\\csac_results\\H_im1_to_im2_";
	char inliersstr[200] = ".\\EVD\\csac_results\\ind_inliers_";
	char result_info[200] = ".\\EVD\\csac_results\\stat_";
	char error_fn1[200] = ".\\EVD\\csac_results\\err1_";
	char error_fn2[200] = ".\\EVD\\csac_results\\err2_";

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
	n_frames0 = readNDoubles(fp, f1x, n_frames0, 0);
	n_frames0 = readNDoubles(fp, f1y, n_frames0, 0);
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
	n_frames1 = readNDoubles(fp, f2x, n_frames1, 0);
	n_frames1 = readNDoubles(fp, f2y, n_frames1, 0);
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
	n_I0 = readNDoubles(fp, I0, n_I0, 0);
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
	n_I1 = readNDoubles(fp, I1, n_I1, 0);
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
	for (i = 0; i < 40; i++)//Initialization of ext_cn2
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
																															   //free buffer
	HANDLE_ERROR(cudaFree(dev_cn2_error));
	HANDLE_ERROR(cudaFree(dev_cn2_global_ind));
	//unbind texture variables
	HANDLE_ERROR(cudaEventDestroy(start2));
	HANDLE_ERROR(cudaEventDestroy(stop2));
	printf("host_cn2_error: %f %f", *(host_cn2_error + ext_cn2 / 20), *(host_cn2_error + ext_cn2 / 10));

	//================================================Produce c(n,4) combination list
	//Preparing: In order to produce c(n,4) combination list, we first need to transform global index of core to i,j index of core in matches.
	//core_ij_indx(i)=1 indicate the ith match pair is core match;
 	thrust::host_vector<int> core_ij_indx(n_matches, 0);
	for (i = 0; i < cc4; i++)
	{
		//		printf("*(host_ij_in_ind+*(host_cc4_core_global_ind + i))=%d ", host_ij_in_ind [*(host_cc4_core_global_ind + i)].x);
		core_ij_indx[host_ij_in_ind[*(host_cc4_core_global_ind + i)].x] += 1;
		core_ij_indx[host_ij_in_ind[*(host_cc4_core_global_ind + i)].y] += 1;
	}
	//count cardinal number of core: card(core_ij_indx)
	int card_core = 0, core_upper_lim = 300;
	for (i = 0; i < n_matches; i++)
	{
		if (core_ij_indx[i]>1)//default set to explore core with not less than 2 occurrences
		{
			//			printf("%d ", i);
			card_core++;
			if (card_core > core_upper_lim)
				break;
		}
	}
	temp = 0;
//	if (card_core < 100)//if the core is no more than 100, explore core with 1 occurrences
//	{
//		card_core = 0;
//		temp = 1;
//		for (i = 0; i < n_matches; i++)
//			if (core_ij_indx[i])
//			{
//			card_core++;
//			if (card_core > core_upper_lim)//set a limitation to the core cardinal to fit the memory provision.
//				break;
//			}
//	}
	printf("\card_core=%d\n", card_core);

	thrust::host_vector<int> cm_ind_thrust(card_core);
	int *cm_ind, *dev_cm_ind;
	cm_ind= (int *)malloc(card_core * sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**)&dev_cm_ind, card_core * sizeof(int)));
	//	int *cm_ind;
	//	cm_ind= (int *)malloc(card_core * sizeof(int));
	j = 0;
	//	fp = fopen("core_list.txt", "w");
	if (temp)
	{
		for (i = 0; i < n_matches; i++)
			if (core_ij_indx[i])
			{
				cm_ind_thrust[j] = i;
				cm_ind[j++] = i;
				if (j > core_upper_lim)//according to the core limitation
					break;
			}
	}
	else//explore core with default set
		for (i = 0; i < n_matches; i++)
			if (core_ij_indx[i] > 1)
			{
				cm_ind_thrust[j] = i;
				cm_ind[j++] = i;
				if (j > core_upper_lim)//according to the core limitation
					break;
			}
	HANDLE_ERROR(cudaMemcpy(dev_cm_ind, cm_ind, card_core * sizeof(int), cudaMemcpyHostToDevice));
	//	fclose(fp);

	int greedy = 0;//1: exploring all the MSS combinations generated on Core; 0: exploring ordinary RANSAC on Core.
	cudaEvent_t start3, stop3;//time counting
	float elapsedTime3=0;//time counting: Time for produce combination list
	cudaEvent_t start4, stop4;//time counting
	float elapsedTime4=0;//time counting: Time for combination list filter
	int2 *dev_results, *host_results, results, temp_results;
	cudaEvent_t start5, stop5;	//time counting: GPU RANSAC
	float elapsedTime5;//time counting
	int4 Xi;

	if (greedy)
	{
		//CUDA produce c(n,4) combination list, n=card_core
		unsigned int c_card_core_4 = card_core*(card_core - 1) / 2 * (card_core - 2) / 3 * (card_core - 3) / 4;
		thrust::device_vector<int4> dev_core_comb_list(c_card_core_4);
		thrust::host_vector<int4> host_core_comb_list(c_card_core_4);

		thrust::device_vector<int> dev_cm_ind_thrust(card_core);
		thrust::copy(cm_ind_thrust.begin(), cm_ind_thrust.end(), dev_cm_ind_thrust.begin());

		blocksPerGrid = card_core - 3;
		threadsPerBlock = card_core - 3;
		HANDLE_ERROR(cudaEventCreate(&start3));//time counting
		HANDLE_ERROR(cudaEventCreate(&stop3));//time counting
		HANDLE_ERROR(cudaEventRecord(start3, 0));//time counting
		prod_comb_list << <blocksPerGrid, threadsPerBlock >> >(thrust::raw_pointer_cast(&dev_core_comb_list[0]), thrust::raw_pointer_cast(&dev_cm_ind_thrust[0]), card_core);
		//	cudaMemcpy(host_core_comb_list, dev_core_comb_list, c_card_core_4 * sizeof(int4), cudaMemcpyDeviceToHost);
		thrust::copy(dev_core_comb_list.begin(), dev_core_comb_list.end(), host_core_comb_list.begin());
		//	for (i = 0; i < c_card_core_4; i++)
		//		if (i < 200)
		//			printf("%d %d %d %d     ", host_core_comb_list[i].x, host_core_comb_list[i].y, host_core_comb_list[i].z, host_core_comb_list[i].w);
		//===========================Data washing for CUDA Ransac
		//Before ransac implementation, we need to filter out redundant case of combination, such as common points, coalinear, illogical distribution.
		//The illogical distribution corresponds to points spacial distribution. The following graph shows an illigal distribution.
		//  Image1:      2  *                                   Image2:           2 o
		//        4  *    /                                              4 o       /
		//               /                                                        /
		//              /     3  *                                     3 o       /
		//             /                                                        /
		//          1 *                                                    1 o /
		HANDLE_ERROR(cudaEventRecord(stop3, 0));//time counting
		HANDLE_ERROR(cudaEventSynchronize(stop3));//time counting
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime3, start3, stop3));//time counting
		printf("Time for produce combination list: %3.1f ms\n", elapsedTime3);//time counting

		thrust::device_vector<int> dev_comb_ele_01judge(c_card_core_4, 1);

		HANDLE_ERROR(cudaEventCreate(&start4));//time counting
		HANDLE_ERROR(cudaEventCreate(&stop4));//time counting
		HANDLE_ERROR(cudaEventRecord(start4, 0));//time counting
		blocksPerGrid = device_prop.multiProcessorCount << 1;
		threadsPerBlock = 512;
		comb_list_filter << <blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(&dev_core_comb_list[0]), c_card_core_4, dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, thrust::raw_pointer_cast(&dev_comb_ele_01judge[0]));
		int ls = c_card_core_4 / 32 + 1;
		thrust::host_vector<int> host_comb_ele_01judge(c_card_core_4, 1);
		thrust::copy(dev_comb_ele_01judge.begin(), dev_comb_ele_01judge.end(), host_comb_ele_01judge.begin());
		j = 0;
		//	for (i = 46552000; i < 46553000; i++)
		//		if (host_comb_ele_01judge[i])
		//			j++;
		//	printf("\n%d\n", j);

		thrust::device_vector<int> s(ls);
		thrust::device_vector<int> s_prefixsum(ls);
		//judge_list_prefixsum mainly used for parallel prefix sum.
		//Let dev_comb_ele_01judge be: j0 ... j31  j32 ... j63 j64 ... j96 ......
		//partial_s(0)=sum(j0 ... j31),
		//partial_s(1)=sum(j32 ... j63),
		//partial_s(i)=sum(j(32*i) ... j(32*(i+1))-1)
		judge_list_prefixsum << <blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(&dev_comb_ele_01judge[0]), c_card_core_4, thrust::raw_pointer_cast(&s[0]), ls);
//		for (i = 0; i < ls; i++)
//			cout<< s[i]<<" ";
		thrust::inclusive_scan(s.begin(), s.end(), s_prefixsum.begin());
		//	for (i = 0; i < ls; i++)
		//		cout << s_prefixsum[i] << " ";

		thrust::device_vector<int4> dev_filtered_4samples_list(s_prefixsum[ls - 1]);
		thrust::host_vector<int4> host_filtered_4samples_list(s_prefixsum[ls - 1]);
		comb_list_reduce << <blocksPerGrid, threadsPerBlock >> > (thrust::raw_pointer_cast(&dev_core_comb_list[0]), thrust::raw_pointer_cast(&dev_comb_ele_01judge[0]), c_card_core_4, thrust::raw_pointer_cast(&s_prefixsum[0]), ls, thrust::raw_pointer_cast(&dev_filtered_4samples_list[0]));

		HANDLE_ERROR(cudaEventRecord(stop4, 0));//time counting
		HANDLE_ERROR(cudaEventSynchronize(stop4));//time counting
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime4, start4, stop4));//time counting
		cout << "The length of LoCSS (list of core sample set):" << c_card_core_4 << endl;
		int l_MSL = s_prefixsum[ls - 1];
		cout << "The length of MSL (minimal sample set) after filter redundant MSS (minimal sample set):" << s_prefixsum[ls - 1] << endl;
		printf("Time for combination list filter: %3.1f ms\n", elapsedTime4);//time counting
																			 //==================Output sampling list.
		thrust::copy(dev_filtered_4samples_list.begin(), dev_filtered_4samples_list.end(), host_filtered_4samples_list.begin());

		//	fp = fopen("filtered_list.txt", "w");
		//	for (i = 0; i < 10000; i++)
		//	{
		//			fprintf(fp,"%d %d %d %d\n", host_filtered_4samples_list[i].x, host_filtered_4samples_list[i].y, host_filtered_4samples_list[i].z, host_filtered_4samples_list[i].w);
		//	}
		//	fclose(fp);
		blocksPerGrid = device_prop.multiProcessorCount << 1;
		//	threadsPerBlock = device_prop.maxThreadsPerMultiProcessor;
		threadsPerBlock = 256;//based the number of variables used in CUDA __global__ function (total number of registers per block supported by 1080 is 64k)
		HANDLE_ERROR(cudaEventCreate(&start5));//time counting
		HANDLE_ERROR(cudaEventCreate(&stop5));//time counting
		HANDLE_ERROR(cudaEventRecord(start5, 0));//time counting

		HANDLE_ERROR(cudaMalloc((void**)&dev_results, blocksPerGrid * sizeof(int2)));

		//===================RANSAC implementation
		//dev_results[i].x is the maximum number of inliers in the i-th block.
		//dev_results[i].y is the corresponding index in dev_filtered_4samples_list.
		//Input:dev_filtered_4samples_list,s_prefixsum,dev_frames0x,...
		//Output:dev_results.x is a vector with the length gridDim, in which the i-th element is the maximum number of inliers in the i-th block.
		//dev_results.y is a vector with the length gridDim. The i-th element is the index in 4 samples list which implied maximum inliers in the i-th block.
		RANSAC_greedy << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) * 2 >> > (thrust::raw_pointer_cast(&dev_filtered_4samples_list[0]), s_prefixsum[ls - 1], dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, dev_results);
		host_results = (int2 *)malloc(blocksPerGrid * sizeof(int2));
		HANDLE_ERROR(cudaMemcpy(host_results, dev_results, blocksPerGrid * sizeof(int2), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaEventRecord(stop5, 0));//time counting
		HANDLE_ERROR(cudaEventSynchronize(stop5));//time counting
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime5, start5, stop5));//time counting
		printf("Time for greedy RANSAC against core list: %3.1f ms\n", elapsedTime5);//time counting
		printf("\nTime for GPU processing totals to: %3.1f ms\n", elapsedTime1 + elapsedTime2 + elapsedTime3 + elapsedTime4 + elapsedTime5);//time counting
																																			//Find optimal results so that results.x=max(host_results.x), results.y is corresponding to host_results.y
		results.x = host_results[0].x;//Initialization
		results.y = host_results[0].y;
		for (i = 1; i < blocksPerGrid; i++)
		{
			//		printf("results.x=%d  results.y=%d\n", host_results[i].x, host_results[i].y);
			if (host_results[i].x > results.x)
			{
				results.x = host_results[i].x;
				results.y = host_results[i].y;
			}
		}
		//	printf("\nresults.x=%d  results.y=%d\n", results.x, results.y);
		if (results.y == 0)
		{
			printf("There is no inlier explored. \n");
			exit(0);
		}
		Xi = host_filtered_4samples_list[results.y];//Xi is the optimal 4 samples which may seek maximum number of inliers.
		HANDLE_ERROR(cudaFree(dev_results));
	}
	else
	{
		blocksPerGrid = device_prop.multiProcessorCount << 1;
		//	threadsPerBlock = device_prop.maxThreadsPerMultiProcessor;
		threadsPerBlock = 256;//based the number of variables used in CUDA __global__ function (total number of registers per block supported by 1080 is 64k)
		HANDLE_ERROR(cudaEventCreate(&start5));//time counting
		HANDLE_ERROR(cudaEventCreate(&stop5));//time counting
		HANDLE_ERROR(cudaEventRecord(start5, 0));//time counting
		
		int s_d_check = 0, n_round = 100;//100 every GPU thread run n_round to return a result to host
		unsigned int T_iter = 10000, max_iter = 10, iter = 0;
		int temp_count = 0, N_I_star = 0, *dev_result_ninliers, *host_result_ninliers, result_ninliers, temp_results;
		double q, eps = 1e-30, epsilon = 1e-6, temp_double;
		int *dev_result_MSS, *host_result_MSS, results_MSS[nmss];

		n_round = card_core * P_CORE;
		temp_double = card_core / n_round  * (card_core - 1) / (n_round - 1) * (card_core - 2) / (n_round - 2) * (card_core - 3) / (n_round - 3);
		n_round = temp_double * 2 / threadsPerBlock / blocksPerGrid;
		if (n_round < 1)
			n_round = 1;

		
		HANDLE_ERROR(cudaMalloc((void**)&dev_result_ninliers, blocksPerGrid * sizeof(int)));//GPU write results back to global memory, since every block works independently in their shared memory.
		host_result_ninliers = (int *)malloc(blocksPerGrid * sizeof(int));
		for (i = 0; i<blocksPerGrid; i++)
			host_result_ninliers[i] = 0;
		HANDLE_ERROR(cudaMemcpy(dev_result_ninliers, host_result_ninliers, blocksPerGrid * sizeof(int), cudaMemcpyHostToDevice));

		//we design a variable with the length of the number of blocks to deliver results back to host memory
		HANDLE_ERROR(cudaMalloc((void**)&dev_result_MSS, blocksPerGrid * nmss * sizeof(int)));
		host_result_MSS = (int *)malloc(blocksPerGrid * nmss * sizeof(int));

		HANDLE_ERROR(cudaMemcpyToSymbol(dev_n_round, &n_round, sizeof(int)));
		for (i = 0; i < nmss; i++)
			s_d_check += i;

		while (iter <= T_iter && iter <= max_iter)
		{	//Input: n_matches, dev_frames0x,...
			//Output:dev_results.x is a vector with the length gridDim, in which the i-th element is the maximum number of inliers in the i-th block.
			//dev_results.y is a vector with the length gridDim. The i-th element is the index in 4 samples list which implied maximum inliers in the i-th block.
			//__global__ void RANSAC_fund(float *dev_frames0x, float *dev_frames0y, float *dev_frames1x, float *dev_frames1y, int l_list, int s_d_check, int *cm_ind, int card_core, int *result_ninliers, int *result_MSS)

			RANSAC << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) * 5 >> > (dev_frames0x, dev_frames0y, dev_frames1x, dev_frames1y, n_matches, s_d_check, dev_cm_ind, card_core, dev_result_ninliers, dev_result_MSS);
			HANDLE_ERROR(cudaMemcpy(host_result_ninliers, dev_result_ninliers, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(host_result_MSS, dev_result_MSS, blocksPerGrid * nmss * sizeof(int), cudaMemcpyDeviceToHost));

			//Find optimal MSS in host_result_MSS based on the corresponding number of inliers
			result_ninliers = host_result_ninliers[0];//Initialization
			for (j = 0; j < nmss; j++)
				results_MSS[j] = host_result_MSS[j];
			//printf("\nhost_result_MSS[j]:\n");
			//for(i=0;i<nmss*blocksPerGrid;i++)
			//	printf("%d ", host_result_MSS[i]);

			for (i = 1; i < blocksPerGrid; i++)
			{
				if (host_result_ninliers[i] > result_ninliers)
				{
					result_ninliers = host_result_ninliers[i];
					for (j = 0; j<nmss; j++)
						results_MSS[j] = host_result_MSS[i*nmss + j];
				}
			}
			printf("current host loop: %d, find %d inliers, corresponding optimal MSS: %d %d %d  %d \n", iter, result_ninliers, results_MSS[0], results_MSS[1], results_MSS[2], results_MSS[3]);
			temp_count = result_ninliers;
			if (temp_count > N_I_star)
			{
				//			rcs_star = rank_cs;
				N_I_star = temp_count;

				//update the number of iterations
				q = (double)(N_I_star) / (n_matches)*((double)(N_I_star - 1) / (n_matches - 1))*((double)(N_I_star - 2) / (n_matches - 2))*((double)(N_I_star - 3) / (n_matches - 3));
				if (q > eps)
					if ((1 - q) > 1e-12)
					{
						temp = log(epsilon) / log(1 - q);
						if (temp > max_iter)
							T_iter = max_iter;
						else
							T_iter = temp;
					}
					else
						T_iter = 0;

				printf("\nupdate the RANSAC results, total iter need: %u, n_inliers: %d, MSS: %d, %d, %d, %d, %d, %d, %d, %d\n", T_iter, N_I_star, results_MSS[0], results_MSS[1], results_MSS[2], results_MSS[3], results_MSS[4], results_MSS[5], results_MSS[6], results_MSS[7]);
			}
			if (temp_count == n_matches)
				break;
			iter = iter + threadsPerBlock*blocksPerGrid*n_round;//the main loop contains every thread runs n_round RANSAC exploring to return results to host memory.
			printf("iter: %d\n", iter);
		}
		HANDLE_ERROR(cudaEventRecord(stop5, 0));//time counting
		HANDLE_ERROR(cudaEventSynchronize(stop5));//time counting
		HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime5, start5, stop5));//time counting
		printf("Time for RANSAC against core list: %3.1f ms\n", elapsedTime5);//time counting
		printf("\nTime for GPU processing totals to: %3.1f ms\n", elapsedTime1 + elapsedTime2 + elapsedTime5);//time counting

		if (result_ninliers == 0)
		{
			printf("There is no inlier explored. \n");
			exit(0);
		}
		Xi.x = results_MSS[0];
		Xi.y = results_MSS[1];
		Xi.z = results_MSS[2];
		Xi.w = results_MSS[3];

	}

	HANDLE_ERROR(cudaFree(dev_frames0x));
	HANDLE_ERROR(cudaFree(dev_frames0y));
	HANDLE_ERROR(cudaFree(dev_frames1x));
	HANDLE_ERROR(cudaFree(dev_frames1y));

	double H[N3][N3], proj_error = 0.0;
	int *CS = (int*)calloc(n_matches, sizeof(int));

	int ninliers;
	//		Xi = host_filtered_4samples_list[host_results[i].y];//Xi is the candidate of the optimal 4 samples resulting in Block, in which is used to seek maximum number of inliers.
	//		proj_error = 0.0;
	ninliers = assembleinliers(Xi, frames0x, frames0y, frames1x, frames1y, n_matches, CS, &proj_error);
	ninliers = 0;
	for (j = 0; j < n_matches; j++)
	{
		if (*(CS + j))
		{
			ninliers++;
		}
	}
	//Seek inliers and re-estimate homography H in matches using ninliers and CS. 
	//Input:frames0x, frames0y, frames1x, frames1y, n_matches, ninliers, CS
	//Output: CS, H, &proj_error=========================
	ninliers = reestimate(frames0x, frames0y, frames1x, frames1y, n_matches, ninliers, CS, H, &proj_error);//1 indicate the first model estimation.

	QueryPerformanceCounter(&time_over);    //计时结束  
	duration = 1000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;

	//eliminate redundent correspondences
//	for (i=0;i<n_matches;i++)
//		for (j = i + 1; j < n_matches; j++)
//		{
//			if (*(CS + i) && *(CS + j))
//			if( fabs(*(frames0x + i) - *(frames0x + j))<0.1 && fabs(*(frames0y + i) - *(frames0y + j))<0.1 && fabs(*(frames1x + i) - *(frames1x + j))<0.1 && fabs(*(frames1y + i) - *(frames1y + j))<0.1)
//			{
//				printf("i=%d:%d,j=%d:%d,%f,%f,%f,%f\n", i, *(CS + i), j, *(CS + j), *(frames0x + i), *(frames1x + j), *(frames0y + i), *(frames1y + j));
//				*(CS + j) = 0;
//				ninliers--;
//			}
//		}

	//==============================================================================================
	//registration decision
	//estimate p_N1 under the assumption that the number of true inliers equals the resulting ninliers
	double mu_hat = 0, sig2_hat = 0, sum_error2 = 0, p_X1, p_X2, error_thresh = host_cn2_error[card_core - 1];
	int *inliers, *outliers, n_outliers, n_error, ind, n_p_N1 = 0, indi, indj;
	n_outliers = n_matches - ninliers;
	inliers = (int*)calloc(ninliers, sizeof(int));
	outliers = (int*)calloc(n_outliers, sizeof(int));
	i = 0; indj = 0;
	for (j = 0; j < n_matches; j++)
	{
		if (*(CS + j))
			*(inliers + i++) = j;//result in inlier list
		else
			*(outliers + indj++) = j;//result in outlier list
	}

	//dual index (i,j) mapping to dictional index: (n_matches-1)+(n_matches-2)+...+(n_matches-i)+(j-i)-1=n_matches*i-((i+1)*i)/2+j-i-1;
//	char error_fn[100] = "errorg2_";
	printf("ext_cn2, ninliers, cc4 : %d %d %d\n", ext_cn2, ninliers, cc4);
	strcat(error_fn2, filename);
	fp = fopen(error_fn2, "w");
//use inliers to evaluate the true correspondes
//the indexes of the inliers are used to extract the PC (pair of correspondces) errors
//n_p_N1 is the number of true PCs.
	for (i = 0; i < ninliers - 1; i++)
	{
		indi = *(inliers + i);
		for (j = i + 1; j < ninliers; j++)
		{
			indj = *(inliers + j);
			//for the dual index produced by the i-th and j-th true matches, it index to 
			ind = n_matches*indi - ((indi + 1)*indi) / 2 + indj - indi - 1;
			if (host_cn2_error_beforesort[ind] == 1000)
			{//dist error 1000 is a signal to identify a pair of distance close matches, the case is excluded in error counting, and should not be counted for probability distribution.
			 //	n_p_N1++;
				continue;
			}
			fprintf(fp, "%.6f\n", host_cn2_error_beforesort[ind]);
//			if (i==751 && j>1438)
//				printf("%d %d %d %d %d %d %.6f\n", i, j, indi, indj, *(host_cc4_core_global_ind + i), ind, host_cn2_error_beforesort[ind]);
			sum_error2 += host_cn2_error_beforesort[ind] * host_cn2_error_beforesort[ind];
			mu_hat += host_cn2_error_beforesort[ind];
			n_p_N1++;
		}
	}
	fclose(fp);
	//	n_error = (ninliers*(ninliers - 1)) / 2;
//	printf("markaaaaaaaaaaaaaaaaaaaaaaaaa\n");

	n_error = n_p_N1;
	mu_hat /= n_error;
	sig2_hat = sum_error2 / (n_error - 1) - n_error *mu_hat / (n_error - 1)*mu_hat;
	double fx, cdf_fx = 0, daitax;
	fx = 1 / sqrt(6.283185307179586*sig2_hat);
//	printf("markbbbbbbbbbbbbbbbbbbbbbbbbbbb\n");

	daitax = error_thresh / 100;
	for (i = 1; i <= 100; i++)
		cdf_fx += exp(-(daitax*i - mu_hat) *(daitax*i - mu_hat) / 2 / sig2_hat);
	cdf_fx *= daitax;
	cdf_fx *= fx;
	p_X2 = cdf_fx;
//	printf("markcccccccccccccccccccccccccccccc\n");

	double cardb;
	double upperbound_N1;
	n_p_N1 = 0;
//count the matches, where its two matches are located near each other within 5 pixels. Their errors are valued 1000 in the function: extract_parallel_dist_cn2
	for (i = 0; i < ninliers; i++)
	{
		indi = *(inliers + i);
		for (j = i + 1; j < ninliers; j++)
		{
			indj = *(inliers + j);
			if (indj < indi)
				ind = n_matches*indj - ((indj + 1)*indj) / 2 + indi - indj - 1;
			else
				ind = n_matches*indi - ((indi + 1)*indi) / 2 + indj - indi - 1;
			if (host_cn2_error_beforesort[ind] == 1000)
			{//dist error 1000 is a signal to identify a pair of distance close matches, save its number for the forcecoming usage.
				n_p_N1++;
				break;
			}
		}
	}
//	printf("markdddddddddddddddddddddddddddddddd\n");

	int ninliers_fused;
	ninliers_fused = ninliers - n_p_N1;//do not count matches with close distance

	n_p_N1 = 0;
	for (i = 0; i < n_outliers; i++)
	{
		indi = *(outliers + i);
		for (j = i + 1; j < n_outliers; j++)
		{
			indj = *(outliers + j);
			if (indj < indi)
				ind = n_matches*indj - ((indj + 1)*indj) / 2 + indi - indj - 1;
			else
				ind = n_matches*indi - ((indi + 1)*indi) / 2 + indj - indi - 1;
			if (host_cn2_error_beforesort[ind] == 1000)
			{//dist error 1000 is a signal to identify a pair of distance close matches, save its number for the forcecoming usage.
				n_p_N1++;
				break;
			}
		}
	}
	int n_outliers_fused;
	n_outliers_fused = n_outliers - n_p_N1;//do not count matches with close distance
//	printf("markeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n");

										   //	cardb = (ninliers*(ninliers - 1)) / 2;
	cardb = (ninliers_fused*(ninliers_fused - 1)) / 2;//C_NI^2 in Eq. (38)
	upperbound_N1 = cardb*p_X2;
	cardb = upperbound_N1;//cosi in Eq. (38)
						  //	if (carda >= 4)
						  //		carda = carda*(carda - 1) / 2 * (carda - 2) / 3 * (carda - 3) / 4;
						  //	else
						  //		carda = 0;
	upperbound_N1 = (1 + sqrt(1 + 32 / p_X2)) / 2;//Tlb
//	printf("markfffffffffffffffffffffffffffffffffff\n");


	n_p_N1 = 0;
	double mu_outin = 0, sum_err_outin = 0;
//	strcpy(error_fn, "errorg1_");
	strcat(error_fn1, filename);
	fp = fopen(error_fn1, "w");
	for (i = 0; i < ninliers; i++)
	{
		indi = *(inliers + i);
		for (j = 0; j < n_outliers; j++)
		{
			indj = *(outliers + j);
			if (indj < indi)
				ind = n_matches*indj - ((indj + 1)*indj) / 2 + indi - indj - 1;
			else
				ind = n_matches*indi - ((indi + 1)*indi) / 2 + indj - indi - 1;
			if (host_cn2_error_beforesort[ind] == 1000)
			{//dist error 1000 is a signal to identify a pair of distance close matches, the case is excluded in error counting, and should not be counted for probability distribution.
			 //n_p_N1++;
				continue;
			}
			fprintf(fp, "%.6f\n", host_cn2_error_beforesort[ind]);
			sum_err_outin += host_cn2_error_beforesort[ind] * host_cn2_error_beforesort[ind];
			mu_outin += host_cn2_error_beforesort[ind];
			n_p_N1++;
		}
	}
	fclose(fp);
//	printf("markgggggggggggggggggggggggggggggggg\n");

	//	n_error = ninliers*n_outliers;
	n_error = n_p_N1;
	mu_outin /= n_error;
	sig2_hat = sum_err_outin / (n_error - 1) - n_error*mu_outin / (n_error - 1)*mu_outin;

	fx = 1 / sqrt(6.283185307179586*sig2_hat);
	daitax = error_thresh / 100;
	for (i = 1; i <= 100; i++)
		cdf_fx += exp(-(daitax*i - mu_outin) *(daitax*i - mu_outin) / 2 / sig2_hat);
	cdf_fx *= daitax;
	cdf_fx *= fx;
	p_X1 = cdf_fx;//P_D
	double cardc;//Card(hat(A)2)

	//	cardc = p_X1*n_error*p_X1*(ninliers-1)/2 * p_X1*(ninliers - 2) / 3 * p_X1*(ninliers - 3) / 4;
	//	cardc+= p_X1* ninliers * n_outliers* p_X1 * (ninliers - 1) / 2 * p_X1*(ninliers - 2) / 3 *(ninliers - 3) / 4 *(n_outliers-1)/2* p_X1*20;
	//	cardc += p_X1* ninliers * n_outliers* p_X1 * (ninliers - 1) / 2 * (n_outliers - 1) / 2 * p_X1*(ninliers - 2) / 3  * (ninliers - 3) / 4 * (n_outliers - 2) / 3 * p_X1*30;
	//	cardc += p_X1* ninliers * n_outliers* p_X1 * (ninliers - 1) / 2 * (n_outliers - 1) / 2 * p_X1*(ninliers - 2) / 3 * (n_outliers - 2) / 3 * (ninliers - 3) / 4 * (n_outliers - 3) / 4 * p_X1 * 9;
	cardc = p_X1*ninliers_fused*n_outliers_fused*p_X1*(ninliers_fused - 1) / 2 * p_X1*(ninliers_fused - 2) / 3 * p_X1*(ninliers_fused - 3) / 4;
	cardc += p_X1* ninliers_fused * n_outliers_fused* p_X1 * (ninliers_fused - 1) / 2 * p_X1*(ninliers_fused - 2) / 3 * (ninliers_fused - 3) / 4 * (n_outliers_fused - 1) / 2 * p_X1 * 20;
	cardc += p_X1* ninliers_fused * n_outliers_fused* p_X1 * (ninliers_fused - 1) / 2 * (n_outliers_fused - 1) / 2 * p_X1*(ninliers_fused - 2) / 3 * (ninliers_fused - 3) / 4 * (n_outliers_fused - 2) / 3 * p_X1 * 36;
	cardc += p_X1* ninliers_fused * n_outliers_fused* p_X1 * (ninliers_fused - 1) / 2 * (n_outliers_fused - 1) / 2 * p_X1*(ninliers_fused - 2) / 3 * (n_outliers_fused - 2) / 3 * (ninliers_fused - 3) / 4 * (n_outliers_fused - 3) / 4 * p_X1 * 24;

	printf("\nThe resulting number of inliers is: %d\n", ninliers);
	//	finish = clock();
	//	duration = (double)(finish - start) / CLOCKS_PER_SEC;

	printf("\nThe time cost against the whole demo, including GPU and CPU processing time (ms): %3.2f\n", duration);
	fp = fopen("time.txt", "w");
	fprintf(fp, "The time cost against the whole demo, including GPU and CPU processing time (ms): %3.2f\n", duration);
	fclose(fp);

	printf("The projection error of the inliers is :%3.2f\n", proj_error);
//	fp = fopen("proj_error.txt", "w");
//	fprintf(fp, "The projection error of the inliers is :%3.2f\n", proj_error);
//	fclose(fp);
	strcat(result_info, filename);
	fp = fopen(result_info, "w");
	fclose(fp);
	fp = fopen(result_info, "a");

	//card_core| ninliers| cn2| Bitonic sort| produce combination list| combination list filter| RANSAC against core list| The length of LoCSS| l_MSL| GPU| the whole time cost| proj_error | cardb (cosi in Eq. (38)) | cardc/Card(hat(A)2) eq. (40)| upperbound_N1/Tlb
	fprintf(fp, "%d\t %d\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t", card_core, ninliers, elapsedTime1, elapsedTime2, elapsedTime3, elapsedTime4, elapsedTime5, elapsedTime1 + elapsedTime2 + elapsedTime3 + elapsedTime4 + elapsedTime5, duration, proj_error, cardb, cardc, upperbound_N1);
	fclose(fp);
	strcat(inliersstr, filename);
	strcat(Hstr, filename);
	fp = fopen(inliersstr, "w");
	//	printf("The indexs of inliers in matches are explored as follows.\n");
	for (j = 0; j < n_matches; j++)
	{
		if (*(CS + j))
			fprintf(fp, "%d\n", j);
		//		fprintf(fp, "%d\n", *(CS + j));
		//		printf("%d ", *(ind_inliers + j));
	}
	fclose(fp);
	fp = fopen(Hstr, "w");
	printf("\nHomography from Image1 to Image2 H=\n");
	for (i = 0; i < N3; i++)
	{
		for (j = 0; j < N3; j++)
		{
			fprintf(fp, "%.16f ", H[i][j]);
			printf("%f ", H[i][j]);
		}
		fprintf(fp, "\n");
		printf("\n");
	}
	fclose(fp);

	free(inliers);
	free(outliers);
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
	free(cm_ind);
 }
