/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
//#include <algorithm>
#include <stdio.h>
//#include <float.h>
//#include <thrust/extrema.h>
//#include <thrust/execution_policy.h>

__global__ void Max(float *d_max, const float * d_logLuminance, int size)
{
  // Initialize shared array
  extern __shared__ float temp[];
  int tid = threadIdx.x;
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  // Check if out of bounds
  if (index >= size)
  {  
    return;
  }  

  // Set all the values in temp
  temp[tid] = d_logLuminance[index];
  __syncthreads();

  // Do the comparisons
  for (int i=blockDim.x/2; i>0; i/=2)
  {
    if (tid<i && temp[tid]<temp[tid+i])
    {
      temp[tid] = temp[tid+i];
    }
  }
  __syncthreads();

  // The first thread writes the blockwise result
  if(tid==0)
  {
    //printf("%f", temp[0]);
    d_max[blockIdx.x] = temp[0];
  
  }
  

}

__global__ void Min(float *d_min, const float * d_logLuminance, int size)
{
  // Initialize shared array
  extern __shared__ float temp[];
  int tid = threadIdx.x;
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  
  // Check if out of bounds
  if (index >= size)
  {  
    return;
  }  

  // Set all the values in temp
  temp[tid] = d_logLuminance[index];
  __syncthreads();

  // Do the comparisons
  for (int i=blockDim.x/2; i>0; i/=2)
  {
    if (tid<i && temp[tid]>temp[tid+i])
    {
      temp[tid] = temp[tid+i];
    }
  }
  __syncthreads();

  // The first thread writes the blockwise result
  if(tid==0)
  {
    //printf("%f", temp[0]);
    d_min[blockIdx.x] = temp[0];
  
  }

}

 __global__ void histogram(unsigned int *d_histo, const float* d_logLuminance, float min_logLum, float range, const size_t numBins)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Use the provided equation to select the bin
  unsigned int target = min((unsigned int)(numBins-1), (unsigned int)(((d_logLuminance[index] - min_logLum) / range) * numBins));
  
  // Add 1 to the corresponding bin
  atomicAdd(&(d_histo[target]), 1);
}

 __global__ void scan(unsigned int * d_cdf, unsigned int * d_histo, const size_t numBins)
{
  //1024 is the number of bins
  __shared__ unsigned int temp[1024*2];
  int id = threadIdx.x; int pout = 0, pin = 1;
  
  // Initialize temp to the histogram bins offset by 1
  temp[id] = (id > 0) ? d_histo[id - 1] : 0;
  __syncthreads();

  // Loop through, adding the offset values the current value
  for (int offset=1; offset<numBins; offset<<=1)
  {
    pout = 1-pout;
    pin = 1-pout;
   
    if (id >= offset)
    {
      temp[pout*numBins+id] = temp[pin*numBins+id] + temp[pin*numBins+id-offset];
    }
    else
    {
      temp[pout*numBins+id] = temp[pin*numBins+id];
    }
    __syncthreads();
  }

  // Write the result to the output
  d_cdf[id] = temp[pout*numBins+id];
}
 
 
void your_histogram_and_prefixsum(const float* d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  
  // Allocate intermediate memory 
  float * h_min = (float *)malloc(sizeof(float));
  float * h_max = (float *)malloc(sizeof(float));
  float *d_min, *d_max; 
  unsigned int * d_histo;
  float * d_nintermediate, *d_xintermediate;

  // Initialize histogram to all 0s 
  unsigned int * h_histo[numBins];
  for(unsigned int i=0; i<numBins; i++)
  {
    h_histo[i] = 0;
  }

  // Kernel dimensions
  unsigned int numVals = numRows * numCols;
  unsigned int threadsPerBlock = 1024;
  unsigned int numBlocks = numVals / threadsPerBlock;
    
  // Allocate GPU memory
  checkCudaErrors(cudaMalloc((void **) &d_histo, numBins*sizeof(unsigned int))); 
  checkCudaErrors(cudaMalloc((void **) &d_nintermediate, 2*numBlocks*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &d_xintermediate, 2*numBlocks*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &d_min, sizeof(float)));
  checkCudaErrors(cudaMalloc((void **) &d_max, sizeof(float)));
  cudaDeviceSynchronize();
  
  // Copy histogram to device
  checkCudaErrors(cudaMemcpy(d_histo, h_histo, numBins*sizeof(int), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();


  // First calls to min and max to get blockwise extrema
  Max<<<numBlocks, threadsPerBlock, 2*threadsPerBlock*sizeof(float)>>>(d_xintermediate,d_logLuminance, numVals);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  Min<<<numBlocks, threadsPerBlock, 2*threadsPerBlock*sizeof(float)>>>(d_nintermediate,d_logLuminance, numVals);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
 // Second calls to min and max to get global extrema
  Max<<<1, numBlocks, numBlocks*sizeof(float)>>>(d_max, d_xintermediate, numBlocks);
  Min<<<1, numBlocks, numBlocks*sizeof(float)>>>(d_min, d_nintermediate, numBlocks);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // Copy the results back to the host
  cudaMemcpy(h_min, d_min, sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_max, d_max, sizeof(int),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  // Set the min, max, and range
  min_logLum = *h_min;
  max_logLum = *h_max; 
  float range = max_logLum - min_logLum;
 
  // Call kernel to bin values
  histogram<<<numBlocks, threadsPerBlock>>>(d_histo, d_logLuminance, min_logLum, range, numBins); 
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  // Do the prefix sum
  scan<<<1, numBins>>>(d_cdf, d_histo, numBins);  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Cleanup
  free(h_min);
  free(h_max); 
  checkCudaErrors(cudaFree(d_histo));
  checkCudaErrors(cudaFree(d_xintermediate)); 
  checkCudaErrors(cudaFree(d_nintermediate)); 
  checkCudaErrors(cudaFree(d_min)); 
  checkCudaErrors(cudaFree(d_max)); 
}
