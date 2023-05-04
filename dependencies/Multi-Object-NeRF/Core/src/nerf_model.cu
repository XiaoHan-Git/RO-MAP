
#include "nerf_model.h"
#include "common.h"  
#include <opencv/cv.hpp>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <chrono>

namespace nerf{

//static 
json NeRF_Model::ClassNetworkConfig;

//-----------------------------------------------------------------
//------------------------CUDA function----------------------------
//-----------------------------------------------------------------

__device__ float network_to_rgb(float val,  ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

struct LossAndGradient {
	Eigen::Array3f loss;
	Eigen::Array3f gradient;
	__host__ __device__ LossAndGradient operator*(float scalar) {
		return {loss * scalar, gradient * scalar};
	}
	__host__ __device__ LossAndGradient operator/(float scalar) {
		return {loss / scalar, gradient / scalar};
	}
};

//L2 loss
__device__ LossAndGradient loss_and_gradient(const Eigen::Vector3f& target, const Eigen::Vector3f& prediction) {
	Eigen::Array3f difference = prediction - target;
	return {
		difference * difference,
		2.0f * difference
	};
}

//AABB
__host__ __device__ Eigen::Vector2f ray_intersect(const BoundingBox& box, const Eigen::Vector3f& pos, const Eigen::Vector3f& dir) 
{   
    Eigen::Vector3f min = box.min;
    Eigen::Vector3f max = box.max;

    float tmin = (min.x() - pos.x()) / dir.x();
    float tmax = (max.x() - pos.x()) / dir.x();

    if (tmin > tmax) {
        tcnn::host_device_swap(tmin, tmax);
    }

    float tymin = (min.y() - pos.y()) / dir.y();
    float tymax = (max.y() - pos.y()) / dir.y();

    if (tymin > tymax) {
        tcnn::host_device_swap(tymin, tymax);
    }

    if (tmin > tymax || tymin > tmax) {
        return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (min.z() - pos.z()) / dir.z();
    float tzmax = (max.z() - pos.z()) / dir.z();

    if (tzmin > tzmax) {
        tcnn::host_device_swap(tzmin, tzmax);
    }

    if (tmin > tzmax || tzmin > tmax) {
        return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    return { tmin, tmax };
}

__device__ Eigen::Vector3f WarpPoint(const Eigen::Vector3f& point, const BoundingBox& box)
{   
    Eigen::Vector3f res = (point - box.min).cwiseQuotient(box.max-box.min);
    return res;
}

__device__ Eigen::Vector3f UnWarpPoint(const Eigen::Vector3f& point,const BoundingBox& box)
{
    Eigen::Vector3f res = box.min + point.cwiseProduct(box.max-box.min);
    return res;
}

//function for importance sampling, not used
/* 
//STL function __lower_bound()
__device__ int lower_bound(float *array, int len, float key)
{
    int first = 0, middle, half;
    while(len > 0)
    {
        half = len >> 1;
        middle = first + half;
        if(array[middle] < key)
        {
            first = middle +1;
            len = len - half -1;
        }
        else
            len = half;
    }
    return first;

}

__device__ void selection_sort_nv(float *data, int left, int right )
{
  for( int i = left ; i <= right ; ++i ){
    float min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j ){
      float val_j = data[j];
      if( val_j < min_val ){
        min_idx = j;
        min_val = val_j;
      }
    }
    // Swap the values.
    if( i != min_idx ){
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__device__ void selection_sort(float* arr, const uint32_t len)
{
    uint32_t i,j;
    float temp;
    for (i = 0 ; i < len - 1 ; i++) 
    {
        uint32_t min = i;
        for (j = i + 1; j < len; j++) 
        {   
            if (arr[j] < arr[min]) 
            {  
                min = j;   
            }
        }
        if (min != i)  
		{
			temp = arr[i];
			arr[i] = arr[min];
			arr[min] = temp;
		}
    }
}

__device__ void QuickSort(float* arr, int start, int end)
{
	if (start >= end)
		return;
	int i = start;
	int j = end;
	
	float baseval = arr[start];
	while (i < j)
	{
		
		while (i < j && arr[j] >= baseval)
		{
			j--;
		}
		if (i < j)
		{
			arr[i] = arr[j];
			i++;
		}
		
		while (i < j && arr[i] < baseval)
		{
			i++;
		}
		if (i < j)
		{
			arr[j] = arr[i];
			j--;
		}
	}
	
	arr[i] = baseval;
	QuickSort(arr, start, i - 1);
	QuickSort(arr, i + 1, end);
}
 */

template <typename T>
__global__ void fill_rollover(const uint32_t n_elements, const uint32_t stride, const uint32_t* n_input_elements_ptr, T* inout) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;

	if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0) return;

	T result = inout[i % (n_input_elements * stride)];
	inout[i] = result;
}

template <typename T>
__global__ void fill_rollover_and_rescale(const uint32_t n_elements, const uint32_t stride, const uint32_t* n_input_elements_ptr, T* inout) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;

	if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0) return;

	T result = inout[i % (n_input_elements * stride)];
	result = (T)((float)result * n_input_elements / n_elements);
	inout[i] = result;
}

__global__ void fill_rollover_rays(const uint32_t n_elements,const uint32_t* n_input_elements_ptr,Ray* rays,uint8_t* raysInstance ,float* rgb,float* depth)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t n_input_elements = *n_input_elements_ptr;
    //printf("%d\n",n_input_elements);
    if (i < n_input_elements || i >= n_elements)
        return;
    uint32_t idx = i % n_input_elements;
    rays[i] = rays[idx];
    raysInstance[i] = raysInstance[idx];
    rgb[i*3] = rgb[idx*3];
    rgb[i*3+1] = rgb[idx*3+1];
    rgb[i*3+2] = rgb[idx*3+2];
    depth[i] = depth[idx];
}

__global__ void generate_grid_samples_nerf_uniform(Eigen::Vector3i res_3i, float* __restrict__ out) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x>=res_3i.x() || y>=res_3i.y() || z>=res_3i.z())
		return;
	uint32_t i = (x+ y*res_3i.x() + z*res_3i.x()*res_3i.y()) * 3;
	Eigen::Vector3f pos = Eigen::Vector3f{(float)x, (float)y, (float)z}.cwiseQuotient((res_3i-Eigen::Vector3i::Ones()).cast<float>());
	
	out[i] = pos[0];
    out[i+1] = pos[1];
    out[i+2] = pos[2];
}

__global__ void output_half_to_float(uint32_t points, uint32_t outwidth, float* density, tcnn::network_precision_t* network_output)
{   
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= points)
        return;
    density[i] = float(network_output[i * outwidth + 3]);
}

__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements,const BoundingBox aabb, const Eigen::Vector3f* __restrict__ pos, float* network_input) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    Eigen::Vector3f input = WarpPoint(pos[i], aabb);
    network_input[3*i] = input[0];
    network_input[3*i+1] = input[1];
    network_input[3*i+2] = input[2];
}

__global__ void extract_rgb_with_activation(const uint32_t n_elements, float* mlpout, Eigen::Vector3f* colors, ENerfActivation rgb_activation) 
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	mlpout += i * 4;
    Eigen::Vector3f rgb;
    rgb[0] = network_to_rgb(mlpout[0],rgb_activation);
    rgb[1] = network_to_rgb(mlpout[1],rgb_activation);
    rgb[2] = network_to_rgb(mlpout[2],rgb_activation);
    colors[i] = rgb;
}

__global__ void trans_mesh_data(const uint32_t n_elements, Eigen::Vector3f* E_verts,
                                Eigen::Vector3f* E_normals,
                                Eigen::Vector3f* E_colors,
                                float* verts,float* normals,uint8_t* colors)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    verts[3*i] = E_verts[i].x();
    verts[3*i+1] = E_verts[i].y();
    verts[3*i+2] = E_verts[i].z();
    Eigen::Vector3f n = E_normals[i].normalized();
    normals[3*i] = n.x();
    normals[3*i+1] = n.y();
    normals[3*i+2] = n.z();
    Eigen::Vector3f c = E_colors[i];
    colors[3*i] = (unsigned char)tcnn::clamp(c.x()*255.f,0.f,255.f);
    colors[3*i+1] = (unsigned char)tcnn::clamp(c.y()*255.f,0.f,255.f);
    colors[3*i+2] = (unsigned char)tcnn::clamp(c.z()*255.f,0.f,255.f);

}

__global__ void trans_indices_data(const uint32_t n_elements,uint32_t* o_indices,uint32_t* d_indices)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    d_indices[i] = o_indices[i];
}

__global__ void GenerateRays(const uint32_t nRays,const size_t mnBbox,
                            BoundingBox ObjBBox,
                            uint32_t* __restrict__ InBBoxRaysCounter,
                            FrameIdAndBbox* pBBox,Eigen::Matrix4f ObjTow,
                            MetaData* pData,float* SampleXY,float* RandColor,
                            Ray* rays,uint8_t* raysInstance, float* rgb_target,float* depth_target,
                            float* fxfycxcy,int H,int W,uint8_t ObjInstanceId,bool useDepth)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    size_t idxBox = i % mnBbox;  // 4096 => mnBboxs * numsRays_Per_Bbox
    pBBox += idxBox;
    
    const float* pixels = pData[pBBox->FrameId].pixels;
    const float* depths;
    if(useDepth)
        depths = pData[pBBox->FrameId].depth;
    const uint8_t* instance = pData[pBBox->FrameId].instance;
    const Eigen::Matrix4f* pPose = pData[pBBox->FrameId].Pose;

    int h = pBBox->h;
    int w = pBBox->w;
    
    //position
    uint32_t x = pBBox->x + uint32_t(SampleXY[2*i] * w);
    uint32_t y = pBBox->y + uint32_t(SampleXY[2*i+1] * h);;

    uint8_t instanceId = instance[y*W+x];
    //Occlusion, skip this ray
    if(instanceId != 0 && instanceId != ObjInstanceId)
        return;
    
    Eigen::Vector3f d = {(float(x) - fxfycxcy[2]) / fxfycxcy[0],
                        (float(y) - fxfycxcy[3]) / fxfycxcy[1],
                        1.0f};
    float d_norm = d.norm(); 
    
    // camera ==> world
    d = pPose->block<3,3>(0,0) * d.normalized();
    Eigen::Vector3f o = pPose->col(3).head<3>();
    // world ==> object
    d = ObjTow.block<3,3>(0,0) * d;
    o = ObjTow.block<3,3>(0,0) * o + ObjTow.col(3).head<3>();
    
    Eigen::Vector2f tminmax = ray_intersect(ObjBBox,o,d);
    if(tminmax[0] != std::numeric_limits<float>::max())
    {
        //in box
        uint32_t idx = atomicAdd(InBBoxRaysCounter,1);
        rays[idx].d = d;
        rays[idx].o = o;
        rays[idx].d_norm = d_norm;
        rays[idx].tmin = fmaxf(tminmax.x(), 0.0f);
        rays[idx].tmax = tminmax[1];
        //Object pixel
        if(instanceId != 0)
        {
            rgb_target[idx*3] = pixels[(y*W+x)*3];
            rgb_target[idx*3+1] = pixels[(y*W+x)*3+1];
            rgb_target[idx*3+2] = pixels[(y*W+x)*3+2];
            if(useDepth)
                depth_target[idx] = depths[y*W+x] * d_norm;
            else
                depth_target[idx] = 0.f;
            raysInstance[idx] = 1;
        }
        else    
        {   //Background, random color
            rgb_target[idx*3] = RandColor[idx*3];
            rgb_target[idx*3+1] = RandColor[idx*3+1];
            rgb_target[idx*3+2] = RandColor[idx*3+2];
            depth_target[idx] = 0.f;
            raysInstance[idx] = 0;
        }
    }
}

__global__ void GenerateRenderRays(const uint32_t nRays,
                                FrameIdAndBbox BBox,
                                BoundingBox ObjBBox,
                                Eigen::Matrix4f Pose,
                                Eigen::Matrix4f ObjTow,
                                Ray* rays,
                                int* RaysInBBox,
                                float* fxfycxcy)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    int w = BBox.w;

    //position
    int x = BBox.x + i % w;
    int y = BBox.y + i / w;

    Eigen::Vector3f d = {(float(x) - fxfycxcy[2]) / fxfycxcy[0],
                        (float(y) - fxfycxcy[3]) / fxfycxcy[1],
                        1.0f};
    float d_norm = d.norm(); 
    
    // c ==> w 
    d = Pose.block<3,3>(0,0) * d.normalized();
    Eigen::Vector3f o = Pose.col(3).head<3>();
    // w ==> o
    d = ObjTow.block<3,3>(0,0) * d;
    o = ObjTow.block<3,3>(0,0) * o + ObjTow.col(3).head<3>();
    
    Eigen::Vector2f tminmax = ray_intersect(ObjBBox,o,d);
    if(tminmax[0] != std::numeric_limits<float>::max())
    {
        rays[i].d = d;
        rays[i].o = o;
        rays[i].d_norm = d_norm;
        rays[i].tmin = fmaxf(tminmax.x(), 0.0f);
        rays[i].tmax = tminmax[1];
        RaysInBBox[i] = 1;
    }
    else
    {
        RaysInBBox[i] = 0;
    }
}

__global__ void GenerateRenderVideoRays(const uint32_t nRays,
                                FrameIdAndBbox BBox,
                                BoundingBox ObjBBox,
                                Eigen::Matrix4f Toc,
                                Ray* rays,
                                int* RaysInBBox,
                                float* fxfycxcy)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    int w = BBox.w;
    //position
    int x = BBox.x + i % w;
    int y = BBox.y + i / w;
    Eigen::Vector3f d = {(float(x) - fxfycxcy[2]) / fxfycxcy[0],
                        (float(y) - fxfycxcy[3]) / fxfycxcy[1],
                        1.0f};
    float d_norm = d.norm(); 
    
    // c ==> o 
    d = Toc.block<3,3>(0,0) * d.normalized();
    Eigen::Vector3f o = Toc.col(3).head<3>();

    Eigen::Vector2f tminmax = ray_intersect(ObjBBox,o,d);
    if(tminmax[0] != std::numeric_limits<float>::max())
    {
        rays[i].d = d;
        rays[i].o = o;
        rays[i].d_norm = d_norm;
        rays[i].tmin = fmaxf(tminmax.x(), 0.0f);
        rays[i].tmax = tminmax[1];
        RaysInBBox[i] = 1;
    }
    else
    {
        RaysInBBox[i] = 0;
    }
}

__global__ void GenerateInputPoints(const uint32_t nRays,const uint32_t nSampleNum,
                        BoundingBox Bbox,Ray* rays, float* PointsInput,float* SamplesDistances,float* RandDt)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;
    Eigen::Vector3f o = rays[i].o;
    Eigen::Vector3f d = rays[i].d;

    float tmin = rays[i].tmin;
    float tmax = rays[i].tmax;
    float dt = (tmax - tmin) / float(nSampleNum);
    float t = 0;
    RandDt += i * nSampleNum;
    
    Eigen::Vector3f point;
    size_t base = i * nSampleNum * 3;
    for(int n=0;n < nSampleNum;n++)
    {
        t = tmin + dt * (float(n) + RandDt[n]);

        point = o + t * d;
        //transform to [0,1]
        point = WarpPoint(point,Bbox);
        //printf("x: %f %f %f\n",point.x(),point.y(),point.z());
        PointsInput[base + n*3] = point.x();
        PointsInput[base + n*3 + 1] = point.y();
        PointsInput[base + n*3 + 2] = point.z();
        SamplesDistances[i*nSampleNum + n] = t;

    }
    
    //Importance sampling, not used, for reference only-------------------------------------------
    /* //Only half of the uniform sample points are generated
    uint32_t HalfnSampleNum = nSampleNum / 2;
    float dt = (tmax - tmin) / float(HalfnSampleNum);
    float t = 0;
    RandDt += i * nSampleNum;
    
    Eigen::Vector3f point;
    size_t base = i * nSampleNum * 3;
    for(int n=0;n < HalfnSampleNum;n++)
    {
        t = tmin + dt * (n + RandDt[n]);

        point = o + t * d;
        point = WarpPoint(point,Bbox);
        //printf("x: %f %f %f\n",point.x(),point.y(),point.z());
        PointsInput[base + n*3] = point.x();
        PointsInput[base + n*3 + 1] = point.y();
        PointsInput[base + n*3 + 2] = point.z();
        SamplesDistances[i*nSampleNum + n] = t;

    } */

}

__global__ void GenerateRenderInputPoints(const uint32_t nRays,const uint32_t nSampleNum,
                        BoundingBox Bbox,Ray* rays,int* RaysInBBox, float* PointsInput,float* SamplesDistances,float* RandDt)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;
    if(RaysInBBox[i] == 0)
    {
        return;
    }

    float tmin = rays[i].tmin;
    float tmax = rays[i].tmax;
    Eigen::Vector3f d = rays[i].d;
    Eigen::Vector3f o = rays[i].o;
    float dt = (tmax - tmin) / float(nSampleNum);
    float t = 0;
    RandDt += i * nSampleNum;
    Eigen::Vector3f point;
    size_t base = i * nSampleNum * 3;

    for(int n=0;n<nSampleNum;n++)
    {
        t = tmin + dt * (float(n) + RandDt[n]);

        point = o + t * d;
        point = WarpPoint(point,Bbox);
        //printf("x: %f %f %f\n",point.x(),point.y(),point.z());
        PointsInput[base + n*3] = point.x();
        PointsInput[base + n*3 + 1] = point.y();
        PointsInput[base + n*3 + 2] = point.z();
        SamplesDistances[i*nSampleNum + n] = t;
    }
}

//Importance sampling, not used, for reference only-------------------------------------------
/* 
__global__ void InverseTransformSampling(const uint32_t nRays,const uint32_t nSampleNum,
                            const uint32_t outwidth,
                            ENerfActivation density_activation,
                            BoundingBox Bbox,
                            Ray* rays,
                            float* PointsInput,
                            float* SamplesDistances,
                            float* RandUniform,
                            tcnn::network_precision_t* RgbSigmaOutput
                            )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    RgbSigmaOutput += i * nSampleNum * outwidth; //+outwidth
    PointsInput += i * nSampleNum * 3;    //+3
    SamplesDistances += i * nSampleNum;       //+1
    RandUniform += i * nSampleNum; //Use the second half
    rays += i;

    uint32_t HalfnSampleNum = nSampleNum / 2;
    float cdf[HalfSampleNum_plus_one];
    float distances[HalfSampleNum_plus_one];
    cdf[0] = 0.0f;
    distances[0] = rays[0].tmin; //tmin

    float T = 1.f;
    float last_distance = 0.0f;
    float dt = 0.0f;
    float weight_sum = 0.0f;
    
    for(int j=0;j < HalfnSampleNum;++j)
    {
        const tcnn::vector_t<tcnn::network_precision_t, 4> local_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)RgbSigmaOutput;
        float density = network_to_density(float(local_output[3]), density_activation);
        
        //accumulate
        dt = SamplesDistances[0] - last_distance;
        last_distance = SamplesDistances[0];
        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;

        weight_sum += weight;
        cdf[j+1] = weight_sum;
        distances[j+1] = last_distance;

		// prepare next's T
		T *= (1.f - alpha);

        // take the next
		RgbSigmaOutput += outwidth;
        SamplesDistances += 1;
    }

    //Generate half of fine sample points
    RandUniform += HalfnSampleNum;
    for(int j=0;j < HalfnSampleNum;++j)
    {
        float randcdf = RandUniform[j] * weight_sum;
        uint32_t idx = lower_bound(cdf,HalfnSampleNum+1,randcdf) - 1;
        //                *       
        //               /|     
        //              / |
        //            /   |
        //  randcdf  *    |   
        //         /      |
        //       /        |
        //      *    *    * 
        //     idx       idx+1 
        //        t_values
        float t_new = distances[idx] + (distances[idx+1] - distances[idx]) * (randcdf - cdf[idx]) / (cdf[idx + 1] -cdf[idx]);
        SamplesDistances[j] = t_new;
    }

    SamplesDistances -= HalfnSampleNum;
    /* if(i == 2000)
    {
        for(int a=0;a<nSampleNum;a++)
            printf("%f ",SamplesDistances[a]);
        printf("\n");
    } 
    //sort SamplesDistances
    //selection_sort(SamplesDistances,nSampleNum);
    selection_sort_nv(SamplesDistances,0,nSampleNum-1);
    QuickSort(SamplesDistances,0,nSampleNum-1);
    
    //Generate points
    float t;
    for(int n=0;n <nSampleNum;n++)
    {
        t = SamplesDistances[n];

        Eigen::Vector3f point = rays[0].o + t * rays[0].d;
        point = WarpPoint(point,Bbox);
        //printf("x: %f %f %f\n",point.x(),point.y(),point.z());
        PointsInput[n*3] = point.x();
        PointsInput[n*3 + 1] = point.y();
        PointsInput[n*3 + 2] = point.z();
        
    }
}
 */
//------------------------------------------------------------------------------------------------

__global__ void VolumeRender(const uint32_t nRays,const uint32_t nSamples,
                            const uint32_t outwidth,
                            BoundingBox box,
                            ENerfActivation rgb_activation,
                            ENerfActivation density_activation,
                            tcnn::network_precision_t* RgbSigmaOutput,
                            float* PointsInput,
                            float* SamplesDistances,
                            Ray* rays,
                            float* RandomColor,
                            uint32_t* InBboxRaysCounter,
                            float* rgb_rays,
                            float* depth_rays,
                            float* mask_rays
                            )
{
    const uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    RgbSigmaOutput += i * nSamples * outwidth; //+outwidth
    PointsInput += i * nSamples * 3;    //+3
    SamplesDistances += i * nSamples;       //+1
    rays += i;
    //Align with the previous fill over
    RandomColor += (i % *InBboxRaysCounter) * 3;

    float T = 1.f;
    float EPSILON = 1e-4f;

    Eigen::Array3f rgb_ray = {0.0f,0.0f,0.0f};
    float depth_ray = 0.f;
    
    uint32_t numsteps = 0;
    float cur_distance = 0.0f;
    float last_distance = 0.0f;
    
    for(;numsteps < nSamples;++numsteps)
    {
        if(T < EPSILON)
            break;

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)RgbSigmaOutput;
        Eigen::Array3f rgb;
        rgb[0] = network_to_rgb(float(local_output[0]),rgb_activation);
        rgb[1] = network_to_rgb(float(local_output[1]),rgb_activation);
        rgb[2] = network_to_rgb(float(local_output[2]),rgb_activation);
        
        cur_distance = SamplesDistances[0];
        float dt = cur_distance - last_distance;
        
        float density = network_to_density(float(local_output[3]), density_activation);

        //accumulate
        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		// accumulate the rgb color by weighted sum
		rgb_ray += weight * rgb;
		// accumulate the depth by weighted sum
		depth_ray += weight * cur_distance;

		// prepare next's T
		T *= (1.f - alpha);

        // take the next
		RgbSigmaOutput += outwidth;
		PointsInput += 3;
        SamplesDistances += 1;
        last_distance = cur_distance;

    }
    
    const Eigen::Array3f background_color = {RandomColor[0], RandomColor[1], RandomColor[2]};
    rgb_ray += T * background_color;
    rgb_rays[i*3] = rgb_ray[0];
    rgb_rays[i*3+1] = rgb_ray[1];
    rgb_rays[i*3+2] = rgb_ray[2];
    depth_rays[i] = depth_ray;
    mask_rays[i] = 1 - T;
    
}

__global__ void VolumeRenderGradient_No_Compacted(const uint32_t nRays,const uint32_t nSamples,
                            const uint32_t outwidth,
                            BoundingBox box,
                            ENerfActivation rgb_activation,
                            ENerfActivation density_activation,
                            float loss_scale,
                            tcnn::network_precision_t* RgbSigmaOutput,
                            float* PointsInput,
                            float* SamplesDistances,
                            Ray* rays,
                            uint8_t* raysInstance,
                            float* Target,
                            float* TargetDepth,
                            float* rgb_rays,
                            float* depth_rays,
                            float* mask_rays,
                            tcnn::network_precision_t* dloss_dout,
                            float* __restrict__ loss_cout)
{

    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    RgbSigmaOutput += i * nSamples * outwidth; //+outwidth
    PointsInput += i * nSamples * 3;    //+3
    SamplesDistances += i * nSamples;       //+1
    rays += i;
    raysInstance += i;
    Target += i * 3;
    TargetDepth += i;
    rgb_rays += i * 3;
    depth_rays += i;
    mask_rays += i;
    dloss_dout += i * nSamples * outwidth;

    //rgb
    Eigen::Array3f rgbtarget(Target[0],Target[1],Target[2]);
    Eigen::Array3f rgb_ray2 = Eigen::Array3f::Zero();    
    Eigen::Array3f rgb_ray(rgb_rays[0],rgb_rays[1],rgb_rays[2]);  //VolumeRender result

    LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray);
    
    // average r,g,b loss to the mean rgb loss
	float mean_loss = lg.loss.mean();
	
    //depth
    float depthtarget = TargetDepth[0];
    float depth_ray2 = 0.0f;
    float depth_ray = depth_rays[0];
    //L1 loss
    float dloss_ddepth = 0.0f;
    const float depth_supervision_lambda = 0.5f;
    if(depthtarget > 0.0f)
        dloss_ddepth = depth_supervision_lambda * (depth_ray - depthtarget >= 0.f ? 1.0f : -1.0f); 
        
    //mask 1.0f or 0.0f;
    float mask_ray = mask_rays[0];

    //for recording
    if(raysInstance[0]==1)
        loss_cout[i] = mean_loss + dloss_ddepth * (depth_ray - depthtarget) + (1 - mask_ray);
    else
        loss_cout[i] = mean_loss + mask_ray;

    loss_scale /= nRays;
    float T = 1.f;
    float EPSILON = 1e-4f;
    uint32_t numsteps = 0;
    float cur_distance = 0.0f;
    float last_distance = 0.0f;

    for(;numsteps < nSamples;++numsteps)
    {
        if(T < EPSILON)
            break;

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)RgbSigmaOutput;
        Eigen::Array3f rgb;
        rgb[0] = network_to_rgb(float(local_output[0]),rgb_activation);
        rgb[1] = network_to_rgb(float(local_output[1]),rgb_activation);
        rgb[2] = network_to_rgb(float(local_output[2]),rgb_activation);
        
        cur_distance = SamplesDistances[0];
        float dt = cur_distance - last_distance;
        float density = network_to_density(float(local_output[3]), density_activation);

        //accumulate
        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		// accumulate the rgb color by weighted sum
		rgb_ray2 += weight * rgb;
		// accumulate the depth by weighted sum
		depth_ray2 += weight * cur_distance;
		// prepare next's T
		T *= (1.f - alpha);

        //gradient
        const Eigen::Array3f suffix = rgb_ray - rgb_ray2;
		const Eigen::Array3f dloss_by_drgb = weight * lg.gradient;
        tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;
        local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_output[0], rgb_activation)); 
        local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_output[1], rgb_activation));
        local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_output[2], rgb_activation));
        
        //density
        float density_derivative = network_to_density_derivative(float(local_output[3]), density_activation);
        const float depth_suffix = depth_ray - depth_ray2;
		const float depth_supervision = dloss_ddepth * (T * cur_distance - depth_suffix);
        const float dmask_ddensity = 1 - mask_ray;
        const float mask_supervision_lambda = 0.5f;
        float dloss_dmask = mask_supervision_lambda * (mask_ray >= 1 ? 1.0f : -1.0f);
        float dloss_by_dmlp;

        if(raysInstance[0] == 1)
        {   //object ray
            dloss_by_dmlp = density_derivative *
            dt * (lg.gradient.matrix().dot((T * rgb - suffix).matrix()) + depth_supervision + dloss_dmask * dmask_ddensity);
        }
        else
        {   //background ray
            dloss_dmask = mask_supervision_lambda * (mask_ray >= 0 ? 1.0f : -1.0f);
            //dloss_by_dmlp = density_derivative * dt * lg.gradient.matrix().dot((T * rgb - suffix).matrix()) + density_derivative * 0.01f;
            dloss_by_dmlp =  density_derivative * dt * dloss_dmask * dmask_ddensity + density_derivative * 0.01f;
        }

        local_dL_doutput[3] = loss_scale * dloss_by_dmlp;
        // write gradient (from loss to the network's output layer)
		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_dout = local_dL_doutput;

        // take the next
		RgbSigmaOutput += outwidth;
        dloss_dout += outwidth;
		PointsInput += 3;
        SamplesDistances += 1;
        last_distance = cur_distance;  
    }
}

//unavailable, for reference only
__global__ void VolumeRenderGradient(const uint32_t nRays,const uint32_t nSamples,
                            const uint32_t outwidth,
                            BoundingBox box,
                            ENerfActivation rgb_activation,
                            ENerfActivation density_activation,
                            tcnn::default_rng_t rng,
                            float loss_scale,
                            tcnn::network_precision_t* RgbSigmaOutput,
                            float* PointsInput,
                            float* CompatcedPointsInput,
                            uint32_t* __restrict__ CompatcedPointsCounter,
                            float* SamplesDistances,
                            Ray* rays,
                            float* Target,
                            float* rgb_rays,
                            float* depth_rays,
                            tcnn::network_precision_t* dloss_dout,
                            float* __restrict__ loss_cout)
{

    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    RgbSigmaOutput += i * nSamples * outwidth; //+outwidth
    PointsInput += i * nSamples * 3;    //+3
    SamplesDistances += i * nSamples;       //+1
    rays += i;

    float T = 1.f;
    float EPSILON = 1e-4f;

    Eigen::Array3f rgb_ray = {0.0f,0.0f,0.0f};
    float depth_ray = 0.f;

    Eigen::Vector3f ray_o = rays->o;

    uint32_t numsteps = 0;

    float last_distance = 0.0f;
    for(;numsteps < nSamples;++numsteps)
    {
        if(T < EPSILON)
            break;

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)RgbSigmaOutput;
        Eigen::Array3f rgb;
        rgb[0] = network_to_rgb(float(local_output[0]),rgb_activation);
        rgb[1] = network_to_rgb(float(local_output[1]),rgb_activation);
        rgb[2] = network_to_rgb(float(local_output[2]),rgb_activation);
        
        Eigen::Vector3f point;
        point[0] = PointsInput[0];
        point[1] = PointsInput[1];
        point[2] = PointsInput[2];
        point = UnWarpPoint(point,box);

        float dt = SamplesDistances[0] - last_distance;
        last_distance = SamplesDistances[0];

        //not * cos
        float cur_depth = (point - ray_o).norm();

        float density = network_to_density(float(local_output[3]), density_activation);

        //accumulate

        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		// accumulate the rgb color by weighted sum
		rgb_ray += weight * rgb;
		// accumulate the depth by weighted sum
		depth_ray += weight * cur_depth;
		// prepare next's T
		T *= (1.f - alpha);

        // take the next
		RgbSigmaOutput += outwidth;
		PointsInput += 3;
        SamplesDistances += 1;
        
    }
    
    const Eigen::Array3f background_color = {rng.next_float(), rng.next_float(), rng.next_float()};
    rgb_ray += T * background_color;    //VolumeRender result

    // Step again, this time computing loss
    RgbSigmaOutput -= numsteps * outwidth; //+outwidth
    PointsInput -= numsteps * 3;    //+3
    SamplesDistances -= numsteps;       //+1
    uint32_t compacted_base = atomicAdd(CompatcedPointsCounter, numsteps);
    CompatcedPointsInput += compacted_base * 3;
    dloss_dout += compacted_base * outwidth;
    Target += i * 3;

    Eigen::Array3f rgbtarget(Target[0],Target[1],Target[2]);
    Eigen::Array3f rgb_ray2 = Eigen::Array3f::Zero();    

    //Eigen::Array3f rgb_ray(rgb_rays[0],rgb_rays[1],rgb_rays[2]);  

    //depth
    //float depth_ray = 0.0f;

    LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray);

    // average r,g,b loss to the mean rgb loss
	float mean_loss = lg.loss.mean();
	if (loss_cout) {
		// loss contribution
		// divided by `n_rays` is to average over all rays
		// this is to output, not to optimize
		loss_cout[i] = mean_loss;
	}

    loss_scale /= nRays;

    T = 1.f;
    EPSILON = 1e-4f;
    last_distance = 0.0f;
    for(uint32_t j=0;j < numsteps;++j)
    {
        if(T < EPSILON)
            break;

        CompatcedPointsInput[0] = PointsInput[0];
        CompatcedPointsInput[1] = PointsInput[1];
        CompatcedPointsInput[2] = PointsInput[2];

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)RgbSigmaOutput;
        Eigen::Array3f rgb;
        rgb[0] = network_to_rgb(float(local_output[0]),rgb_activation);
        rgb[1] = network_to_rgb(float(local_output[1]),rgb_activation);
        rgb[2] = network_to_rgb(float(local_output[2]),rgb_activation);
        
        float dt = SamplesDistances[0] - last_distance;
        last_distance = SamplesDistances[0];
        float density = network_to_density(float(local_output[3]), density_activation);

        //accumulate
        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		// accumulate the rgb color by weighted sum
		rgb_ray2 += weight * rgb;
		// accumulate the depth by weighted sum
		//depth_ray += weight * cur_depth;
		// prepare next's T
		T *= (1.f - alpha);

        //gradient
        const Eigen::Array3f suffix = rgb_ray - rgb_ray2;
		const Eigen::Array3f dloss_by_drgb = weight * lg.gradient;
        tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;
        local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_output[0], rgb_activation)); 
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_output[1], rgb_activation));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_output[2], rgb_activation));

        float density_derivative = network_to_density_derivative(float(local_output[3]), density_activation);

        //const float depth_suffix = depth_ray - depth_ray2;
		//const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);
        float dloss_by_dmlp = density_derivative * (
			dt * (lg.gradient.matrix().dot((T * rgb - suffix).matrix())));

        local_dL_doutput[3] = loss_scale * dloss_by_dmlp;

        // write gradient (from loss to the network's output layer)
		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_dout = local_dL_doutput;

        // take the next
		RgbSigmaOutput += outwidth;
        dloss_dout += outwidth;
		PointsInput += 3;
        CompatcedPointsInput += 3;
        SamplesDistances += 1;
    }
}

__global__ void VolumeRender_Render(const uint32_t nRays,const uint32_t nSamples,
                            const uint32_t outwidth,
                            BoundingBox box,
                            ENerfActivation rgb_activation,
                            ENerfActivation density_activation,
                            float BackgroundColor,
                            float* RgbSigmaOutput,
                            float* PointsInput,
                            float* SamplesDistances,
                            Ray* rays,
                            int* RaysInBBox,
                            float* rgb_rays,
                            float* depth_rays,
                            float* mask_rays
                            )
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= nRays)
        return;

    if(RaysInBBox[i] == 0)
    {   //not in BBox
        rgb_rays[i*3] = BackgroundColor;
        rgb_rays[i*3+1] = BackgroundColor;
        rgb_rays[i*3+2] = BackgroundColor;
        depth_rays[i] = 0.0f;
        mask_rays[i] = 0.0f;
        return;
    }

    RgbSigmaOutput += i * nSamples * outwidth; //+outwidth
    PointsInput += i * nSamples * 3;    //+3
    SamplesDistances += i * nSamples;       //+1
    rays += i;

    float T = 1.f;
    float EPSILON = 1e-4f;
    Eigen::Array3f rgb_ray = {0.0f,0.0f,0.0f};
    float depth_ray = 0.f;
    float d_norm = rays->d_norm;
    uint32_t numsteps = 0;
    float cur_distance = 0.0f;
    float last_distance = 0.0f;

    for(;numsteps < nSamples;++numsteps)
    {
        if(T < EPSILON)
            break;

        const tcnn::vector_t<float, 4> local_output = *(tcnn::vector_t<float, 4>*)RgbSigmaOutput;
        Eigen::Array3f rgb;
        rgb[0] = network_to_rgb(float(local_output[0]),rgb_activation);
        rgb[1] = network_to_rgb(float(local_output[1]),rgb_activation);
        rgb[2] = network_to_rgb(float(local_output[2]),rgb_activation);
        
        cur_distance = SamplesDistances[0];
        float dt = cur_distance - last_distance;
    
        float density = network_to_density(float(local_output[3]), density_activation);

        //accumulate
        const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		// accumulate the rgb color by weighted sum
		rgb_ray += weight * rgb;
		// accumulate the depth by weighted sum
		depth_ray += weight * cur_distance;
		// prepare next's T
		T *= (1.f - alpha);

        // take the next
		RgbSigmaOutput += outwidth;
		PointsInput += 3;
        SamplesDistances += 1;
        last_distance = cur_distance;
    }
    
    const Eigen::Array3f background_color = {BackgroundColor,BackgroundColor,BackgroundColor};
    rgb_ray += T * background_color;
    if(1.0f - T > 0.5f)
    {
        rgb_rays[i*3] = rgb_ray[0];
        rgb_rays[i*3+1] = rgb_ray[1];
        rgb_rays[i*3+2] = rgb_ray[2];
        depth_rays[i] = depth_ray / d_norm;
        mask_rays[i] = 1.0f;
    }
    else
    {
        rgb_rays[i*3] = BackgroundColor;
        rgb_rays[i*3+1] = BackgroundColor;
        rgb_rays[i*3+2] = BackgroundColor;
        depth_rays[i] = 0.0f;
        mask_rays[i] = 0.0f;
    } 
}

__global__ void SumLoss(const uint32_t nRays,float* loss,float* temp) 
{
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nRays)
        return;
    s[tid] = loss[i];
    __syncthreads();
   
    for (int step = blockDim.x / 2; step > 0; step >>= 1) 
    {
        if (tid < step) 
        {
            s[tid] += s[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
    temp[blockIdx.x] = s[0];
    }
}

//------------------------------------------------------------------
//--------------------------C++ function----------------------------
//------------------------------------------------------------------

NeRF_Model::NeRF_Model(int id, int GPUid, const BoundingBox& BBox, const Eigen::Matrix4f& ObjTow, uint8_t InstanceId)
{
    mId = id;
    mGPUid = GPUid;
    mBoundingBox = BBox;
    mObjTow = ObjTow;
    mInstanceId = InstanceId;
    cudaSetDevice(mGPUid);
    mNetworkConfig = ClassNetworkConfig;
    CUDA_CHECK_THROW(cudaStreamCreate(&mpTrainStream));
    CUDA_CHECK_THROW(cudaStreamCreate(&mpInferenceStream));
}

bool NeRF_Model::ReadNetworkConfig(const string config_path)
{
    json config;
    std::ifstream file(config_path);
    if(!file)
    {
        cerr << "config file error..."<<endl;
        return false;
    }
	config = json::parse(file, nullptr, true, true);
    ClassNetworkConfig = config;
    return true;
}

bool NeRF_Model::ResetNetwork()
{
    cudaSetDevice(mGPUid);

    json config = mNetworkConfig;
    json& encoding_config = config["encoding"];
	json& loss_config = config["loss"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];
    //loss 
    loss_config["otype"] = "L2";

    //encoding
    mnFeatureDims = encoding_config.value("n_features_per_level", 2u);
    mnNumLevels = encoding_config.value("n_levels",16u);
    mnBaseResolution = encoding_config.value("base_resolution",0);
    mnLog2HashmapSize = encoding_config.value("log2_hashmap_size",15);

    //level scale
    float desired_resolution = 2048.0f;
    mfLevelScale = std::exp(std::log(desired_resolution / (float)mnBaseResolution) / (mnNumLevels-1));

    cout<<"GridEncoding:"
    <<"Nmin=" << mnBaseResolution
    <<" b=" << mfLevelScale
    <<" F=" << mnFeatureDims
    <<" T=2^" << mnLog2HashmapSize
    <<" L=" << mnNumLevels <<endl;

    mpLoss.reset(tcnn::create_loss<precision_t>(loss_config));
    mpOptimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
    // point(x,y,z)(3) --> (density,rgb)(4)
    mpNetwork = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(3,4,encoding_config,network_config);
    mpEncoding = mpNetwork->encoding();
    size_t nEncodingParms = mpEncoding->n_params();

    cout<<"Model: "<< mpEncoding->input_width()
    <<"--[" << string(encoding_config["otype"])
    <<"]-->" <<mpEncoding->padded_output_width()
    <<"--[" << string(network_config["otype"])
    <<"(n_neurons=" << (int)network_config["n_neurons"] << "|n_hidden_layers=" << ((int)network_config["n_hidden_layers"]) << ")"
    << "]-->"
    <<mpNetwork->output_width()
	<< "(density=1,color=3)"<<endl;

    size_t nNetworkParams = mpNetwork->n_params() - nEncodingParms;
    cout << " total_encoding_params=" << nEncodingParms << " total_network_params=" << nNetworkParams <<endl;

    mpTrainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(mpNetwork, mpOptimizer, mpLoss);
	mnTrainingStep = 0;

    //random
    m_rng = tcnn::default_rng_t{m_seed};

    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

void NeRF_Model::AllocateBatchWorkspace(cudaStream_t pStream,const uint32_t PaddedOutputWidth)
{
    if(mbBatchDataAllocated)
        return;
    
    //random number generator
    curandStatus_t cst;
    cst = curandCreateGenerator(&mGen, CURAND_RNG_PSEUDO_XORWOW);
    assert(CURAND_STATUS_SUCCESS == cst);
    curandSetStream(mGen,pStream);
    
    int samplenum = mnRaysPerBatch * 2;     //rays * 2

    mBatchMemory.SampleXY.resize(samplenum);
    mBatchMemory.Rays.resize(mnRaysPerBatch);
    mBatchMemory.RaysInstance.resize(mnRaysPerBatch);
    mBatchMemory.RandColors.resize(mnRaysPerBatch * 3);
    mLossMemory.resize(mnRaysPerBatch);
    mSumLossMemory.resize(16);
    mnPaddedOutWidth = PaddedOutputWidth;
    auto scratch = tcnn::allocate_workspace_and_distribute<
    float,
    float,
    float,
    float,
    float,
    tcnn::network_precision_t,
    tcnn::network_precision_t,
    float,
    float,
    float,
    float,
    uint32_t,
    uint32_t
    >(
        pStream,&mBatchAlloc,
        3 * mnRaysPerBatch,
        mnRaysPerBatch,
        3 * mnBatchSize,
        3 * mnBatchSize,
        mnSampleNum * mnRaysPerBatch,
        PaddedOutputWidth * mnBatchSize,
        PaddedOutputWidth * mnBatchSize,
        3 * mnRaysPerBatch,
        mnRaysPerBatch,
        mnRaysPerBatch,
        mnSampleNum * mnRaysPerBatch,
        1,
        1
    );
    
    float* Target = std::get<0>(scratch);
    float* TargetDepth = std::get<1>(scratch);
    float* PointsInput = std::get<2>(scratch);  //R G B * Rays
    float* CompactedPointsInput = std::get<3>(scratch);  //R G B * Rays
    float* SamplesDistances = std::get<4>(scratch);   
    tcnn::network_precision_t* output = std::get<5>(scratch);   
    tcnn::network_precision_t* dloss_dout = std::get<6>(scratch);  
    float* rgb_rays = std::get<7>(scratch);
    float* depth_rays = std::get<8>(scratch);
    float* mask_rays = std::get<9>(scratch);
    float* RandDt = std::get<10>(scratch);
    uint32_t* CompactedPointsCounter = std::get<11>(scratch);
    uint32_t* InBboxRaysCounter = std::get<12>(scratch);

    //set pointer and size
    mBatchMemory.Target.set(Target,3,mnRaysPerBatch);
    mBatchMemory.TargetDepth.set(TargetDepth,1,mnRaysPerBatch);
    mBatchMemory.PointsInput.set(PointsInput,3,mnBatchSize);
    mBatchMemory.CompactedPointsInput.set(CompactedPointsInput,3,mnBatchSize);
    mBatchMemory.SamplesDistances.set(SamplesDistances,mnSampleNum,mnRaysPerBatch);
    mBatchMemory.RgbSigmaOutput.set(output,mnPaddedOutWidth,mnBatchSize);
    mBatchMemory.dloss_dout.set(dloss_dout,mnPaddedOutWidth,mnBatchSize);
    mBatchMemory.rgb_rays.set(rgb_rays,3,mnRaysPerBatch);
    mBatchMemory.depth_rays.set(depth_rays,1,mnRaysPerBatch);
    mBatchMemory.mask_rays.set(mask_rays,1,mnRaysPerBatch);
    mBatchMemory.RandDt.set(RandDt,mnSampleNum,mnRaysPerBatch);
    mBatchMemory.CompactedPointsCounter.set(CompactedPointsCounter,1,1);
    mBatchMemory.InBboxRaysCounter.set(InBboxRaysCounter,1,1);
    mbBatchDataAllocated = true;

    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
}

void NeRF_Model::GenerateBatch(cudaStream_t pStream,std::shared_ptr<NeRF_Dataset> pTrainData)
{
    //sample rays
    curandGenerateUniform(mGen,mBatchMemory.SampleXY.data(),mnRaysPerBatch * 2);
    //Random background color
    curandGenerateUniform(mGen,mBatchMemory.RandColors.data(),mnRaysPerBatch * 3);
    mBatchMemory.InBboxRaysCounter.memset_async(pStream,0);

    //Average distribution of sampling points
    tcnn::linear_kernel(GenerateRays,0,pStream,
        mnRaysPerBatch,
        mnBbox,
        mBoundingBox,
        mBatchMemory.InBboxRaysCounter.data(),
        mFrameIdAndBboxMemory.data(),
        mObjTow,
        pTrainData->mMetadataMemory.data(),
        mBatchMemory.SampleXY.data(),
        mBatchMemory.RandColors.data(),
        mBatchMemory.Rays.data(),
        mBatchMemory.RaysInstance.data(),
        mBatchMemory.Target.data(),
        mBatchMemory.TargetDepth.data(),
        pTrainData->mIntrinsicsMemory.data(),
        pTrainData->H,
        pTrainData->W,
        mInstanceId,
        mbUseDepth
    );

    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    fill_rollover_rays<<<tcnn::n_blocks_linear(mnRaysPerBatch), tcnn::n_threads_linear, 0, pStream>>>(
		mnRaysPerBatch, mBatchMemory.InBboxRaysCounter.data(),
        mBatchMemory.Rays.data(),mBatchMemory.RaysInstance.data(),
        mBatchMemory.Target.data(),mBatchMemory.TargetDepth.data()
	);

    //Generate rand step 
    curandGenerateUniform(mGen,mBatchMemory.RandDt.data(),mnRaysPerBatch * mnSampleNum);
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    tcnn::linear_kernel(GenerateInputPoints,0,pStream,
        mnRaysPerBatch,
        mnSampleNum,
        mBoundingBox,
        mBatchMemory.Rays.data(),
        mBatchMemory.PointsInput.data(),
        mBatchMemory.SamplesDistances.data(),
        mBatchMemory.RandDt.data()
        );

    //Importance sampling, not used, for reference only-------------------------------------------
    /* 
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
    //inverse transform sampling
    const uint32_t padded_output_width = mpNetwork->padded_output_width();
    mpNetwork->inference_mixed_precision_impl(pStream, mBatchMemory.PointsInput, mBatchMemory.RgbSigmaOutput, false);
    
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
    tcnn::linear_kernel(InverseTransformSampling,0,pStream,
        mnRaysPerBatch,
        mnSampleNum,
        padded_output_width,
        mDensityActivation,
        mBoundingBox,
        mBatchMemory.Rays.data(),
        mBatchMemory.PointsInput.data(),
        mBatchMemory.SamplesDistances.data(),
        mBatchMemory.RandDt.data(),
        mBatchMemory.RgbSigmaOutput.data()
        );
    */
}

void NeRF_Model::Step(cudaStream_t pStream)
{
    const uint32_t padded_output_width = mpNetwork->padded_output_width();

    BatchData& Batch = mBatchMemory;
    mpNetwork->inference_mixed_precision_impl(pStream, Batch.PointsInput, Batch.RgbSigmaOutput, false);

    m_rng.advance();
    Batch.CompactedPointsCounter.memset_async(pStream,0);
    //Batch.dloss_dout.memset_async(pStream,0);
    //VolumeRender and Gradient
    tcnn::linear_kernel(VolumeRenderGradient,0,pStream,
        mnRaysPerBatch,
        mnSampleNum,
        padded_output_width,
        mBoundingBox,
        mRgbActivation,
        mDensityActivation,
        m_rng,
        mLoss_Scale,
        Batch.RgbSigmaOutput.data(),
        Batch.PointsInput.data(),
        Batch.CompactedPointsInput.data(),
        Batch.CompactedPointsCounter.data(),
        Batch.SamplesDistances.data(),
        Batch.Rays.data(),
        Batch.Target.data(),
        Batch.rgb_rays.data(),
        Batch.depth_rays.data(),
        Batch.dloss_dout.data(),
        mLossMemory.data()
        );

    fill_rollover_and_rescale<tcnn::network_precision_t><<<tcnn::n_blocks_linear(mnBatchSize * padded_output_width), tcnn::n_threads_linear, 0, pStream>>>(
		mnBatchSize, padded_output_width, Batch.CompactedPointsCounter.data(),Batch.dloss_dout.data()
	);

    fill_rollover<float><<<tcnn::n_blocks_linear(mnBatchSize * 3), tcnn::n_threads_linear, 0, pStream>>>(
		mnBatchSize, 3, Batch.CompactedPointsCounter.data(),Batch.CompactedPointsInput.data()
	);

    {
        auto ctx = mpNetwork->forward(pStream,Batch.CompactedPointsInput,&Batch.RgbSigmaOutput,false,false);
        //backward
        mpNetwork->backward(pStream, *ctx, Batch.CompactedPointsInput, Batch.RgbSigmaOutput, Batch.dloss_dout, nullptr, false, tcnn::EGradientMode::Overwrite);
    }
}

void NeRF_Model::Step_No_Compacted(cudaStream_t pStream)
{
    const uint32_t padded_output_width = mpNetwork->padded_output_width();
    BatchData& Batch = mBatchMemory;
    {
        auto ctx = mpNetwork->forward(pStream,Batch.PointsInput,&Batch.RgbSigmaOutput,false,false);

        //VolumeRender
        tcnn::linear_kernel(VolumeRender,0,pStream,
        mnRaysPerBatch,
        mnSampleNum,
        padded_output_width,
        mBoundingBox,
        mRgbActivation,
        mDensityActivation,
        Batch.RgbSigmaOutput.data(),
        Batch.PointsInput.data(),
        Batch.SamplesDistances.data(),
        Batch.Rays.data(),
        Batch.RandColors.data(),
        Batch.InBboxRaysCounter.data(),
        Batch.rgb_rays.data(),
        Batch.depth_rays.data(),
        Batch.mask_rays.data()
        );
        
        Batch.dloss_dout.memset_async(pStream,0);
        //Gradient
        tcnn::linear_kernel(VolumeRenderGradient_No_Compacted,0,pStream,
        mnRaysPerBatch,
        mnSampleNum,
        padded_output_width,
        mBoundingBox,
        mRgbActivation,
        mDensityActivation,
        mLoss_Scale,
        Batch.RgbSigmaOutput.data(),
        Batch.PointsInput.data(),
        Batch.SamplesDistances.data(),
        Batch.Rays.data(),
        Batch.RaysInstance.data(),
        Batch.Target.data(),
        Batch.TargetDepth.data(),
        Batch.rgb_rays.data(),
        Batch.depth_rays.data(),
        Batch.mask_rays.data(),
        Batch.dloss_dout.data(),
        mLossMemory.data()
        );
        
        SumLoss<<<16,256,0,pStream>>>(mnRaysPerBatch,mLossMemory.data(),mSumLossMemory.data());
        //backward
        mpNetwork->backward(pStream, *ctx, Batch.PointsInput, Batch.RgbSigmaOutput, Batch.dloss_dout, nullptr, false, tcnn::EGradientMode::Overwrite);
    }

}

void NeRF_Model::UpdateFrameIdAndBbox(const std::vector<FrameIdAndBbox>& FrameIdBbox)
{
    mFrameIdAndBboxMemory.resize_and_copy_from_host(FrameIdBbox);
    mnBbox = FrameIdBbox.size();
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
}

void NeRF_Model::UpdateFrameIdAndBboxOnline(const std::vector<FrameIdAndBbox>& FrameIdBbox,size_t newnumBbox)
{
    if(mbFirstUpdateBbox)
    {
        mFrameIdAndBboxMemory.resize(FrameIdBbox.size());
        mbFirstUpdateBbox = false;
        mnBbox = 0;
    }
        
    CUDA_CHECK_THROW(cudaMemcpy(mFrameIdAndBboxMemory.data() + mnBbox, FrameIdBbox.data() + mnBbox, newnumBbox * sizeof(FrameIdAndBbox), cudaMemcpyHostToDevice));
    mnBbox += newnumBbox;
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
}

bool NeRF_Model::Train_Step(std::shared_ptr<nerf::NeRF_Dataset> pTrainData)
{
    auto creation_time = std::chrono::steady_clock::now();

    //loop
    for(int i=0;i<500;i++)
    {
        GenerateBatch(mpTrainStream,pTrainData);

        //step is similar to instant-ngp 
        //Step();      //Note：Not suitable for the current version and cannot be used
        //step_No_Compacted is more concise, Without Compact, saving a forward process
        Step_No_Compacted(mpTrainStream);

        mpTrainer->optimizer_step(mpTrainStream, mLoss_Scale);
        CUDA_CHECK_THROW(cudaStreamSynchronize(mpTrainStream));
        mnTrainingStep++;

    } 

    float loss_cout[16];
    mSumLossMemory.copy_to_host(loss_cout);
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    mfPerTrainLoss = 0;
    for(int i=0;i<16;i++)
    {
        mfPerTrainLoss += loss_cout[i];
    }
    mfPerTrainLoss /= mnRaysPerBatch;
    auto train_time = std::chrono::steady_clock::now();
    //This timing method is not accurate, refer to NVIDIA Nsight Systems
    cout<<"Id: "<<mId<<" train_time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_time - creation_time).count();
    cout<<" Step: "<<mnTrainingStep<<" loss: "<<mfPerTrainLoss<<endl;

    return true;
}


bool NeRF_Model::Train_Step_Online(std::shared_ptr<nerf::NeRF_Dataset> pTrainData,size_t DataMutexIdx,int iter)
{
    auto creation_time = std::chrono::steady_clock::now();

    //loop
    for(int i=0;i<iter;i++)
    {
        {
            std::unique_lock<std::mutex> lock(*pTrainData->mvUpdateMutex[DataMutexIdx]);
            GenerateBatch(mpTrainStream,pTrainData);
        }
        
        Step_No_Compacted(mpTrainStream);
        mpTrainer->optimizer_step(mpTrainStream, mLoss_Scale);
        CUDA_CHECK_THROW(cudaStreamSynchronize(mpTrainStream));
        mnTrainingStep++;
        float loss_cout[16];
        mSumLossMemory.copy_to_host(loss_cout);
        CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
        mfPerTrainLoss = 0;
        for(int i=0;i<16;i++)
        {
            mfPerTrainLoss += loss_cout[i];
        }
        mfPerTrainLoss /= mnRaysPerBatch;
        mHisLoss.push_back(mfPerTrainLoss);
    } 
    auto train_time = std::chrono::steady_clock::now();
    cout<<"Id: "<<mId<<" train_time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(train_time - creation_time).count();
    cout<<" Step: "<<mnTrainingStep<<" loss: "<<mfPerTrainLoss<<endl;
    return true;
}


void NeRF_Model::Render(cudaStream_t pStream, const FrameIdAndBbox box,Eigen::Matrix4f Twc, cv::Mat& img,cv::Mat& depth_img, cv::Mat& mask_img, std::shared_ptr<NeRF_Dataset> pTrainData)
{
    
    //Render rand
    curandGenerator_t RenderGen;
    //Render 
    uint32_t OutputWidth = 4;
    uint32_t RenderBatchSize;     //mnRenderRaysPerBatch * mnRenderSampleNum
    uint32_t RenderRaysPerBatch;  //H * W
    uint32_t RenderRaysPerBatch128; //Multiple of 128
    // allocate_workspace_and_distribute
    tcnn::GPUMemoryArena::Allocation RenderBatchAlloc;
    RenderData RenderBatchMemory;

    // Allocate Render Batch Workspace
    RenderRaysPerBatch = box.h * box.w;  //rays 
    //The number of pixels in the 2D bounding box is not necessarily an exponential of 2
    uint32_t num = (RenderRaysPerBatch + (128 / mnRenderSampleNum) - 1) / (128 / mnRenderSampleNum);
    RenderRaysPerBatch128 = num * (128 / mnRenderSampleNum);
    RenderBatchSize = RenderRaysPerBatch128 * mnRenderSampleNum;
    RenderBatchMemory.Rays.resize(RenderRaysPerBatch);
    RenderBatchMemory.RaysInBBox.resize(RenderRaysPerBatch);

    curandStatus_t cst;
    cst = curandCreateGenerator(&RenderGen, CURAND_RNG_PSEUDO_XORWOW);
    assert(CURAND_STATUS_SUCCESS == cst);
    curandSetStream(RenderGen,pStream);
    
    auto scratch = tcnn::allocate_workspace_and_distribute<
    float,
    float,
    float,
    float,
    float,
    float,
    float
    >(
        pStream,&RenderBatchAlloc,
        3 * RenderBatchSize,
        mnRenderSampleNum * RenderRaysPerBatch,
        OutputWidth * RenderBatchSize,
        3 * RenderRaysPerBatch,
        RenderRaysPerBatch,
        RenderRaysPerBatch,
        RenderRaysPerBatch * mnRenderSampleNum
    );

    float* PointsInput = std::get<0>(scratch);  //R G B * Rays
    float* SamplesDistances = std::get<1>(scratch);   
    float* RgbSigmaOutput = std::get<2>(scratch);   
    float* rgb_rays = std::get<3>(scratch);
    float* depth_rays = std::get<4>(scratch);
    float* mask_rays = std::get<5>(scratch);
    float* RandDt = std::get<6>(scratch);

    RenderBatchMemory.PointsInput.set(PointsInput,3,RenderBatchSize);
    RenderBatchMemory.SamplesDistances.set(SamplesDistances,mnRenderSampleNum,RenderRaysPerBatch);
    RenderBatchMemory.RgbSigmaOutput.set(RgbSigmaOutput,OutputWidth,RenderBatchSize);
    RenderBatchMemory.rgb_rays.set(rgb_rays,3,RenderRaysPerBatch);
    RenderBatchMemory.depth_rays.set(depth_rays,1,RenderRaysPerBatch);
    RenderBatchMemory.mask_rays.set(mask_rays,1,RenderRaysPerBatch);
    RenderBatchMemory.RandDt.set(RandDt,mnRenderSampleNum,RenderRaysPerBatch);

    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
    
    //Generate rays
    tcnn::linear_kernel(GenerateRenderRays,0,pStream,
        RenderRaysPerBatch,
        box,
        mBoundingBox,
        Twc,
        mObjTow,
        RenderBatchMemory.Rays.data(),
        RenderBatchMemory.RaysInBBox.data(),
        pTrainData->mIntrinsicsMemory.data()
        );
    
    //Generate rand step 
    curandGenerateUniform(RenderGen,RenderBatchMemory.RandDt.data(),RenderRaysPerBatch * mnRenderSampleNum);

    tcnn::linear_kernel(GenerateRenderInputPoints,0,pStream,
        RenderRaysPerBatch,
        mnRenderSampleNum,
        mBoundingBox,
        RenderBatchMemory.Rays.data(),
        RenderBatchMemory.RaysInBBox.data(),
        RenderBatchMemory.PointsInput.data(),
        RenderBatchMemory.SamplesDistances.data(),
        RenderBatchMemory.RandDt.data()
        );
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    mpNetwork->inference(pStream,RenderBatchMemory.PointsInput,RenderBatchMemory.RgbSigmaOutput);
    
    tcnn::linear_kernel(VolumeRender_Render,0,pStream,
    RenderRaysPerBatch,
    mnRenderSampleNum,
    OutputWidth,
    mBoundingBox,
    mRgbActivation,
    mDensityActivation,
    1.0f,
    RenderBatchMemory.RgbSigmaOutput.data(),
    RenderBatchMemory.PointsInput.data(),
    RenderBatchMemory.SamplesDistances.data(),
    RenderBatchMemory.Rays.data(),
    RenderBatchMemory.RaysInBBox.data(),
    RenderBatchMemory.rgb_rays.data(),
    RenderBatchMemory.depth_rays.data(),
    RenderBatchMemory.mask_rays.data()
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    float* ptr = img.ptr<float>(0,0);
    size_t pixel_size = img.rows * img.cols * img.channels();
    CUDA_CHECK_THROW(cudaMemcpyAsync(ptr,RenderBatchMemory.rgb_rays.data(),pixel_size*sizeof(float),cudaMemcpyDeviceToHost,pStream));

    float* depth_ptr = depth_img.ptr<float>(0,0);
    size_t depth_pixel_size = depth_img.rows * depth_img.cols * depth_img.channels();
    CUDA_CHECK_THROW(cudaMemcpyAsync(depth_ptr,RenderBatchMemory.depth_rays.data(),depth_pixel_size*sizeof(float),cudaMemcpyDeviceToHost,pStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    float* mask_ptr = mask_img.ptr<float>(0,0);
    size_t mask_pixel_size = mask_img.rows * mask_img.cols * mask_img.channels();
    CUDA_CHECK_THROW(cudaMemcpyAsync(mask_ptr,RenderBatchMemory.mask_rays.data(),mask_pixel_size*sizeof(float),cudaMemcpyDeviceToHost,pStream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

}

void NeRF_Model::RenderVideo(cudaStream_t pStream, std::shared_ptr<NeRF_Dataset> pTrainData,const string& img_path_folder, const string& depth_path_folder,const float radius)
{   
    //Generate 360-degree camera poses
    vector<Eigen::Matrix4f> vToc;
    int theta_num = 60;
    float theta = 360 / float(theta_num);
    float cur_theta = 0.0f;
    float phi = 30;
    for(int i=0;i<theta_num;i++)
    {   
        cur_theta += theta;
        Eigen::Matrix4f Toc = GenerateToc(cur_theta,phi,radius);
        vToc.push_back(Toc);
    }

    //half img
    FrameIdAndBbox box;
    box.x = pTrainData->W / 4;
    box.y = pTrainData->H / 4;
    box.w = pTrainData->W / 2;
    box.h = pTrainData->H / 2;

    //Render rand
    curandGenerator_t RenderGen;
    //Render 
    uint32_t OutputWidth = 4;
    uint32_t RenderBatchSize;     //mnRenderRaysPerBatch * mnSampleNum
    uint32_t RenderRaysPerBatch;  //H * W
    uint32_t RenderRaysPerBatch128; //Multiple of 128
    // allocate_workspace_and_distribute
    tcnn::GPUMemoryArena::Allocation RenderBatchAlloc;
    RenderData RenderBatchMemory;

    // Allocate Render Batch Workspace
    RenderRaysPerBatch = box.h * box.w;  //rays 
    uint32_t num = (RenderRaysPerBatch + (128 / mnRenderSampleNum) - 1) / (128 / mnRenderSampleNum);
    RenderRaysPerBatch128 = num * (128 / mnRenderSampleNum);
    RenderBatchSize = RenderRaysPerBatch128 * mnRenderSampleNum;
    RenderBatchMemory.Rays.resize(RenderRaysPerBatch);
    RenderBatchMemory.RaysInBBox.resize(RenderRaysPerBatch);

    curandStatus_t cst;
    cst = curandCreateGenerator(&RenderGen, CURAND_RNG_PSEUDO_XORWOW);
    assert(CURAND_STATUS_SUCCESS == cst);
    curandSetStream(RenderGen,pStream);
    
    auto scratch = tcnn::allocate_workspace_and_distribute<
    float,
    float,
    float,
    float,
    float,
    float,
    float
    >(
        pStream,&RenderBatchAlloc,
        3 * RenderBatchSize,
        mnRenderSampleNum * RenderRaysPerBatch,
        OutputWidth * RenderBatchSize,
        3 * RenderRaysPerBatch,
        RenderRaysPerBatch,
        RenderRaysPerBatch,
        RenderRaysPerBatch * mnRenderSampleNum
    );

    float* PointsInput = std::get<0>(scratch);  //R G B * Rays
    float* SamplesDistances = std::get<1>(scratch);   
    float* RgbSigmaOutput = std::get<2>(scratch);   
    float* rgb_rays = std::get<3>(scratch);
    float* depth_rays = std::get<4>(scratch);
    float* mask_rays = std::get<5>(scratch);
    float* RandDt = std::get<6>(scratch);

    RenderBatchMemory.PointsInput.set(PointsInput,3,RenderBatchSize);
    RenderBatchMemory.SamplesDistances.set(SamplesDistances,mnRenderSampleNum,RenderRaysPerBatch);
    RenderBatchMemory.RgbSigmaOutput.set(RgbSigmaOutput,OutputWidth,RenderBatchSize);
    RenderBatchMemory.rgb_rays.set(rgb_rays,3,RenderRaysPerBatch);
    RenderBatchMemory.depth_rays.set(depth_rays,1,RenderRaysPerBatch);
    RenderBatchMemory.mask_rays.set(mask_rays,1,RenderRaysPerBatch);
    RenderBatchMemory.RandDt.set(RandDt,mnRenderSampleNum,RenderRaysPerBatch);

    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
    
    for(int i=0;i<vToc.size();i++)
    {
        Eigen::Matrix4f Toc = vToc[i];

        //Generate rays
        tcnn::linear_kernel(GenerateRenderVideoRays,0,pStream,
            RenderRaysPerBatch,
            box,
            mBoundingBox,
            Toc,
            RenderBatchMemory.Rays.data(),
            RenderBatchMemory.RaysInBBox.data(),
            pTrainData->mIntrinsicsMemory.data()
            );

        //Generate rand step 
        curandGenerateUniform(RenderGen,RenderBatchMemory.RandDt.data(),RenderRaysPerBatch * mnRenderSampleNum);

        tcnn::linear_kernel(GenerateRenderInputPoints,0,pStream,
        RenderRaysPerBatch,
        mnRenderSampleNum,
        mBoundingBox,
        RenderBatchMemory.Rays.data(),
        RenderBatchMemory.RaysInBBox.data(),
        RenderBatchMemory.PointsInput.data(),
        RenderBatchMemory.SamplesDistances.data(),
        RenderBatchMemory.RandDt.data()
        );
        CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

        mpNetwork->inference(pStream,RenderBatchMemory.PointsInput,RenderBatchMemory.RgbSigmaOutput);

        tcnn::linear_kernel(VolumeRender_Render,0,pStream,
            RenderRaysPerBatch,
            mnRenderSampleNum,
            OutputWidth,
            mBoundingBox,
            mRgbActivation,
            mDensityActivation,
            1.0f,
            RenderBatchMemory.RgbSigmaOutput.data(),
            RenderBatchMemory.PointsInput.data(),
            RenderBatchMemory.SamplesDistances.data(),
            RenderBatchMemory.Rays.data(),
            RenderBatchMemory.RaysInBBox.data(),
            RenderBatchMemory.rgb_rays.data(),
            RenderBatchMemory.depth_rays.data(),
            RenderBatchMemory.mask_rays.data()
        );
        CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

        cv::Mat img(box.h,box.w,CV_32FC3);
        float* ptr = img.ptr<float>(0,0);
        size_t pixel_size = img.rows * img.cols * img.channels();
        CUDA_CHECK_THROW(cudaMemcpyAsync(ptr,RenderBatchMemory.rgb_rays.data(),pixel_size*sizeof(float),cudaMemcpyDeviceToHost,pStream));

        cv::Mat depth_img(box.h,box.w,CV_32FC1);
        float* depth_ptr = depth_img.ptr<float>(0,0);
        size_t depth_pixel_size = depth_img.rows * depth_img.cols * depth_img.channels();
        CUDA_CHECK_THROW(cudaMemcpyAsync(depth_ptr,RenderBatchMemory.depth_rays.data(),depth_pixel_size*sizeof(float),cudaMemcpyDeviceToHost,pStream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

        string img_path = img_path_folder+"/"+to_string(i)+".png";
        cv::cvtColor(img,img,cv::COLOR_RGB2BGR);
        img.convertTo(img,CV_8UC3,255);
        cv::imwrite(img_path,img);
        //cout<<"save img to => "<<img_path<<endl;

        string depth_path = depth_path_folder + "/" +to_string(i)+".png";
        //cv::normalize(depth_img,depth_img,1.0,0.0,CV_MINMAX);
        depth_img.convertTo(depth_img,CV_16UC1,20000);
        cv::imwrite(depth_path,depth_img);
        //cout<<"save depth img to => "<<depth_path<<endl;
    }    

}

void NeRF_Model::GenerateMesh(cudaStream_t pStream, MeshData& mMeshData)
{
    BoundingBox box = mBoundingBox;
    Eigen::Vector3i res3i = GetMarchingCubesRes(mMesh.res, box);
    float thresh = mMesh.thresh;
    tcnn::GPUMemory<float> density = GetDensityOnGrid(res3i, box,pStream);

    MarchingCubes(box,res3i,thresh,density,mMesh.verts,mMesh.indices,pStream);
    compute_mesh_1ring(mMesh.verts, mMesh.indices, mMesh.verts_smoothed, mMesh.vert_normals,pStream);
    compute_mesh_vertex_colors(box,pStream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
}


tcnn::GPUMemory<float> NeRF_Model::GetDensityOnGrid(Eigen::Vector3i res3i, const BoundingBox& aabb,cudaStream_t pStream)
{
    const uint32_t n_elements = (res3i.x()*res3i.y()*res3i.z());
	tcnn::GPUMemory<float> density(n_elements,pStream);

    const uint32_t padded_output_width = mpNetwork->padded_output_width();

    tcnn::GPUMemoryArena::Allocation alloc;
	auto scratch = tcnn::allocate_workspace_and_distribute<
		float,
		tcnn::network_precision_t
	>(pStream, &alloc, n_elements * 3, n_elements * padded_output_width);

    float* positions = std::get<0>(scratch);
    tcnn::network_precision_t* mlp_out = std::get<1>(scratch);

    const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { tcnn::div_round_up((uint32_t)res3i.x(), threads.x), tcnn::div_round_up((uint32_t)res3i.y(), threads.y), tcnn::div_round_up((uint32_t)res3i.z(), threads.z)};

    generate_grid_samples_nerf_uniform<<<blocks, threads, 0, pStream>>>(res3i, positions);

    tcnn::GPUMatrix<float> PointsInput(positions,3, n_elements);
    tcnn::GPUMatrix<tcnn::network_precision_t> RgbSigmaOutput(mlp_out, padded_output_width, n_elements);

    mpNetwork->inference_mixed_precision_impl(pStream, PointsInput, RgbSigmaOutput);
    
    tcnn::linear_kernel(output_half_to_float,0,pStream,
    n_elements,
    padded_output_width,
    density.data(),
    RgbSigmaOutput.data()
    );

    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));

    return density;
}

void NeRF_Model::compute_mesh_vertex_colors(const BoundingBox aabb,cudaStream_t pStream)
{
    uint32_t n_verts = (uint32_t)mMesh.verts.size();
	if (!n_verts) {
		return;
	}

	mMesh.vert_colors.resize(n_verts);
	mMesh.vert_colors.memset(0);
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

    tcnn::GPUMatrix<float> positions_matrix(3,n_verts,pStream);
	tcnn::GPUMatrix<float> color_matrix(4,n_verts,pStream);
    tcnn::linear_kernel(generate_nerf_network_inputs_from_positions, 0, pStream,
     n_verts, 
     aabb, 
     mMesh.verts.data(), 
     positions_matrix.data());

	mpNetwork->inference(pStream, positions_matrix, color_matrix);
    tcnn::linear_kernel(extract_rgb_with_activation, 0, pStream, n_verts, color_matrix.data(),mMesh.vert_colors.data(), mRgbActivation);

}

void NeRF_Model::TransCPUMesh(cudaStream_t pStream,CPUMeshData& cpudata)
{
    tcnn::GPUMemory<float> verts;
    tcnn::GPUMemory<float> vert_normals;
    tcnn::GPUMemory<uint8_t> vert_colors;
    
    uint32_t n_verts = (uint32_t)mMesh.verts.size();
    verts.resize(n_verts * 3);
    vert_normals.resize(n_verts * 3);
    vert_colors.resize(n_verts * 3);
    tcnn::linear_kernel(trans_mesh_data, 0, pStream, n_verts, mMesh.verts.data(),mMesh.vert_normals.data(),mMesh.vert_colors.data(), 
                        verts.data(),vert_normals.data(),vert_colors.data());
    
    std::unique_lock<std::mutex> lock(cpudata.mesh_mutex);
    cpudata.have_reslult = true;
    cpudata.verts.resize(n_verts * 3);
    cpudata.normals.resize(n_verts * 3);
    cpudata.colors.resize(n_verts * 3);
	cpudata.indices.resize(mMesh.indices.size());

    CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
    verts.copy_to_host(cpudata.verts);
    vert_normals.copy_to_host(cpudata.normals);
    vert_colors.copy_to_host(cpudata.colors);
    mMesh.indices.copy_to_host(cpudata.indices);
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
}

void NeRF_Model::TransMesh(MeshData& meshdata)
{

    std::unique_lock<std::mutex> lock(meshdata.mesh_mutex);

    meshdata.have_reslult = true;
    if(meshdata.mVBO_verts != 0)
        glDeleteBuffers(1,&meshdata.mVBO_verts);
    if(meshdata.mVBO_colors != 0)
        glDeleteBuffers(1,&meshdata.mVBO_colors);
    if(meshdata.mVBO_normals != 0)
        glDeleteBuffers(1,&meshdata.mVBO_normals);
    if(meshdata.mEBO_indices != 0)
        glDeleteBuffers(1,&meshdata.mEBO_indices);

    cudaGraphicsResource* cuda_res_verts;
    cudaGraphicsResource* cuda_res_normals;
    cudaGraphicsResource* cuda_res_colors;
    cudaGraphicsResource* cuda_res_indices;

    uint32_t n_verts = (uint32_t)mMesh.verts.size();
    
    glGenBuffers(1,&meshdata.mVBO_verts);
    glBindBuffer(GL_ARRAY_BUFFER, meshdata.mVBO_verts);
    glBufferData(GL_ARRAY_BUFFER, n_verts*3*sizeof(float), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
  
    CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_res_verts, meshdata.mVBO_verts, cudaGraphicsRegisterFlagsWriteDiscard));
    
    glGenBuffers(1,&meshdata.mVBO_normals);
    glBindBuffer(GL_ARRAY_BUFFER, meshdata.mVBO_normals);
    glBufferData(GL_ARRAY_BUFFER, n_verts*3*sizeof(float), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_res_normals, meshdata.mVBO_normals, cudaGraphicsRegisterFlagsWriteDiscard));
    
    glGenBuffers(1,&meshdata.mVBO_colors);
    glBindBuffer(GL_ARRAY_BUFFER, meshdata.mVBO_colors);
    glBufferData(GL_ARRAY_BUFFER, n_verts*3*sizeof(uint8_t), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_res_colors, meshdata.mVBO_colors, cudaGraphicsRegisterFlagsWriteDiscard));
    
    uint32_t n_indices = (uint32_t)mMesh.indices.size();
    meshdata.indices_size = n_indices;
    glGenBuffers(1,&meshdata.mEBO_indices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshdata.mEBO_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, n_indices*sizeof(uint32_t), nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CUDA_CHECK_THROW(cudaGraphicsGLRegisterBuffer(&cuda_res_indices, meshdata.mEBO_indices, cudaGraphicsRegisterFlagsWriteDiscard));
    
    CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &cuda_res_verts, cudaStreamPerThread));
    cudaGraphicsMapResources(1, &cuda_res_normals, cudaStreamPerThread);
    cudaGraphicsMapResources(1, &cuda_res_colors, cudaStreamPerThread);
    cudaGraphicsMapResources(1, &cuda_res_indices, cudaStreamPerThread);
    
    size_t num_bytes;
    void* verts_ptr;
    void* normals_ptr;
    void* colors_ptr;
    void* indices_ptr;
    cudaGraphicsResourceGetMappedPointer(&verts_ptr,&num_bytes,cuda_res_verts);
    cudaGraphicsResourceGetMappedPointer(&normals_ptr,&num_bytes,cuda_res_normals);
    cudaGraphicsResourceGetMappedPointer(&colors_ptr,&num_bytes,cuda_res_colors);
    cudaGraphicsResourceGetMappedPointer(&indices_ptr,&num_bytes,cuda_res_indices);
    
    tcnn::linear_kernel(trans_mesh_data, 0, cudaStreamPerThread, n_verts, mMesh.verts.data(),mMesh.vert_normals.data(),mMesh.vert_colors.data(), 
                        (float*)verts_ptr,(float*)normals_ptr,(uint8_t*)colors_ptr);

    tcnn::linear_kernel(trans_indices_data, 0, cudaStreamPerThread, n_indices, mMesh.indices.data(),(uint32_t*)indices_ptr);

    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    
    cudaGraphicsUnmapResources(1, &cuda_res_verts, cudaStreamPerThread);
    cudaGraphicsUnmapResources(1, &cuda_res_normals, cudaStreamPerThread);
    cudaGraphicsUnmapResources(1, &cuda_res_colors, cudaStreamPerThread);
    cudaGraphicsUnmapResources(1, &cuda_res_indices, cudaStreamPerThread);

    cudaGraphicsUnregisterResource(cuda_res_verts);
    cudaGraphicsUnregisterResource(cuda_res_normals);
    cudaGraphicsUnregisterResource(cuda_res_colors);
    cudaGraphicsUnregisterResource(cuda_res_indices);

}

void NeRF_Model::SaveMesh(const string outname)
{
    save_mesh(mMesh.verts, mMesh.vert_normals, mMesh.vert_colors, mMesh.indices, outname.c_str(), mMesh.unwrap,mfScale, mOffset);
}

Eigen::Matrix4f NeRF_Model::GenerateToc(const float theta,const float phi,const float r)
{   
    float z = r * sin(phi * M_PI / 180.0f);
    float x = r * cos(phi * M_PI / 180.0f) * cos(theta * M_PI / 180.0f);
    float y = r * cos(phi * M_PI / 180.0f) * sin(theta * M_PI / 180.0f);
    Eigen::Vector3f t(x,y,z);
    Eigen::Vector3f z_axis = -t;
    z_axis.normalize();
    float r_v= (theta + 90.0f) * M_PI / 180.0f;
    Eigen::Vector3f x_axis(cos(r_v),sin(r_v),0);
    x_axis.normalize();
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    Eigen::Matrix4f Toc = Eigen::Matrix4f::Identity();
    Toc.col(0).head<3>() = x_axis;
    Toc.col(1).head<3>() = y_axis;
    Toc.col(2).head<3>() = z_axis;
    Toc.col(3).head<3>() = t;
    return Toc;
}

}
