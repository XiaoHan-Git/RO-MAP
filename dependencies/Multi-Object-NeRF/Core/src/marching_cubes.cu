/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   marching_cubes.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

/** Modification
* @date 12/06/2022
* @author Xiao Han
*/

#include "marching_cubes.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <third_party/tiny-cuda-nn/dependencies/stbi/stb_image_write.h>

namespace nerf{

Eigen::Vector3i GetMarchingCubesRes(uint32_t res_1d, const BoundingBox& aabb)
{   
    /* float scale = res_1d / (aabb.max - aabb.min).maxCoeff();
	Eigen::Vector3i res3i = ((aabb.max - aabb.min) * scale + Eigen::Vector3f::Constant(0.5f)).cast<int>();
	res3i.x() = tcnn::next_multiple((unsigned int)res3i.x(), 16u);
	res3i.y() = tcnn::next_multiple((unsigned int)res3i.y(), 16u);
	res3i.z() = tcnn::next_multiple((unsigned int)res3i.z(), 16u); */
	
	Eigen::Vector3i res3i(res_1d,res_1d,res_1d);
	
	return res3i;

}

__global__ void gen_vertices(BoundingBox aabb, Eigen::Vector3i res3i, const float* __restrict__ density, int*__restrict__ vertidx_grid, Eigen::Vector3f* verts_out, float thresh, uint32_t* __restrict__ counters) 
{
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x>=res3i.x() || y>=res3i.y() || z>=res3i.z()) 
		return;
	Eigen::Vector3f scale = (aabb.max-aabb.min).cwiseQuotient((res3i-Eigen::Vector3i::Ones()).cast<float>());
	Eigen::Vector3f offset = aabb.min;
	uint32_t res2 = res3i.x() * res3i.y();
	uint32_t res3 = res3i.x() * res3i.y() * res3i.z();
	uint32_t idx = x + y * res3i.x() + z * res2;
	float f0 = density[idx];
	bool inside=(f0>thresh);
	if (x<res3i.x()-1) {
		float f1 = density[idx+1];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx] = Eigen::Vector3f{float(x)+dt, float(y), float(z)}.cwiseProduct(scale) + offset;
			}
		}
	}
	if (y<res3i.y()-1) {
		float f1 = density[idx+res3i.x()];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx+res3]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx]= Eigen::Vector3f{float(x), float(y)+dt, float(z)}.cwiseProduct(scale) + offset;
			}
		}
	}
	if (z<res3i.z()-1) {
		float f1 = density[idx+res2];
		if (inside != (f1>thresh)) {
			uint32_t vidx = atomicAdd(counters,1);
			if (verts_out) {
				vertidx_grid[idx+res3*2]=vidx+1;
				float prevf=f0,nextf=f1;
				float dt=((thresh-prevf)/(nextf-prevf));
				verts_out[vidx]= Eigen::Vector3f{float(x), float(y), float(z)+dt}.cwiseProduct(scale) + offset;
			}
		}
	}
}

__global__ void gen_faces(Eigen::Vector3i res3i, const float* __restrict__ density, const int*__restrict__ vertidx_grid, uint32_t* indices_out, float thresh, uint32_t *__restrict__ counters) 
{
	// marching cubes tables from https://github.com/pmneila/PyMCubes/blob/master/mcubes/src/marchingcubes.cpp which in turn seems to be from https://web.archive.org/web/20181127124338/http://paulbourke.net/geometry/polygonise/
	// License is BSD 3-clause, which can be found here: https://github.com/pmneila/PyMCubes/blob/master/LICENSE
	/*
	static constexpr uint16_t edge_table[256] =
	{
		0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
	};
	*/
	static constexpr int8_t triangle_table[256][16] =
	{
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};

	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x>=res3i.x()-1 || y>=res3i.y()-1 || z>=res3i.z()-1) return;
	uint32_t res1 = res3i.x();
	uint32_t res2 = res3i.x() * res3i.y();
	uint32_t res3 = res3i.x() * res3i.y() * res3i.z();
	uint32_t idx = x + y * res3i.x() + z * res2;

	uint32_t idx_x=idx;
	uint32_t idx_y=idx + res3;
	uint32_t idx_z=idx + res3 * 2;

	int mask=0;
	if (density[idx]>thresh) mask|=1;
	if (density[idx+1]>thresh) mask|=2;
	if (density[idx+1+res1]>thresh) mask|=4;
	if (density[idx+res1]>thresh) mask|=8;
	idx+=res2;
	if (density[idx]>thresh) mask|=16;
	if (density[idx+1]>thresh) mask|=32;
	if (density[idx+1+res1]>thresh) mask|=64;
	if (density[idx+res1]>thresh) mask|=128;
	idx-=res2;

	if (!mask || mask==255) return;
	int local_edges[12];
	if (vertidx_grid) {
		local_edges[0]=vertidx_grid[idx_x];
		local_edges[1]=vertidx_grid[idx_y+1];
		local_edges[2]=vertidx_grid[idx_x+res1];
		local_edges[3]=vertidx_grid[idx_y];

		local_edges[4]=vertidx_grid[idx_x+res2];
		local_edges[5]=vertidx_grid[idx_y+1+res2];
		local_edges[6]=vertidx_grid[idx_x+res1+res2];
		local_edges[7]=vertidx_grid[idx_y+res2];

		local_edges[8]=vertidx_grid[idx_z];
		local_edges[9]=vertidx_grid[idx_z+1];
		local_edges[10]=vertidx_grid[idx_z+1+res1];
		local_edges[11]=vertidx_grid[idx_z+res1];
	}
	uint32_t tricount=0;
	const int8_t *triangles=triangle_table[mask];
	for (;tricount<15;tricount+=3) if (triangles[tricount]<0) break;
	uint32_t tidx = atomicAdd(counters+1,tricount);
	if (indices_out) {
		for (int i=0;i<15;++i) {
			int j = triangles[i];
			if (j<0) break;
			if (!local_edges[j]) {
				printf("at %d %d %d, mask is %d, j is %d, local_edges is 0\n", x,y,z,mask,j);
			}
			indices_out[tidx+i]=local_edges[j]-1;
		}
	}
}

__global__ void accumulate_1ring(uint32_t num_tris, const uint32_t* indices, const Eigen::Vector3f* verts_in, Eigen::Vector4f* verts_out, Eigen::Vector3f *normals_out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i>=num_tris) return;
	uint32_t ia=indices[i*3+0];
	uint32_t ib=indices[i*3+1];
	uint32_t ic=indices[i*3+2];
	Eigen::Vector3f pa=verts_in[ia];
	Eigen::Vector3f pb=verts_in[ib];
	Eigen::Vector3f pc=verts_in[ic];

	atomicAdd(&verts_out[ia][0], pb.x()+pc.x());
	atomicAdd(&verts_out[ia][1], pb.y()+pc.y());
	atomicAdd(&verts_out[ia][2], pb.z()+pc.z());
	atomicAdd(&verts_out[ia][3], 2.f);
	atomicAdd(&verts_out[ib][0], pa.x()+pc.x());
	atomicAdd(&verts_out[ib][1], pa.y()+pc.y());
	atomicAdd(&verts_out[ib][2], pa.z()+pc.z());
	atomicAdd(&verts_out[ib][3], 2.f);
	atomicAdd(&verts_out[ic][0], pb.x()+pa.x());
	atomicAdd(&verts_out[ic][1], pb.y()+pa.y());
	atomicAdd(&verts_out[ic][2], pb.z()+pa.z());
	atomicAdd(&verts_out[ic][3], 2.f);

	if (normals_out) {
		Eigen::Vector3f n= (pb-pa).cross(pa-pc); // don't normalise so it's weighted by area
		atomicAdd(&normals_out[ia][0], n.x());
		atomicAdd(&normals_out[ia][1], n.y());
		atomicAdd(&normals_out[ia][2], n.z());
		atomicAdd(&normals_out[ib][0], n.x());
		atomicAdd(&normals_out[ib][1], n.y());
		atomicAdd(&normals_out[ib][2], n.z());
		atomicAdd(&normals_out[ic][0], n.x());
		atomicAdd(&normals_out[ic][1], n.y());
		atomicAdd(&normals_out[ic][2], n.z());
	}
}

void MarchingCubes(const BoundingBox box,const Eigen::Vector3i res3i,const float thresh,const tcnn::GPUMemory<float>& density,tcnn::GPUMemory<Eigen::Vector3f>& verts_out,tcnn::GPUMemory<uint32_t>& indices_out,cudaStream_t pStream)
{
	tcnn::GPUMemory<uint32_t> counters;

	counters.enlarge(4);
	counters.memset(0);

	size_t n_bytes = res3i.x() * (size_t)res3i.y() * res3i.z() * 3 * sizeof(int);
	auto workspace = tcnn::allocate_workspace(pStream, n_bytes);
	CUDA_CHECK_THROW(cudaMemsetAsync(workspace.data(), -1, n_bytes, pStream));

	int* vertex_grid = (int*)workspace.data();

	const dim3 threads = { 4, 4, 4 };
	const dim3 blocks = { tcnn::div_round_up((uint32_t)res3i.x(), threads.x), tcnn::div_round_up((uint32_t)res3i.y(), threads.y), tcnn::div_round_up((uint32_t)res3i.z(), threads.z) };
	// count only
	CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
	gen_vertices<<<blocks, threads, 0,pStream>>>(box, res3i, density.data(), nullptr, nullptr, thresh, counters.data());
	gen_faces<<<blocks, threads,0,pStream>>>(res3i, density.data(), nullptr, nullptr, thresh, counters.data());
	CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
	std::vector<uint32_t> cpucounters; cpucounters.resize(4);
	counters.copy_to_host(cpucounters);
	CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

	//std::cout << "#vertices=" << cpucounters[0] << " #triangles=" << (cpucounters[1]/3)<<std::endl;

	uint32_t n_verts=(cpucounters[0]+127)&~127; // round for later nn stuff
	verts_out.resize(n_verts);
	verts_out.memset(0);
	indices_out.resize(cpucounters[1]);
	CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
	// actually generate verts
	gen_vertices<<<blocks, threads, 0,pStream>>>(box,res3i, density.data(), vertex_grid, verts_out.data(), thresh, counters.data()+2);
	gen_faces<<<blocks, threads,0,pStream>>>(res3i, density.data(), vertex_grid, indices_out.data(), thresh, counters.data()+2);

	CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
}

void save_mesh(
	tcnn::GPUMemory<Eigen::Vector3f>& verts,
	tcnn::GPUMemory<Eigen::Vector3f>& normals,
	tcnn::GPUMemory<Eigen::Vector3f>& colors,
	tcnn::GPUMemory<uint32_t>& indices,
	const char* outputname,
	bool unwrap_it,
	float nerf_scale,
	Eigen::Vector3f nerf_offset
) 
{
	std::vector<Eigen::Vector3f> cpuverts; cpuverts.resize(verts.size());
	std::vector<Eigen::Vector3f> cpunormals; cpunormals.resize(normals.size());
	std::vector<Eigen::Vector3f> cpucolors; cpucolors.resize(colors.size());
	std::vector<uint32_t> cpuindices; cpuindices.resize(indices.size());
	verts.copy_to_host(cpuverts);
	normals.copy_to_host(cpunormals);
	colors.copy_to_host(cpucolors);
	indices.copy_to_host(cpuindices);

	uint32_t numquads = ((cpuindices.size()/3)+1)/2;
	uint32_t numquadsx = uint32_t(sqrtf(numquads)+4) & (~3);
	uint32_t numquadsy = (numquads+numquadsx-1)/numquadsx;
	uint32_t quadresy = 8;
	uint32_t quadresx = quadresy+3;
	uint32_t texw = quadresx*numquadsx;
	uint32_t texh = quadresy*numquadsy;

	if (unwrap_it) {
		uint8_t* tex = (uint8_t*)malloc(texw*texh*3);
		for (uint32_t y = 0; y < texh; ++y) {
			for (uint32_t x = 0; x < texw; ++x) {
				uint32_t q = (x/quadresx)+(y/quadresy)*numquadsx;
				// 0 x x 3 - - 4
				// | .\x x\. . |
				// | . .\x x\. |
				// 2 - - 1 x x 5
				uint32_t xi = x % quadresx, yi = y % quadresy;
				uint32_t t = q*2 + (xi>yi+1);
				int r = (t*923)&255;
				int g = (t*3572)&255;
				int b = (t*5423)&255;
				//if (xi==yi+1 || xi==yi+2)
				//	r=g=b=0;
				tex[x*3+y*3*texw+0]=r;
				tex[x*3+y*3*texw+1]=g;
				tex[x*3+y*3*texw+2]=b;
			}
		}
		std::string Soutputname(outputname);
		Soutputname.substr(Soutputname.size()-4) = std::string(".tga");
		stbi_write_tga(Soutputname.c_str(), texw, texh, 3, tex);
		free(tex);
	}

	FILE* f = fopen(outputname,"wb");
	if (!f) {
		throw std::runtime_error{"Failed to open " + std::string(outputname) + " for writing."};
	}

	std::string Soutputname(outputname);
	if (Soutputname.substr(Soutputname.size()-3) == "ply") {
		// ply file
		fprintf(f,
			"ply\n"
			"format ascii 1.0\n"
			"comment output from https://github.com/NVlabs/instant-ngp\n"
			"element vertex %u\n"
			"property float x\n"
			"property float y\n"
			"property float z\n"
			"property float nx\n"
			"property float ny\n"
			"property float nz\n"
			"property uchar red\n"
			"property uchar green\n"
			"property uchar blue\n"
			"element face %u\n"
			"property list uchar int vertex_index\n"
			"end_header\n"
			, (unsigned int)cpuverts.size()
			, (unsigned int)cpuindices.size()/3
		);
		for (size_t i=0;i<cpuverts.size();++i) 
		{
			Eigen::Vector3f p=(cpuverts[i]-nerf_offset)/nerf_scale;
			Eigen::Vector3f c=cpucolors[i];
			Eigen::Vector3f n=cpunormals[i].normalized();
			unsigned char c8[3]={(unsigned char)tcnn::clamp(c.x()*255.f,0.f,255.f),(unsigned char)tcnn::clamp(c.y()*255.f,0.f,255.f),(unsigned char)tcnn::clamp(c.z()*255.f,0.f,255.f)};
			fprintf(f,"%0.5f %0.5f %0.5f %0.3f %0.3f %0.3f %d %d %d\n", p.x(), p.y(), p.z(), n.x(), n.y(), n.z(), c8[0], c8[1], c8[2]);
		}
		for (size_t i=0;i<cpuindices.size();i+=3) {
			fprintf(f,"3 %d %d %d\n", cpuindices[i+2], cpuindices[i+1], cpuindices[i+0]);
		}
	} else {
		// obj file
		if (unwrap_it) {
			fprintf(f, "mtllib nerf.mtl\n");
		}
		for (size_t i = 0; i < cpuverts.size(); ++i) {
			Eigen::Vector3f p = (cpuverts[i]-nerf_offset)/nerf_scale;
			Eigen::Vector3f c = cpucolors[i];
			fprintf(f,"v %0.5f %0.5f %0.5f %0.3f %0.3f %0.3f\n", p.x(), p.y(), p.z(), tcnn::clamp(c.x(), 0.f, 1.f), tcnn::clamp(c.y(), 0.f, 1.f), tcnn::clamp(c.z(), 0.f, 1.f));
		}
		for (auto &v: cpunormals) {
			auto n = v.normalized();
			fprintf(f,"vn %0.5f %0.5f %0.5f\n", n.x(), n.y(), n.z());
		}
		if (unwrap_it) {
			for (size_t i = 0; i < cpuindices.size(); i++) {
				uint32_t q = (uint32_t)(i/6);
				uint32_t x = (q%numquadsx)*quadresx;
				uint32_t y = (q/numquadsx)*quadresy;
				uint32_t d = quadresy-1;
				switch (i % 6) {
					case 0: break;
					case 1: x += d; y += d; break;
					case 2: y += d; break;
					case 3: x += 3; break;
					case 4: x += 3+d; break;
					case 5: x += 3+d; y += d; break;
				}
				fprintf(f,"vt %0.5f %0.5f\n", ((float)x+0.5f)/float(texw), 1.f-((float)y+0.5f)/float(texh));
			}
			fprintf(f, "g default\nusemtl nerf\ns 1\n");
			for (size_t i = 0; i < cpuindices.size(); i += 3) {
				fprintf(f,"f %u/%u/%u %u/%u/%u %u/%u/%u\n",
					cpuindices[i+2]+1,(uint32_t)i+3,  cpuindices[i+2]+1,
					cpuindices[i+1]+1,(uint32_t)i+2,cpuindices[i+1]+1,
					cpuindices[i+0]+1,(uint32_t)i+1,cpuindices[i+0]+1
				);
			}
		} else {
			for (size_t i = 0; i < cpuindices.size(); i += 3) {
				fprintf(f,"f %u//%u %u//%u %u//%u\n",
					cpuindices[i+2]+1, cpuindices[i+2]+1, cpuindices[i+1]+1, cpuindices[i+1]+1, cpuindices[i+0]+1, cpuindices[i+0]+1
				);
			}
		}
	}
	fclose(f);
}

void compute_mesh_1ring(const tcnn::GPUMemory<Eigen::Vector3f>& verts, const tcnn::GPUMemory<uint32_t>& indices, tcnn::GPUMemory<Eigen::Vector4f>& output_pos, tcnn::GPUMemory<Eigen::Vector3f>& output_normals,cudaStream_t pStream)
{
    output_pos.resize(verts.size());
	output_pos.memset(0);
	output_normals.resize(verts.size());
	output_normals.memset(0);
	tcnn::linear_kernel(accumulate_1ring, 0, pStream, indices.size()/3, indices.data(), verts.data(), output_pos.data(), output_normals.data());
	CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
	CUDA_CHECK_THROW(cudaStreamSynchronize(pStream));
}


}
