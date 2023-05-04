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
#pragma once
#include <iostream>
#include <stdio.h>

#include "nerf_data.h"
#include "common.h"

namespace nerf{

// marching cubes related state
struct MeshState {
    float thresh = 2.0f;
    int res = 64;
    bool unwrap = false;

    tcnn::GPUMemory<Eigen::Vector3f> verts;
    tcnn::GPUMemory<Eigen::Vector3f> vert_normals;
    tcnn::GPUMemory<Eigen::Vector3f> vert_colors;
    tcnn::GPUMemory<Eigen::Vector4f> verts_smoothed; // homogenous
    tcnn::GPUMemory<uint32_t> indices;

    void clear() {
        indices={};
        verts={};
        vert_normals={};
        vert_colors={};
        verts_smoothed={};
    }
};
	
Eigen::Vector3i GetMarchingCubesRes(uint32_t res_1d, const BoundingBox& aabb);

void MarchingCubes(const BoundingBox box,const Eigen::Vector3i res3i,const float thresh,const tcnn::GPUMemory<float>& density,tcnn::GPUMemory<Eigen::Vector3f>& verts_out,tcnn::GPUMemory<uint32_t>& indices_out,cudaStream_t pStream);

void save_mesh(
	tcnn::GPUMemory<Eigen::Vector3f>& verts,
	tcnn::GPUMemory<Eigen::Vector3f>& normals,
	tcnn::GPUMemory<Eigen::Vector3f>& colors,
	tcnn::GPUMemory<uint32_t>& indices,
	const char* outputname,
	bool unwrap_it,
	float nerf_scale,
	Eigen::Vector3f nerf_offset
);

void compute_mesh_1ring(const tcnn::GPUMemory<Eigen::Vector3f>& verts, const tcnn::GPUMemory<uint32_t>& indices, tcnn::GPUMemory<Eigen::Vector4f>& output_pos, tcnn::GPUMemory<Eigen::Vector3f>& output_normals,cudaStream_t pStream);

}
