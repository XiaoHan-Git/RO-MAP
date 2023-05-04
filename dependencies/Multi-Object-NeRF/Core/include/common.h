#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

using namespace std;

#define SampleNum 32;
#define HalfSampleNum_plus_one 17

namespace nerf
{

struct FrameIdAndBbox
{
    uint32_t FrameId;
    uint32_t x,y,h,w;

};

struct BoundingBox
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3f min = Eigen::Vector3f::Zero();
    Eigen::Vector3f max = Eigen::Vector3f::Zero();
};

struct CPUMeshData
{
    std::vector<float> verts;
    std::vector<float> normals;
    std::vector<uint8_t> colors;
    std::vector<uint32_t> indices;
    bool have_reslult = false;
    std::mutex mesh_mutex;
    
};

//openGL VBO
struct MeshData
{
    GLuint mVBO_verts = 0;
    GLuint mVBO_normals = 0;
    GLuint mVBO_colors = 0;
    GLuint mEBO_indices = 0;
    uint32_t indices_size = 0;
    bool have_reslult = false;
    std::mutex mesh_mutex;
    
};

}