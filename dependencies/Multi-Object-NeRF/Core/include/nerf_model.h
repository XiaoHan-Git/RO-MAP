#pragma once
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <curand.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#include "json/json.hpp"
#include "marching_cubes.h"
#include "nerf_data.h"
#include "common.h"

using namespace std;

using json = nlohmann::json;
using precision_t = tcnn::network_precision_t;

namespace nerf{

enum class ENerfActivation : int {
	None,
	ReLU,
	Logistic,
	Exponential,
};

struct Ray
{
    Eigen::Vector3f o;
    Eigen::Vector3f d;
    float d_norm;
    float tmin;
    float tmax;

};

//Train
struct BatchData
{
    tcnn::GPUMemory<float> SampleXY; //mnRaysPerBatch * 2  (x,y)
    tcnn::GPUMemory<Ray> Rays;          //mnRaysPerBatch
    tcnn::GPUMemory<uint8_t> RaysInstance; //mnRaysPerBatch
    tcnn::GPUMemory<float> RandColors; //mnRaysPerBatch * 3  (r,g,b)
    tcnn::GPUMatrix<uint32_t> InBboxRaysCounter;    //1

    tcnn::GPUMatrix<float> Target;   // (3,mnRaysPerBatch)
    tcnn::GPUMatrix<float> TargetDepth;   // (1,mnRaysPerBatch)
    tcnn::GPUMatrix<float> PointsInput;     // (3,mnRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<float> CompactedPointsInput;     // (3,mnRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<uint32_t> CompactedPointsCounter;  // 1
    tcnn::GPUMatrix<float> SamplesDistances;   // (mnSampleNum,mnRaysPerBatch)
    tcnn::GPUMatrix<float> RandDt;      // (mnSampleNum,mnRaysPerBatch)
    tcnn::GPUMatrix<tcnn::network_precision_t> RgbSigmaOutput; // (outwidth,mnRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<tcnn::network_precision_t> dloss_dout;    // (outwidth,mnRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<float> rgb_rays;    //(3,mnRaysPerBatch)
    tcnn::GPUMatrix<float> depth_rays;  //(1,mnRaysPerBatch)
    tcnn::GPUMatrix<float> mask_rays;  //(1,mnRaysPerBatch)    
};

//Test
struct RenderData
{
    tcnn::GPUMemory<Ray> Rays;          //mnRenderRaysPerBatch
    tcnn::GPUMemory<int> RaysInBBox;    //mnRenderRaysPerBatch (0 or 1)
    tcnn::GPUMatrix<float> PointsInput;     // (3,mnRenderRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<float> SamplesDistances;   // (mnSampleNum,mnRenderRaysPerBatch)
    tcnn::GPUMatrix<float> RandDt;   // (mnSampleNum,mnRenderRaysPerBatch)
    tcnn::GPUMatrix<float> RgbSigmaOutput; // (outwidth,mnRenderRaysPerBatch * mnSampleNum)
    tcnn::GPUMatrix<float> rgb_rays;    //(3,mnRenderRaysPerBatch)
    tcnn::GPUMatrix<float> depth_rays;  //(1,mnRenderRaysPerBatch)
    tcnn::GPUMatrix<float> mask_rays;  //(1,mnRaysPerBatch)
};

class NeRF_Model
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NeRF_Model(int id, int GPUid, const BoundingBox& mBoundingBox, const Eigen::Matrix4f& mObjTow, uint8_t mInstanceId);
    static bool ReadNetworkConfig(const string config_path);
    bool ResetNetwork();

    //train
    void AllocateBatchWorkspace(cudaStream_t pStream,const uint32_t OutputWidth);
    void GenerateBatch(cudaStream_t pStream,std::shared_ptr<NeRF_Dataset> pTrainData);
    void Step(cudaStream_t pStream);    //Not suitable for the current version and cannot be used
    void Step_No_Compacted(cudaStream_t pStream);
    bool Train_Step(std::shared_ptr<nerf::NeRF_Dataset> pTrainData);
    bool Train_Step_Online(std::shared_ptr<nerf::NeRF_Dataset> pTrainData,size_t DataMutexIdx,int iter);

    //Render
    void Render(cudaStream_t pStream,const FrameIdAndBbox box,Eigen::Matrix4f Twc, cv::Mat& img,cv::Mat& depth_img, cv::Mat& mask_img, std::shared_ptr<NeRF_Dataset> pTrainData);
    void RenderVideo(cudaStream_t pStream, std::shared_ptr<NeRF_Dataset> pTrainData,const string& img_path, const string& depth_path, const float radius);
    Eigen::Matrix4f GenerateToc(const float theta,const float phi,const float r);

    //mesh
    void GenerateMesh(cudaStream_t pStream,MeshData& mMeshData);
    void TransCPUMesh(cudaStream_t pStream,CPUMeshData& cpudata);
    void TransMesh(MeshData& meshdata);
    void SaveMesh(const string outname);
    tcnn::GPUMemory<float> GetDensityOnGrid(Eigen::Vector3i res3i, const BoundingBox& aabb,cudaStream_t pStream);
    void compute_mesh_vertex_colors(const BoundingBox aabb,cudaStream_t pStream);
    
    //offline train function and attribute
    void UpdateFrameIdAndBbox(const std::vector<FrameIdAndBbox>& FrameIdBbox);
    bool mbUseDepth = false;
    
    //online train function and attribute
    void UpdateFrameIdAndBboxOnline(const std::vector<FrameIdAndBbox>& FrameIdBbox, size_t newnumBbox);
    bool mbFirstUpdateBbox = true;

    int mId;

    static json ClassNetworkConfig;
    json mNetworkConfig;

    //3D Bbox,pose,frame and 2d bbox
    BoundingBox mBoundingBox;
    Eigen::Matrix4f mObjTow = Eigen::Matrix4f::Zero();
    uint8_t mInstanceId;

    tcnn::GPUMemory<BoundingBox> mBboxMemory;
    tcnn::GPUMemory<FrameIdAndBbox> mFrameIdAndBboxMemory;
    size_t mnBbox = 0;
    tcnn::GPUMemory<float> mLossMemory;
    tcnn::GPUMemory<float> mSumLossMemory;
    float mfPerTrainLoss = 0;
    vector<float> mHisLoss;

    float mfScale = 1;
    Eigen::Vector3f mOffset = Eigen::Vector3f::Zero();

    //CUDA GPU
    int mGPUid =-1;
    //CUDA Stream
    cudaStream_t mpTrainStream;
    cudaStream_t mpInferenceStream;

    uint32_t m_seed = 1337;
    tcnn::default_rng_t m_rng;

    //encodnig
    int mnNumLevels = 0;
    int mnBaseResolution = 0;
    int mnLog2HashmapSize = 0;
    float mfLevelScale = 0;
    int mnFeatureDims = 0;

    //ptr
    std::shared_ptr<tcnn::Loss<precision_t>> mpLoss;
    std::shared_ptr<tcnn::Optimizer<precision_t>> mpOptimizer;
	std::shared_ptr<tcnn::Encoding<precision_t>> mpEncoding;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> mpNetwork;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> mpTrainer;

    //train
    int mnTrainingStep = 0;
    ENerfActivation mRgbActivation = ENerfActivation::Logistic;
    ENerfActivation mDensityActivation = ENerfActivation::Exponential;
    float mLoss_Scale = 128.0f;

    //model data(train batch, Render Batch)
    //rand
    curandGenerator_t mGen;
    //batch
    uint32_t mnBatchSize = 1 << 17;     //131072
    uint32_t mnRaysPerBatch = 1 << 12;  //4096
    uint32_t mnSampleNum = SampleNum;      //32
    uint32_t mnRenderSampleNum = 2 * SampleNum;      //64
    uint32_t mnPaddedOutWidth;
    //allocate_workspace_and_distribute
    tcnn::GPUMemoryArena::Allocation mBatchAlloc;
    bool mbBatchDataAllocated = false;
    BatchData mBatchMemory;

    //Mesh
    MeshState mMesh;

};

}