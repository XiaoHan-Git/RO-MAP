#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>

#include <opencv/cv.hpp>
#include <Eigen/Core>
#include "nerf.h"

using namespace std;

namespace nerf{

//image dataset
class NeRF_Dataset;

//Offline multi-object NeRF, more concise than online
class NerfManagerOffline
{
public:
    NerfManagerOffline(const string datasetPath, const string networkConfigFile, bool useDenseDepth);

    bool Init();

    bool ReadDataset();

    bool CreateNeRF(const string objectFile);

    bool WaitThreadsEnd();

    //tools function for visualization
    std::shared_ptr<NeRF> GetNeRF(int idx);
    vector<std::shared_ptr<NeRF>> GetAllNeRF();
    vector<Eigen::Matrix4f> GetAllTwc();
    void GetIntrinsics(float& fx,float& fy,float& cx,float& cy);

    string msNetworkConfigFile;
    string msDatasetPath;
    bool mbUseDenseDepth;
    int mNumGPU;
    //ptr
    vector<shared_ptr<NeRF_Dataset>> mvpDataset;
    vector<std::shared_ptr<NeRF>> mvpNeRFs;
    //One model corresponds to one thread
    vector<std::thread> mvThreads;

};


//online
class NerfManagerOnline
{
public:
    NerfManagerOnline(const string network_config_file,bool UseSparseDepth, int TrainStepIterations);

    bool Init();

    void DatasetInit(float fx,float fy,float cx,float cy,int H,int W, size_t imgs);

    void NewFrameToDataset(unsigned int imgId,const string timestamp, cv::Mat& img, cv::Mat& instance, const cv::Mat& depth_img, const Eigen::Matrix4f& pose);

    //Update the frame pose passed in before, not used
    void UpdateDataset(unsigned int CurId,unsigned int FrameNum,const vector<Eigen::Matrix4f>& Poses);

    size_t CreateNeRF(const int Class, const Eigen::Matrix4f &ObjTow, const nerf::BoundingBox &BoundingBox);

    int GetFrameIdx(double timastamp);

    //update object observation
    void UpdateNeRFBbox(const size_t idx, const vector<nerf::FrameIdAndBbox>& vFrameBbox,const int train_step);

    void DrawMesh(size_t idx);

    bool WaitThreadsEnd();

    void RenderNeRFsTest(const string out_path,const size_t Idx,const vector<string>& timestamp, const vector<FrameIdAndBbox>& vBbox, const vector<Eigen::Matrix4f>& vTwc,const float radius);

    string mNetworkConfigFile;
    bool mbUseSparseDepth;
    int mnTrainStepIterations;
    int mNumGPU;
    //ptr
    vector<shared_ptr<NeRF_Dataset>> mvpDataset;
    vector<std::shared_ptr<NeRF>> mvpNeRFs;
    //One model corresponds to one thread
    vector<std::thread> mvThreads;

};
}