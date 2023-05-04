#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include "nerf_data.h"
#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include "json/json.hpp"
using json = nlohmann::json;

namespace nerf{

NeRF_Dataset::~NeRF_Dataset()
{
    for(int i=0;i<mnImages;i++)
    {
        mvPixelMemory[i].free_memory();
        mvDepthMemory[i].free_memory();
        mvInstanceMemory[i].free_memory();
    }
    mPosesMemory.free_memory();
    mMetadataMemory.free_memory();
    mIntrinsicsMemory.free_memory();
    
}

bool NeRF_Dataset::ReadDataset(const string datasetPath)
{
    //Read camera parameters
    //Check settings file
    string configPath = datasetPath + "/config.yaml";
    cv::FileStorage fsSettings(configPath.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << configPath << endl;
       exit(0);
    }
    fx = fsSettings["Camera.fx"];
    fy = fsSettings["Camera.fy"];
    cx = fsSettings["Camera.cx"];
    cy = fsSettings["Camera.cy"];
    H = fsSettings["Camera.H"];
    W = fsSettings["Camera.W"];
    if(mbUseDepth)
        mfDepthScale = fsSettings["DepthMapFactor"];

    ifstream f;
    string imgFile = datasetPath + "/img.txt";
    f.open(imgFile.c_str());

    // skip comments
    string s0;
    getline(f,s0);

    uint32_t i = 0;
    string rgbPath;
    string depthPath;
    string instancePath;
    string stamp;
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string sImgName;
            ss >> stamp;
            ss >> sImgName;

            rgbPath = datasetPath + "/rgb/" + sImgName;
            if(mbUseDepth)
                depthPath = datasetPath + "/depth/" + sImgName;
            instancePath = datasetPath + "/instance/" + sImgName;
            mvImagesPath.push_back(rgbPath);
            if(mbUseDepth)
                mvDepthsPath.push_back(depthPath);
            mvInstancesPath.push_back(instancePath);
            mStampToIdx[stamp] = i;
            ++i;
        }
    }
    f.close();

    //Read camera pose
    string groundtruthPoseFile = datasetPath + "/groundtruth.txt";
    f.open(groundtruthPoseFile.c_str());
    // skip comments
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            ss >> stamp;
            float tx,ty,tz,qx,qy,qz,qw;
            ss >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
            Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
            Eigen::Quaternionf q(qw,qx,qy,qz);
            Eigen::Vector3f t(tx,ty,tz);
            Twc.topLeftCorner(3,3) = q.toRotationMatrix();
            Twc.col(3).head<3>() = t;
            mvIamgesPose.push_back(Twc);
        
        }
    }
    f.close();

    if(mvIamgesPose.empty())
    {
        cerr << "Load dataset error...No images..."<<endl;
        return false;
    }
    mnImages = mvIamgesPose.size();
    return true;
}

bool NeRF_Dataset::DataToGPU()
{
    cout << "Load Images to GPU ..."<<endl;
    //read to GPU
    if(mnImages == 0)
    {
        cerr << "No images..."<<endl;
        return false;
    }
    cudaSetDevice(mGPUid);

    //Allocate memory
    mvPixelMemory.resize(mnImages);
    if(mbUseDepth)
        mvDepthMemory.resize(mnImages);
    mvInstanceMemory.resize(mnImages);

    mMetadataMemory.resize(mnImages);
    vector<MetaData> metadata_tmp;
    metadata_tmp.resize(mnImages);

    mPosesMemory.resize_and_copy_from_host(mvIamgesPose);

    string rgbPath;
    string depthPath;
    string instancePath;
    cv::Mat rgbImg;
    cv::Mat depthImg;
    cv::Mat instanceImg;

    for(int i=0;i<mnImages;i++)
    {
        //color
        rgbPath = mvImagesPath[i];
        rgbImg = cv::imread(rgbPath,cv::IMREAD_COLOR);
        if(rgbImg.empty())  //success
        {  
            cerr<<"Can not read image... path: "<<rgbPath<<endl;  
            exit(0);  
        }  
        cv::cvtColor(rgbImg,rgbImg,cv::COLOR_BGR2RGB);
        rgbImg.convertTo(rgbImg,CV_32FC3,1.0/255);
        float* ptr = rgbImg.ptr<float>(0,0);
        size_t pixel_size = rgbImg.rows * rgbImg.cols * rgbImg.channels();
        mvPixelMemory[i].resize(pixel_size);
        mvPixelMemory[i].copy_from_host(ptr);
        CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

        //depth
        if(mbUseDepth)
        {
            depthPath = mvDepthsPath[i];
            //cv_16u
            depthImg = cv::imread(depthPath,cv::IMREAD_UNCHANGED);
            if(depthImg.empty())  //success
            {  
                cerr<<"Can not read image... path: "<<depthPath<<endl;  
                exit(0);  
            }
            depthImg.convertTo(depthImg,CV_32FC1,mfDepthScale);
            float* depth_ptr = depthImg.ptr<float>(0,0);
            size_t depth_size = depthImg.rows * depthImg.cols * depthImg.channels();
            mvDepthMemory[i].resize(depth_size);
            mvDepthMemory[i].copy_from_host(depth_ptr);
            CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
        }
        
        //instance
        instancePath = mvInstancesPath[i];
        //cv_8u
        instanceImg = cv::imread(instancePath,cv::IMREAD_UNCHANGED);
        if(instanceImg.empty())  //success
        {  
            cerr<<"Can not read image... path: "<<instancePath<<endl;  
            exit(0);  
        }

        uint8_t* instance_ptr = instanceImg.ptr<uint8_t>(0,0);
        size_t instance_size = instanceImg.rows * instanceImg.cols * instanceImg.channels();
        mvInstanceMemory[i].resize(instance_size);
        mvInstanceMemory[i].copy_from_host(instance_ptr);

        //collection of data
        MetaData meta;
        meta.pixels = mvPixelMemory[i].data();
        if(mbUseDepth)
            meta.depth = mvDepthMemory[i].data();
        else
            meta.depth = nullptr;
        meta.instance = mvInstanceMemory[i].data();
        meta.Pose = mPosesMemory.data()+i;
        metadata_tmp[i] = meta;

        CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    }
    
    mMetadataMemory.resize_and_copy_from_host(metadata_tmp);

    float fxfycxcy[4];
    mIntrinsicsMemory.resize(4);
    fxfycxcy[0] = fx;
    fxfycxcy[1] = fy;
    fxfycxcy[2] = cx;
    fxfycxcy[3] = cy;
    CUDA_CHECK_THROW(cudaMemcpyAsync(mIntrinsicsMemory.data(),fxfycxcy,sizeof(float)*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
    
    //Synchronize per thread default stream
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    //CUDA_CHECK_THROW(cudaDeviceSynchronize());
    
    cout << "Load Images to GPU completed..."<<endl;
    return true;
}

bool NeRF_Dataset::InitDataToGPU()
{
    if(mnImages == 0)
    {
        cerr << "Please set the predicted amount of image memory..."<<endl;
        return false;
    }
    
    cudaSetDevice(mGPUid);
    
    mvPixelMemory.resize(mnImages);
    if(mbUseDepth)
        mvDepthMemory.resize(mnImages);
    mvInstanceMemory.resize(mnImages);
    
    mMetadataMemory.resize(mnImages);
    mPosesMemory.resize(mnImages);
    mvIamgesPose_online.resize(mnImages);

    float fxfycxcy[4];
    mIntrinsicsMemory.resize(4);
    fxfycxcy[0] = fx;
    fxfycxcy[1] = fy;
    fxfycxcy[2] = cx;
    fxfycxcy[3] = cy;
    
    CUDA_CHECK_THROW(cudaMemcpyAsync(mIntrinsicsMemory.data(),fxfycxcy,sizeof(float)*4,cudaMemcpyHostToDevice,cudaStreamPerThread));

    //Synchronize per thread default stream
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

    cout << "Init Data to GPU completed..."<<endl;
    return true;

}

void NeRF_Dataset::FrameDataToGPU(unsigned int imgId,const string timestamp)
{
    
    cudaSetDevice(mGPUid);

    if(Temp_Img.empty())  //success
    {  
        cerr<<"img error ... "<<endl;  
        exit(0);  
    }  
    
    mStampToIdx[timestamp] = imgId;

    cv::cvtColor(Temp_Img,Temp_Img,cv::COLOR_BGR2RGB);
    Temp_Img.convertTo(Temp_Img,CV_32FC3,1.0/255);
    float* ptr = Temp_Img.ptr<float>(0,0);
    size_t pixel_size = Temp_Img.rows * Temp_Img.cols * Temp_Img.channels();
    mvPixelMemory[imgId].resize(pixel_size);
    mvPixelMemory[imgId].copy_from_host(ptr);
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

    //depth
    if(mbUseDepth)
    {
        if(Temp_Depth.empty())  //success
        {  
            cerr<<"depth img error ... "<<endl;  
            exit(0);  
        }
        //Temp_Depth.convertTo(Temp_Depth,CV_32FC1,depth_scale);
        float* depth_ptr = Temp_Depth.ptr<float>(0,0);
        size_t depth_size = Temp_Depth.rows * Temp_Depth.cols * Temp_Depth.channels();
        mvDepthMemory[imgId].resize(depth_size);
        mvDepthMemory[imgId].copy_from_host(depth_ptr);
        CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    }
    
    //cv_8u
    if(Temp_Instance.empty())  //success
    {  
        cerr<<"instance img error ... "<<endl;  
        exit(0);  
    }
    //cv::cvtColor(instance_img,instance_img,cv::COLOR_BGRA2GRAY);
    uint8_t* instance_ptr = Temp_Instance.ptr<uint8_t>(0,0);
    size_t instance_size = Temp_Instance.rows * Temp_Instance.cols * Temp_Instance.channels();
    mvInstanceMemory[imgId].resize(instance_size);
    mvInstanceMemory[imgId].copy_from_host(instance_ptr);

    //pose
    CUDA_CHECK_THROW(cudaMemcpyAsync(mPosesMemory.data()+imgId,Temp_Pose.data(),sizeof(Eigen::Matrix4f),cudaMemcpyHostToDevice,cudaStreamPerThread));
    //mvIamgesPose_online[imgId] = Temp_Pose;

    MetaData meta;
    meta.pixels = mvPixelMemory[imgId].data();
    if(mbUseDepth)
        meta.depth = mvDepthMemory[imgId].data();
    else
        meta.depth = nullptr;
    meta.instance = mvInstanceMemory[imgId].data();
    meta.Pose = mPosesMemory.data()+imgId;

    CUDA_CHECK_THROW(cudaMemcpyAsync(mMetadataMemory.data()+imgId,&meta,sizeof(MetaData),cudaMemcpyHostToDevice,cudaStreamPerThread));
    
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    mFrameDataNum += 1;
}

void NeRF_Dataset::UpdateDataGPU(unsigned int CurId,unsigned int FrameNum)
{
    cudaSetDevice(mGPUid);

    vector<std::unique_lock<std::mutex>> vLock;
    for(int i=0;i<mvUpdateMutex.size();i++)
        vLock.emplace_back(*mvUpdateMutex[i]);

    unsigned int head = CurId - FrameNum;
    CUDA_CHECK_THROW(cudaMemcpyAsync(mPosesMemory.data()+head,mvTemp_Update_Pose.data(),FrameNum * sizeof(Eigen::Matrix4f),cudaMemcpyHostToDevice,cudaStreamPerThread));
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));
    cout<< "UpdateDataGPU"<<endl;
}

}
