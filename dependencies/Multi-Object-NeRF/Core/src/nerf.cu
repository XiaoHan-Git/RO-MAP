
#include "nerf.h"
#include "nerf_data.h"
#include "nerf_model.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <GL/glew.h>
#include <unistd.h>
#include <sys/stat.h>

namespace nerf
{

int NeRF::GPUnum = -1;
int NeRF::curGPUid = -1;
int NeRF::curId = -1;


NeRF::NeRF()
{
    if(GPUnum < 1)
    {
        cerr<< "Cannot detect GPU ..."<<endl;
        exit(0);
    }
    curId += 1;
    curGPUid += 1;
    if(curGPUid >= GPUnum)
        curGPUid = 0;

    mId = curId;
    mGPUid = curGPUid;

}

bool NeRF::CreateModelOffline(const string path, bool useDenseDepth)
{
    if(!ReadBboxOffline(path))
    {
        cerr<< "... Read Bbox error ..."<<endl;
        exit(0);
    }
    
    //Model
    mbUseDepth = useDenseDepth;
    mpModel = std::make_shared<NeRF_Model>(mId,mGPUid,mBoundingBox,mObjTow,mInstanceId);
    mpModel->mbUseDepth = mbUseDepth;
    
    if(!mpModel->ResetNetwork())
    {
        cerr<< "... Create Model error ..."<<endl;
        exit(0);
    }
    return true;
}

bool NeRF::ReadBboxOffline(const string path)
{
    std::ifstream f(path);
    if(!f)
    {
        cerr << "Object Bbox file error..."<<endl;
        return false;
    }

    string s;
    //skip comments
    getline(f,s);

    getline(f,s);
    stringstream ss;
    ss << s;
    ss >> mClass;
    mInstanceId = uint8_t(mClass);
    //tx,ty,tz,x,y,z,w,a1,a2,a3
    float num[10];
    for(int i=0;i<10;i++)
    {
        ss >> num[i];
    }
    Eigen::Vector3f t(num[0],num[1],num[2]);
    Eigen::Quaternionf q(num[6],num[3],num[4],num[5]);
    
    //Pose and 3D bounding box
    Eigen::Matrix4f Two = Eigen::Matrix4f::Identity();
    Two.topLeftCorner(3,3) = q.toRotationMatrix();
    Two.col(3).head<3>() = t;
    mObjTow = Eigen::Matrix4f::Identity();
    mObjTow = Two.inverse();
    mBoundingBox.min = Eigen::Vector3f(-num[7],-num[8],-num[9]);
    mBoundingBox.max = Eigen::Vector3f(num[7],num[8],num[9]);

    //2D bbox
    FrameIdAndBbox item;
    string stamp;
    while(!f.eof())
    {
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            ss >> stamp;
            ss >> item.x;
            ss >> item.y;
            ss >> item.h;
            ss >> item.w;
            //stamp -> idx
            item.FrameId = mpTrainData->mStampToIdx[stamp];
            mFrameIdBbox.push_back(item);
        }
    }

    mnBbox = mFrameIdBbox.size();
    f.close();
    return true;
}

void NeRF::TrainOffline(const int iterations)
{
    cudaSetDevice(mGPUid);
    
    auto start = std::chrono::steady_clock::now();
    //Allocate
    if(!mpModel->mbBatchDataAllocated)
        mpModel->AllocateBatchWorkspace(mpModel->mpTrainStream,mpModel->mpNetwork->padded_output_width());
    
    //2d bbox cpu -> gpu
    mpModel->UpdateFrameIdAndBbox(mFrameIdBbox);
    auto allocate_time = std::chrono::steady_clock::now();
    cout<<"allocate_time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(allocate_time - start).count()<<std::endl;

    //Training
    for(int i=1;i<=iterations;i++)
    {
        mpModel->Train_Step(mpTrainData);
        if(i % 2 == 0)
        {
            mpModel->GenerateMesh(mpModel->mpInferenceStream,mMeshData);
            //TransMesh uses VBO, visualization directly uses GPU data, but there are bugs
            //TransCPUMesh. The data is sent to the cpu first, and then opengl sends it to the GPU, wasting time and resources
            //mpModel->TransMesh(mMeshData);
            mpModel->TransCPUMesh(mpModel->mpInferenceStream,mCPUMeshData);
        }
    }
    
    std::string filename = "./output/" + to_string(mId) + ".ply";
    mpModel->SaveMesh(filename);

    cout << "Training completed, press Ctrl+C to exit" <<endl;
}

//online 
void NeRF::SetAttributes(const int Class,const Eigen::Matrix4f& ObjTow,const BoundingBox& BoundingBox,size_t maxnumBbox)
{
    mClass = Class;
    mInstanceId = uint8_t(Class);
    mObjTow = ObjTow;
    mBoundingBox = BoundingBox;
    
    //Appropriately expand the 3D Bounding Box
    if(Class == 41 || Class == 73)
    {
        mBoundingBox.max = 1.2f * mBoundingBox.max;
        mBoundingBox.min = 1.2f * mBoundingBox.min;
    }
    else
    {
        mBoundingBox.max = 1.1f * mBoundingBox.max;
        mBoundingBox.min = 1.1f * mBoundingBox.min;
    }
    
    mFrameIdBbox.resize(maxnumBbox);
    mnBbox = 0;
}

bool NeRF::CreateModelOnline(bool useSparseDepth, int Iterations)
{
    mbUseDepth = useSparseDepth;
    mnIteration = Iterations;
    mpModel = std::make_shared<NeRF_Model>(mId,mGPUid,mBoundingBox,mObjTow,mInstanceId);
    mpModel->mbUseDepth = mbUseDepth;
    return true;
}

void NeRF::TrainOnline()
{
    cudaSetDevice(mGPUid);
    
    if(!mpModel->ResetNetwork())
    {
        cerr<< "... Create Model error ..."<<endl;
        exit(0);
    }

    auto start = std::chrono::steady_clock::now();
    if(!mpModel->mbBatchDataAllocated)
        mpModel->AllocateBatchWorkspace(mpModel->mpTrainStream,mpModel->mpNetwork->padded_output_width());
    auto allocate_time = std::chrono::steady_clock::now();
    cout<<"allocate_time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(allocate_time - start).count()<<std::endl;

    int train_step_count = 0;
    while(1)
    {
        int train_step = 0;
        //update bbox
        {
            std::unique_lock<std::mutex> lock(mUpdateBbox);
            //no update, wait
            if(mnBbox == mpModel->mnBbox)
                mCond.wait(lock);
            //update
            if(mnBbox > mpModel->mnBbox)
            {
                mpModel->UpdateFrameIdAndBboxOnline(mFrameIdBbox,mnBbox - mpModel->mnBbox);
                train_step = mnTrainStep;
                mnTrainStep = 0;
            }
        }
        //train
        if(mpModel->mnBbox > 10)
        {
            for(int i=0;i<train_step;i++)
            {
                mpModel->Train_Step_Online(mpTrainData,mDataMutexIdx,GetTrainIters(train_step_count));
                train_step_count += 1;
                if(train_step_count % 2 == 0)
                {
                    mpModel->GenerateMesh(mpModel->mpInferenceStream,mMeshData);
                    
                    //TransMesh uses VBO, visualization directly uses GPU data, but there are bugs
                    //TransCPUMesh. The data is sent to the cpu first, and then opengl sends it to the GPU, wasting time and resources
                    //mpModel->TransMesh(mMeshData);
                    mpModel->TransCPUMesh(mpModel->mpInferenceStream,mCPUMeshData);
                }
            }
        }

        if(CheckFinish())
            break;

        usleep(3000);
    }
    
    //last time 
    mpModel->Train_Step_Online(mpTrainData,mDataMutexIdx,GetTrainIters(train_step_count));
    mpModel->GenerateMesh(mpModel->mpInferenceStream,mMeshData);
    //mpModel->TransMesh(mMeshData);
    mpModel->TransCPUMesh(mpModel->mpInferenceStream,mCPUMeshData);
    cout<<"Id: "<<mId<<" finished! "<<endl;
    
}

void NeRF::RenderTestImg(const string out_path, const vector<string>& timestamp,const vector<Eigen::Matrix4f>& testTwc, const vector<FrameIdAndBbox>& testBbox,const float radius)
{
    //create folder
    string path_folder = out_path + "/" + to_string(mId);
    string command = "mkdir -p " + path_folder; 
    if(access(path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error");
    }

    string img_path_folder = path_folder + "/test_img";
    command = "mkdir -p " + img_path_folder; 
    if(access(img_path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error");
    }

    string depth_path_folder = path_folder + "/test_depth";
    command = "mkdir -p " + depth_path_folder; 
    if(access(depth_path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error"); 
    }

    string mask_path_folder = path_folder + "/test_mask";
    command = "mkdir -p " + mask_path_folder; 
    if(access(mask_path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error"); 
    }

    //Render 360-degree video
    string video_img_path_folder = path_folder + "/video_img";
    command = "mkdir -p " + video_img_path_folder; 
    if(access(video_img_path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error"); 
    }

    string video_depth_path_folder = path_folder + "/video_depth";
    command = "mkdir -p " + video_depth_path_folder; 
    if(access(video_depth_path_folder.c_str(),0) != 0)
    {
        if(system(command.c_str()) != 0)
            throw std::runtime_error("mkdir error"); 
    }

    cudaSetDevice(mGPUid);

    string test_filename = path_folder + "/test.txt";
    ofstream f;
    f.open(test_filename.c_str());
    f << fixed;
    f << "#stamp  box.x  box.y  box.h  box.w  tx  ty  tz  qx  qy  qz  qw (object-centric)"<<endl;
    cout<<"Render Object "<<mId<<" test imgs to "<<img_path_folder<< " ... please wait..."<<endl;
    for(size_t i=0;i<timestamp.size();i++)
    {
        string stamp = timestamp[i];
        Eigen::Matrix4f Twc = testTwc[i];
        FrameIdAndBbox box = testBbox[i];
    
        string img_path =  img_path_folder + "/"+ stamp +".png";
        cv::Mat img(box.h,box.w,CV_32FC3);
        string depth_path = depth_path_folder + "/"+ stamp +".png";
        cv::Mat depth_img(box.h,box.w,CV_32FC1);
        string mask_path = mask_path_folder + "/"+ stamp +".png";
        cv::Mat mask_img(box.h,box.w,CV_32FC1);

        Eigen::Matrix4f Toc = mObjTow * Twc;
        Eigen::Matrix3f R = Toc.topLeftCorner(3,3);
        Eigen::Quaternionf q(R);
        Eigen::Vector3f t = Toc.col(3).head<3>();
        f<< stamp <<" "<<box.x<<" "<<box.y<<" "<<box.h<<" "<<box.w<<" "<<t[0]<<" "<<t[1]<< " "<<t[2]<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
        
        //Render test image
        mpModel->Render(mpModel->mpInferenceStream,box,Twc,img,depth_img,mask_img,mpTrainData);

        cv::cvtColor(img,img,cv::COLOR_RGB2BGR);
        img.convertTo(img,CV_8UC3,255);
        cv::imwrite(img_path,img);
        //cout<<"save img to => "<<img_path<<endl;

        //cv::normalize(depth_img,depth_img,1.0,0.0,CV_MINMAX);
        // *20000, looks obvious
        depth_img.convertTo(depth_img,CV_16UC1,20000);
        cv::imwrite(depth_path,depth_img);
        //cout<<"save depth img to => "<<depth_path<<endl;

        mask_img.convertTo(mask_img,CV_8UC1,255);
        cv::imwrite(mask_path,mask_img);
        //cout<<"save img to => "<<mask_path<<endl;
    }

    f.close();

    //save training data  
    string train_filename = path_folder + "/train.txt";
    f.open(train_filename.c_str());
    f << fixed;
    f<<"#class Bbox"<<endl;
    Eigen::Vector3f Bbox = mBoundingBox.max;

    f << mClass << " "; 
    for(int i=0;i<3;i++)
    {
        f << Bbox[i] << " ";
    }
    f << endl;
    f<<"#stamp box.x box.y box.h box.w  tx  ty  tz  qx  qy  qz  qw (object-centric)"<<endl;
    vector<Eigen::Matrix4f> vTwc_train_online;
    uint32_t num = mpTrainData->mFrameDataNum;
    vTwc_train_online.resize(num);
    CUDA_CHECK_THROW(cudaMemcpyAsync(vTwc_train_online.data(),mpTrainData->mPosesMemory.data(),num * sizeof(Eigen::Matrix4f),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    CUDA_CHECK_THROW(cudaStreamSynchronize(cudaStreamPerThread));

    for(int i=0;i<mnBbox;i++)
    {
        FrameIdAndBbox& box = mFrameIdBbox[i];
        uint32_t Idx = box.FrameId;
        auto find_item = std::find_if(mpTrainData->mStampToIdx.begin(), mpTrainData->mStampToIdx.end(),
                                  [Idx](const auto& item) { return item.second == Idx; });
        
        string stamp = find_item->first;
        Eigen::Matrix4f Twc_train = vTwc_train_online[Idx];
        Eigen::Matrix4f Toc = mObjTow * Twc_train;
        Eigen::Matrix3f R = Toc.topLeftCorner(3,3);
        Eigen::Quaternionf q(R);
        Eigen::Vector3f t = Toc.col(3).head<3>();
        f<< stamp <<" "<<box.x<<" "<<box.y<<" "<<box.h<<" "<<box.w <<" "<<t[0]<<" "<<t[1]<< " "<<t[2]<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<endl;
    }
   
    f.close();

    // Render 360 Video
    cout<<"Render Object "<<mId<<" 360 video imgs to "<<video_img_path_folder<< " ... please wait..."<<endl;
    mpModel->RenderVideo(mpModel->mpInferenceStream,mpTrainData,video_img_path_folder,video_depth_path_folder,radius);

    cout<<"Save Object Mesh ... please wait..."<<endl;
    if(mCPUMeshData.have_reslult)
    {
        mpModel->GenerateMesh(mpModel->mpInferenceStream,mMeshData);
        string mesh_path = path_folder + "/obj.ply";
        mpModel->SaveMesh(mesh_path);
    }  
}

void NeRF::UpdateFrameBBox(const vector<nerf::FrameIdAndBbox>& vFrameBbox,const int train_step)
{
    std::unique_lock<std::mutex> lock(mUpdateBbox);
    size_t idx;
    //Update 2D Bbox
    for(size_t i=0;i<vFrameBbox.size();i++)
    {
        idx = mnBbox + i;
        mFrameIdBbox[idx] = vFrameBbox[i];
    }
    mnBbox += vFrameBbox.size();
    mnTrainStep = train_step;

    //notify bbox update
    mCond.notify_all();
}

inline int NeRF::GetTrainIters(int trainStep)
{
    //Gradually increase the number of training iterations
    /* if(trainStep == 0)
        return int(mnIteration * 2 / 5);
    else if(trainStep < 3)
        return int(mnIteration * 3 / 5);
    else 
        return mnIteration; */
    
    //more stable
    return mnIteration;
}

bool NeRF::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mFinishMutex);
    return mbFinishRequested;
}

void NeRF::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mFinishMutex);
    mbFinishRequested = true;
    mCond.notify_all();
}

vector<Eigen::Matrix4f> NeRF::GetTwc()
{
    vector<Eigen::Matrix4f> Twc;

    for(int i=0;i<mFrameIdBbox.size();i++)
    {
        FrameIdAndBbox& box = mFrameIdBbox[i];
        
        Twc.push_back(mpTrainData->mvIamgesPose[box.FrameId]);
    }
    return Twc;
    
}

BoundingBox NeRF::GetBoundingBox()
{
    return mBoundingBox;
}

Eigen::Matrix4f NeRF::GetObjTow()
{
    return mObjTow;
}

CPUMeshData& NeRF::GetCPUMeshData()
{
    return mCPUMeshData;
}

vector<FrameIdAndBbox> NeRF::GetFrameIdAndBBox()
{
    return mFrameIdBbox;
}

void NeRF::DrawCPUMesh()
{
    nerf::CPUMeshData& mesh = mCPUMeshData;
    {
        std::unique_lock<std::mutex> lock(mesh.mesh_mutex,try_to_lock);

        if (lock.owns_lock())
        {
            if(mesh.have_reslult)
            {
                glEnableClientState(GL_VERTEX_ARRAY);	
                glEnableClientState(GL_NORMAL_ARRAY);
                glEnableClientState(GL_COLOR_ARRAY);
                glVertexPointer(3,GL_FLOAT,	0,mesh.verts.data());	
                glColorPointer(3,GL_UNSIGNED_BYTE,0,mesh.colors.data());
                glNormalPointer(GL_FLOAT, 0, mesh.normals.data());
                glDrawElements(GL_TRIANGLES,mesh.indices.size(),GL_UNSIGNED_INT,mesh.indices.data());
                glDisableClientState(GL_VERTEX_ARRAY);    
                glDisableClientState(GL_NORMAL_ARRAY);
                glDisableClientState(GL_COLOR_ARRAY);
            }
        } 
    }
}

void NeRF::DrawMesh()
{
    /* {
        std::unique_lock<std::mutex> lock(mMeshData.mesh_mutex);
        cudaSetDevice(mGPUid);
        if(mMeshData.changed)
        {
            mpModel->TransMesh(mMeshData);
            mMeshData.changed = false;
        }
    } */
    
    std::unique_lock<std::mutex> lock(mMeshData.mesh_mutex);
    if(mMeshData.have_reslult)
    {
        glBindBuffer(GL_ARRAY_BUFFER, mMeshData.mVBO_verts);
        glVertexPointer(3,GL_FLOAT,	0,0);
        glEnableClientState(GL_VERTEX_ARRAY);	

        glBindBuffer(GL_ARRAY_BUFFER, mMeshData.mVBO_normals);
        glNormalPointer(GL_FLOAT,0,0);
        glEnableClientState(GL_NORMAL_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, mMeshData.mVBO_colors);
        glColorPointer(3,GL_UNSIGNED_BYTE,0,0);
        glEnableClientState(GL_COLOR_ARRAY);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mMeshData.mEBO_indices);
        glDrawElements(GL_TRIANGLES,mMeshData.indices_size,GL_UNSIGNED_INT,0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


        glDisableClientState(GL_VERTEX_ARRAY); 
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDisableClientState(GL_NORMAL_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER,0);

        glDisableClientState(GL_COLOR_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER,0);

    }
    

}

}

