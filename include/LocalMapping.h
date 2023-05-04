/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/23/2022
* Author: Xiao Han
*/

#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "dependencies/Multi-Object-NeRF/Core/include/nerf_manager.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class LocalMapping
{
public:
    LocalMapping(Map* pMap, const float bMonocular);

    void SetLoopCloser(LoopClosing* pLoopCloser);

    void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

    //object-nerf-slam
    void SetNeRFManager(nerf::NerfManagerOnline* pNeRFManager);
    void InsertKeyFrameAndImg(KeyFrame *pKF,const cv::Mat& img,const cv::Mat& Instance);
    void NewDataToGPU();
    void UpdateObjNeRF();
    void GetUpdateBbox(Object_Map* pObj, vector<nerf::FrameIdAndBbox>& vFrameBbox);
    static float mfAngleChange;

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    //RO-MAP--------------------------------------
    //Update Object Size And Shape
    void UpdateObjSizeAndPose();

    //Merge Possible Objects
    void MergeObjects();

    //Merge Overlap Objects
    void MergeOverlapObjects();

    //t test 
    float tTest[101][4] = {0};

    std::mutex mNewDataNeRF;
    std::list<KeyFrame*> mlNewKeyFramesNeRF;
    std::list<cv::Mat> mlNewImgNeRF;
    std::list<cv::Mat> mlNewInstanceNeRF;
    unsigned int mCurImgId = 0;
    std::vector<KeyFrame*> mvNeRFDataKeyFrames;

    nerf::NerfManagerOnline* mpNeRFManager;

    unsigned long int mnLastUpdateObjFramaeId;

    set<Object_Map*> mvUpdateObj;

    //-----------------------------------------


    bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    std::list<KeyFrame*> mlNewKeyFrames;

    KeyFrame* mpCurrentKeyFrame;

    std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
