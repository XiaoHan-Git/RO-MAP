/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/12/2022
* Author: Xiao Han
*/

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"
#include "ObjectFrame.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>


namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame();
    cv::Mat DrawFrameBboxAndLines();
    cv::Mat FrameInstance();

protected:

    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // Info of the frame to be drawn
    cv::Mat mIm;
    cv::Mat mImColor;
    cv::Mat mImInstance;

    int N;
    vector<cv::KeyPoint> mvCurrentKeys;
    vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys;
    vector<int> mvIniMatches;
    int mState;

    Map* mpMap;

    std::mutex mMutex;

    //SQ-SLAM
    vector<Bbox> mvBbox;
    vector<Eigen::MatrixXd> mvObjLines;
    vector<int> mvObjectPointsIdx;
    vector<cv::Point> mvPoints;
    
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
