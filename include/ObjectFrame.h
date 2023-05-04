/*
* Create: SQ-SLAM
* Version: 1.0
* Created: 05/11/2022
* Author: Xiao Han
*/
#ifndef OBJECT_FRAME_H
#define OBJECT_FRAME_H

#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include <vector>

namespace ORB_SLAM2
{
class Frame;
class MapPoint;

class Bbox : public cv::Rect
{
public:
    Bbox();

    int mnClass;
    float mfConfidence;
    bool mbEdge;
    bool mbEdgeAndSmall;
};


class Object_Frame 
{
public:
    Object_Frame();

    // merge short edges into long. edges n*4  each edge should start from left to right! 
    void MergeLines();

    //Mappoints with different depth of field may be in the same detection box,
    //which is easy to filter through the BoxPlot
    void FilterMPByBoxPlot(cv::Mat& FrameTcw);

    //Calculate the mean and standard deviation
    void CalculateMeanAndStandard();

    //Construct Bbox by reprojecting MapPoints, for data association
    void ConstructBboxByMapPoints(const Frame& CurrentFrame);

    void SetPose(const cv::Mat &Tcw);

    cv::Mat GetPose();

    int mnClass;
    float mfConfidence;

    Bbox mBbox;
    long int mnFrameId;
    std::vector<int> mvIdxKeyPoints;
    Eigen::MatrixXd mLines;

    std::vector<MapPoint*> mvpMapPoints;
    cv::Mat mSumPointsPos;
    //position mean 
    cv::Mat mPosMean;
    //size
    float mfStandardX,mfStandardY,mfStandardZ;
    cv::Rect mBboxByMapPoints;
    
    cv::Mat mTcw;
    
    bool mbBad;

};


}

#endif //ORB_SLAM2_OBJECT_H
