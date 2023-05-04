/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/18/2022
* Author: Xiao Han
*/
#ifndef OBJECT_MAP_H
#define OBJECT_MAP_H

#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "ObjectFrame.h"
#include "dependencies/g2o/g2o/types/se3quat.h"
#include <vector>
#include <map>
#include <mutex>

using namespace g2o;
using namespace std;

namespace ORB_SLAM2
{
class Frame;
class MapPoint;
class Map;

struct Cuboid
{
    // 11 DOF
    // pose
    SE3Quat mTobjw;
    // size
    double a1,a2,a3;
    float mfMaxDist;

};


class Object_Map 
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Object_Map(Map* pMap);

    //isbad
    bool IsBad();

    //Set Bad
    void SetBad(const string reason);

    // Update Sum Points Pos using new add mappoints;
    void UpdateMapPoints();

    //include new accociate MPs, replaced MPs, object merge MPs
    void AddNewMapPoints(MapPoint* pMP);

    // Calculate Mean And Standard
    void CalculateMeanAndStandard();

    // Calculate position Mean And Standard, for marge objects
    void CalculatePosMeanAndStandard();

    // Filter Outlier using Reprojection
    void FilterOutlier(const Frame& CurrentFrame);
    
    // Filter Outlier using EIF
    void EIFFilterOutlier();
    
    // Calculate Object Pose by MapPoints
    void CalculateObjectPose(const Frame& CurrentFrame);

    // Calculate Yaw Error
    float CalculateYawError(const cv::Mat& SampleRwobj,const cv::Mat& twobj, const Frame& CurrentFrame,vector<vector<int>>& AssLines);

    //Calculate Object size and Shape
    void CalculateObjectShape();
    
    //updata covisibility relationship
    void UpdateCovRelation(const vector<Object_Map*>& CovObjs);

    //After associating the new MapPoints, whether the bbox projected into the image change greatly
    bool whetherAssociation(const Object_Frame& ObjFrame, const Frame& CurrentFrame);

    //Construct Bbox by reprojecting MapPoints, for data association
    void ConstructBboxByMapPoints(const Frame& CurrentFrame);

    // Merge possible same Object
    void MergeObject(Object_Map* pObj,const double CurKeyFrameStamp);

    //Get Replace Object pointer
    Object_Map* GetReplaced();

    //Get Overlap Objects by MapPoints
    set<Object_Map*> GetShareObject();

    //
    void InsertHistoryBboxAndTwc(const Frame& CurrentFrame);


    static long unsigned int nNextId;

    long unsigned int mnId; 
    static bool mnCheckMPsObs;
    static float mfEIFthreshold;
    static float MergeMPsDistMultiple;
    static int mnEIFObsNumbers;
    static bool MergeDifferentClass;
    
    long unsigned int mnCreatFrameId;
    //latest observation frame
    long unsigned int mnlatestObsFrameId;

    //bad
    bool mbBad; 
    Object_Map* mpReplaced;

    bool mbFirstInit;
    //class 
    int mnClass;
    //Observation times
    int mnObs;
    //using Tracking data association
    Bbox mLastBbox;
    Bbox mLastLastBbox;

    std::map<double,Bbox> mHistoryBbox;
    std::map<double,Eigen::Matrix4f> mHistoryTwc;
    std::map<double,Bbox> mKeyFrameHistoryBbox;
    std::map<double,Bbox> mKeyFrameHistoryBbox_Temp;

    cv::Rect mMPsProjectRect;
    
    //Used to obtain pose information
    Eigen::MatrixXd mlatestFrameLines;
    //using for project axis
    float mfLength;

    std::vector<MapPoint*> mvpMapPoints;
    std::vector<MapPoint*> mvpNewAddMapPoints;
    cv::Mat mSumPointsPos;
    //position mean and standard deviation
    vector<cv::Mat> mvHistoryPos;
    cv::Mat mHistoryPosMean;
    float mfPosStandardX,mfPosStandardY,mfPosStandardZ;
    //size
    cv::Mat mPosMean;
    float mfStandardX,mfStandardY,mfStandardZ;
    
    //Backend merge object-----------
    // potential associated objects.
    std::map<Object_Map*, int> mPossibleSameObj;     
    // object id and times simultaneous appearances .
    std::map<Object_Map*, int> mmAppearSameTimes;

    //3D BoundBox
    Cuboid mShape;

    //rotation
    double mdYaw;
    // there Type: 18 intervals  <times Score Yaw>    
    std::map<int, Eigen::Vector3d> mmYawAndScore;

    //The t stored in this T has not undergone rotation transformation
    SE3Quat mTobjw;

    Map* mpMap;
    
    //NeRF
    bool haveNeRF = false;
    size_t pNeRFIdx;
    Eigen::Matrix4f mTow_NeRF;
    Eigen::Vector3f BBox_NeRF;
    
    Eigen::Vector2f twc_xy;
    Eigen::Vector2f twc_xy_last = Eigen::Vector2f::Zero();
    
protected:
    //mutex
    std::mutex mMutex;
    std::mutex mMutexMapPoints;
    std::mutex mMutexNewMapPoints;

};

}


#endif