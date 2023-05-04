/**
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/18/2022
* Author: Xiao Han
*/

#include "ObjectMap.h"
#include "Map.h"
#include "Converter.h"
#include "OptimizeObject.h"
#include <chrono>
#include "EIF.h"

namespace ORB_SLAM2
{
long unsigned int Object_Map::nNextId=0;
bool Object_Map::mnCheckMPsObs = false;
float Object_Map::mfEIFthreshold, Object_Map::MergeMPsDistMultiple;
int Object_Map::mnEIFObsNumbers;
bool Object_Map::MergeDifferentClass = false;


Object_Map::Object_Map(Map* pMap) : mbBad(false),mbFirstInit(true),mnObs(0),mpMap(pMap),mSumPointsPos(cv::Mat::zeros(3,1,CV_32F)),
                                    mTobjw(SE3Quat()),mpReplaced(static_cast<Object_Map*>(NULL))
{
    mnId=nNextId++;
}

bool Object_Map::IsBad()
{
    unique_lock<mutex> lock(mMutex);
    return mbBad;  
}

void Object_Map::SetBad(const string reason)
{
    unique_lock<mutex> lock(mMutex);
    unique_lock<mutex> lock1(mMutexMapPoints);
    unique_lock<mutex> lock2(mMutexNewMapPoints);

    for(MapPoint* pMP : mvpMapPoints)
    {
        pMP->EraseObject(this);
    }

    //cout<<"mnId: "<<mnId<<" Class: "<<mnClass<<" reason: "<<reason<<endl;
    mbBad = true;
}


//include new accociate MPs, replaced MPs, object merge MPs
void Object_Map::AddNewMapPoints(MapPoint* pMP)
{
    unique_lock<mutex> lock(mMutexNewMapPoints);
    mvpNewAddMapPoints.push_back(pMP);
    
}

// Update MapPoints using new add mappoints;
void Object_Map::UpdateMapPoints()
{
    if(mvpNewAddMapPoints.empty())
        return;
        
    unique_lock<mutex> lock(mMutexMapPoints);

    set<MapPoint*> mvpMPs(mvpMapPoints.begin(),mvpMapPoints.end());

    for(MapPoint* pMP : mvpNewAddMapPoints)
    {   
        if(mvpMPs.find(pMP) != mvpMPs.end())
            continue;
        pMP->AddObject(this);
        mvpMapPoints.push_back(pMP);

    }
    mvpNewAddMapPoints.clear();
}

//Calculate the mean and standard deviation
void Object_Map::CalculateMeanAndStandard()
{
    if(IsBad())
        return;
    
    unique_lock<mutex> lock(mMutexMapPoints);

    mSumPointsPos = cv::Mat::zeros(3,1,CV_32F);
    for(MapPoint* pMP : mvpMapPoints)
    {   
        mSumPointsPos += pMP->GetWorldPos();
    }
    mPosMean = mSumPointsPos / mvpMapPoints.size();
    
}

void Object_Map::EIFFilterOutlier()
{   

    unique_lock<mutex> lock(mMutexMapPoints);

    //Extended Isolation Forest
    std::mt19937 rng(12345);
	std::vector<std::array<float, 3>> data;
    
    if(mKeyFrameHistoryBbox.size() < 5 || mvpMapPoints.size() < 20)
        return;
    
	for (size_t i = 0; i < mvpMapPoints.size(); i++)
	{
		std::array<float, 3> temp;
        cv::Mat pos = mvpMapPoints[i]->GetWorldPos();
		for (uint32_t j = 0; j < 3; j++)
		{   
			temp[j] = pos.at<float>(j);
		}
		data.push_back(temp);
	}
    
    //auto t1 = std::chrono::system_clock::now();

	EIF::EIForest<float, 3> forest;
    
    double th = mfEIFthreshold;
    
    //Appropriately expand the EIF threshold for non-textured objects
    if(mnClass == 73 || mnClass == 46 || mnClass == 41)
    {
        th = th + 0.02;
    }

    double th_serious = th + 0.1;

    int point_num = 0;
    if(mvpMapPoints.size() > 100)
        point_num = mvpMapPoints.size() / 2;
    else
        point_num = mvpMapPoints.size() * 2 / 3;

	if(!forest.Build(40, 12345, data,point_num))
	{
		std::cerr << "Failed to build Isolation Forest.\n";
	}
    
	std::vector<double> anomaly_scores;

	if(!forest.GetAnomalyScores(data, anomaly_scores))
	{
		std::cerr << "Failed to calculate anomaly scores.\n";
	}
    
    vector<MapPoint*> newVpMapPoints;
    for(size_t i = 0,iend = mvpMapPoints.size();i<iend;i++)
    {   
        MapPoint* pMP = mvpMapPoints[i];

        //outlier                   If the point is added for a long time, it is considered stable
        if(mnCheckMPsObs)
        {
            if(anomaly_scores[i] > th_serious)
            {
                pMP->EraseObject(this);
            }
            else if(anomaly_scores[i] > th && mnlatestObsFrameId - pMP->mAssociateObjects[this] < mnEIFObsNumbers)
            {
                pMP->EraseObject(this);
            }
            else
                newVpMapPoints.push_back(pMP);
        }
        else
        {
            if(anomaly_scores[i] > th)
            {
                pMP->EraseObject(this);
            }
            else
                newVpMapPoints.push_back(pMP);
        } 

    }

    mvpMapPoints = newVpMapPoints;
    //auto t2 = std::chrono::system_clock::now();
    //std::cout<< "EIF time: "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()<<std::endl;

}

void Object_Map::FilterOutlier(const Frame& CurrentFrame)
{

    unique_lock<mutex> lock(mMutexMapPoints);

    bool Reprojection = true;

    if(mnlatestObsFrameId != CurrentFrame.mnId)
        Reprojection = false;
    
    //Make sure the Bbox is not at the edge of the image
    if(mLastBbox.x < CurrentFrame.mnMinX + 30 || mLastBbox.x + mLastBbox.width > CurrentFrame.mnMaxX - 30)
        Reprojection = false;
    if(mLastBbox.y < CurrentFrame.mnMinY + 30 || mLastBbox.y + mLastBbox.height > CurrentFrame.mnMaxY - 30)
        Reprojection = false;
    
    //Too small Bbox means a long distance and is prone to errors
    if(mLastBbox.area() < (CurrentFrame.mnMaxX -CurrentFrame.mnMinX) * (CurrentFrame.mnMaxY - CurrentFrame.mnMinY) * 0.05)
        Reprojection = false;
    
    //Reprojection Filter Outlier
    //now it is CurrentFrame Bbox
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    for(vector<MapPoint*>::iterator it=mvpMapPoints.begin();it!=mvpMapPoints.end();)
    {
        
        if((*it)->isBad())
        {   

            (*it)->EraseObject(this);
            (*it) = mvpMapPoints.back();
            mvpMapPoints.pop_back();
            continue;
        }
        
        if(Reprojection)
        {
            cv::Mat FramePos = Rcw * (*it)->GetWorldPos() + tcw;
            float invz = 1.0 / FramePos.at<float>(2);
            // camera -> image.
            float u = CurrentFrame.fx * FramePos.at<float>(0) * invz + CurrentFrame.cx;
            float v = CurrentFrame.fy * FramePos.at<float>(1) * invz + CurrentFrame.cy;
            cv::Point point(u,v);
            if(!mLastBbox.contains(point))
            {
                mSumPointsPos -= (*it)->GetWorldPos();
                (*it)->EraseObject(this);
                (*it) = mvpMapPoints.back();
                mvpMapPoints.pop_back();
            }
            else
                ++it;           
        }
        else
            ++it;
    } 
}

//Calculate the mean and standard deviation
void Object_Map::CalculatePosMeanAndStandard()
{
    if(mbBad)
        return;

    unique_lock<mutex> lock(mMutex);

    cv::Mat mSumHistoryPos = cv::Mat::zeros(3,1,CV_32F);
    for(const cv::Mat& Pos : mvHistoryPos)
        mSumHistoryPos += Pos;
    mHistoryPosMean = mSumHistoryPos / mvHistoryPos.size();

    float meanX = mHistoryPosMean.at<float>(0);
    float meanY = mHistoryPosMean.at<float>(1);
    float meanZ = mHistoryPosMean.at<float>(2);

    float sumX = 0, sumY = 0, sumZ = 0;
    for(const cv::Mat& Pos : mvHistoryPos)
    {
        sumX += (meanX - Pos.at<float>(0)) * (meanX - Pos.at<float>(0));
        sumY += (meanY - Pos.at<float>(1)) * (meanY - Pos.at<float>(1));
        sumZ += (meanZ - Pos.at<float>(2)) * (meanZ - Pos.at<float>(2));
    }
    mfPosStandardX = sqrt(sumX / mvHistoryPos.size());
    mfPosStandardY = sqrt(sumY / mvHistoryPos.size());
    mfPosStandardZ = sqrt(sumZ / mvHistoryPos.size());

}

void Object_Map::CalculateObjectPose(const Frame& CurrentFrame)
{
    if(mbBad)
        return;
    
    //Note that there are two translations here, 
    //  Tobjw   mshape->Tobjw
    //because the same group of map points have different center points in the world coordinate system
    //and the object coordinate system under the action of object rotation
    //Since the rotation is not considered in the translation of the Object_Frame, 
    //in order to maintain consistency when associating the Object_Frame and the Object_Map, an additional t is stored here.
    
    cv::Mat twobj = cv::Mat::zeros(3,1,CV_32F);
    vector<float> X_axis,Y_axis,Z_axis;

    {
        unique_lock<mutex> lock(mMutexMapPoints);

        for(MapPoint* pMP : mvpMapPoints)
        {   
            if(pMP->isBad())
                continue;

            cv::Mat Pos = pMP->GetWorldPos();
            X_axis.push_back(Pos.at<float>(0));
            Y_axis.push_back(Pos.at<float>(1));
            Z_axis.push_back(Pos.at<float>(2));
        }
        
        sort(X_axis.begin(),X_axis.end());
        sort(Y_axis.begin(),Y_axis.end());
        sort(Z_axis.begin(),Z_axis.end());

        twobj.at<float>(0) = (X_axis[0] + X_axis[X_axis.size()-1]) / 2;
        twobj.at<float>(1) = (Y_axis[0] + Y_axis[Y_axis.size()-1]) / 2;
        twobj.at<float>(2) = (Z_axis[0] + Z_axis[Z_axis.size()-1]) / 2;
        
        //using for project axis, to calculate rotation
        vector<float> length;
        length.push_back((X_axis[X_axis.size()-1] - X_axis[0]) / 2);
        length.push_back((Y_axis[Y_axis.size()-1] - Y_axis[0]) / 2);
        length.push_back((Z_axis[Z_axis.size()-1] - Z_axis[0]) / 2);
        sort(length.begin(),length.end());
        mfLength = length[2];

    }
    
    //calculate and update yaw
    if(mlatestFrameLines.rows() > 2 && !mLastBbox.mbEdgeAndSmall)
    {
        cv::Mat SampleRwobj;
        //calculate object Rotation

        vector<vector<int>> AssLines;
        vector<vector<int>> BestAssLines;
        Eigen::Matrix3d yawR;

        // -45° - 45°    90° / 5° = 18;
        //Take the middle
        // -42.5° - 42.5°    85° / 5° = 17;

        float sampleYaw = 0;
        float bestYaw = 0;
        float bestScore = 0;
        int bestIdx = -1;

        for(int i=0;i<18;i++)
        {   
            //sample yaw
            sampleYaw = (i * 5.0 - 42.5) / 180.0 * M_PI;
            yawR = Converter::eulerAnglesToMatrix(sampleYaw);
            SampleRwobj = Converter::toCvMat(yawR);
            
            AssLines.clear();
            float score = CalculateYawError(SampleRwobj,twobj,CurrentFrame,AssLines);

            if( score > bestScore)
            {   
                // two direction have association 
                if(!AssLines[0].empty() || !AssLines[1].empty())
                {
                    bestScore = score;
                    bestYaw = sampleYaw;
                    bestIdx = i;
                    BestAssLines = AssLines;
                }   
            }
        }
        //cout << "bestScore: "<<bestScore<<endl;
        
        if(bestScore != 0)
        {   
            //Refine rotation estimation
            float OptimizeYaw = ObjectOptimizer::OptimizeRotation(*this,BestAssLines,bestYaw,twobj,CurrentFrame);
            //cout << "OptimizeYaw: "<<OptimizeYaw<<endl;
            if(abs(bestYaw - OptimizeYaw) < 0.087266)  // 5/180 * PI
                bestYaw = OptimizeYaw;
        }
        
        //update yaw (history)
        if(bestScore!=0)
        {
            if(mmYawAndScore.count(bestIdx))
            {   
                //update times, score, yaw
                Eigen::Vector3d& Item = mmYawAndScore[bestIdx];
                Item(0) += 1.0; 
                Item(1) = (Item[1] * (1 - 1/Item(0)) + bestScore * 1/Item(0));
                Item(2) = (Item[2] * (1 - 1/Item(0)) + bestYaw * 1/Item(0)); 
            }
            else
            {   
                Eigen::Vector3d Item(1.0,bestScore,bestYaw);
                mmYawAndScore[bestIdx] = Item;
            }   
        }
    }
    else if(mnObs > 50 &&  mvpMapPoints.size() > 50)
    {
        //PCA
        Eigen::MatrixXd points(2,X_axis.size());
        points.row(0) = VectorXf::Map(&X_axis[0],X_axis.size()).cast<double>();
        points.row(1) = VectorXf::Map(&Y_axis[0],Y_axis.size()).cast<double>();
        double meanX = points.row(0).mean();
        double meanY = points.row(1).mean();

        points.row(0) = points.row(0) - Eigen::MatrixXd::Ones(1,X_axis.size()) * meanX;
        points.row(1) = points.row(1) - Eigen::MatrixXd::Ones(1,X_axis.size()) * meanY;
        
        Eigen::Matrix2d covariance = points * points.transpose() / double(X_axis.size());
        double ratio = max(covariance(0,0),covariance(1,1)) / min(covariance(0,0),covariance(1,1));

        double score = 0;
        double yaw = 0;
        int yawIdx = 0;
        //The standard deviation is greater than 1.1
        if(ratio > 1.21)
        {
            Eigen::EigenSolver<Eigen::Matrix2d> es(covariance);
            Eigen::Matrix2d EigenVectors = es.pseudoEigenvectors();
            Eigen::Matrix2d EigenValues = es.pseudoEigenvalueMatrix();
            //cout<<"covariance: "<<covariance<<endl;
        
            yaw = atan2(EigenVectors(1,0),EigenVectors(0,0)) * 180.0 / M_PI;
            if(yaw > 45.0 && yaw < 135.0)
                yaw = yaw - 90;
            else if(yaw >= 135.0)
                yaw = yaw  - 180.0;
            else if(yaw <= -135.0)
                yaw = 180 + yaw;
            else if(yaw < -45.0 && yaw > -135.0)
                yaw = 90 + yaw;
            
            yawIdx = int(abs(yaw + 42.5 / 5.0));
            yaw = yaw / 180.0 * M_PI;

            // 0 - 1 The score of PCA is less than that of projection
            score = mvpMapPoints.size() / mnObs;
            if(score > 5)
                score = 1;
        }

        if(score!=0)
        {   
        
            if(mmYawAndScore.count(yawIdx))
            {   
                //update times, score, yaw
                Eigen::Vector3d& Item = mmYawAndScore[yawIdx];
                Item(0) += 1.0; 
                Item(1) = (Item[1] * (1 - 1/Item(0)) + score * 1/Item(0));
                Item(2) = (Item[2] * (1 - 1/Item(0)) + yaw * 1/Item(0)); 
            }
            else
            {   
                Eigen::Vector3d Item(1.0,score,yaw);
                mmYawAndScore[yawIdx] = Item;
            }   
        }

    }

    //get the result yaw
    float resYaw = 0;
    
    if(!mmYawAndScore.empty())
    {
        vector<Eigen::Vector3d> YawAndScore;
        for(std::map<int,Eigen::Vector3d>::iterator it = mmYawAndScore.begin();it!=mmYawAndScore.end();it++)
        {
            YawAndScore.push_back(it->second);
        }

        if(YawAndScore.size() > 1)
        {
            sort(YawAndScore.begin(),YawAndScore.end(),[](const Eigen::Vector3d& v1,const Eigen::Vector3d& v2){return v1(1) > v2(1);});
            if(YawAndScore[0](0) > mnObs / 4.0)
                resYaw = YawAndScore[0](2);
            else if(YawAndScore[0](0) > mnObs / 6.0 && YawAndScore[0](0) > YawAndScore[1](0))
                resYaw = YawAndScore[0](2);
            else
            {
                sort(YawAndScore.begin(),YawAndScore.end(),[](const Eigen::Vector3d& v1,const Eigen::Vector3d& v2){return v1(0) > v2(0);});
                resYaw = YawAndScore[0](2);
            }
        }
        else
        {
            resYaw = YawAndScore[0](2);
        }
    }
    
    //cout <<"resYaw: "<< resYaw <<endl;
    Eigen::Matrix3d Rwobj =Converter::eulerAnglesToMatrix(resYaw);
    mTobjw = SE3Quat(Rwobj,Converter::toVector3d(twobj));
    mTobjw = mTobjw.inverse();

}

float Object_Map::CalculateYawError(const cv::Mat& SampleRwobj,const cv::Mat& twobj, const Frame& CurrentFrame,vector<vector<int>>& AssLines)
{
    //project axix to frame
    // center  X Y Z(3 points on the axis)

    vector<cv::Mat> PointPos;
    cv::Mat center = cv::Mat::zeros(3,1,CV_32F);
    PointPos.push_back(center);
    cv::Mat center_X = cv::Mat::zeros(3,1,CV_32F);
    center_X.at<float>(0) = mfLength;
    PointPos.push_back(center_X);
    cv::Mat center_Y = cv::Mat::zeros(3,1,CV_32F);
    center_Y.at<float>(1) = mfLength;
    PointPos.push_back(center_Y);
    cv::Mat center_Z = cv::Mat::zeros(3,1,CV_32F);
    center_Z.at<float>(2) = mfLength;
    PointPos.push_back(center_Z);

    //Project 
    vector<cv::Point2f> points;
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    for(const cv::Mat& Pos : PointPos)
    {
        cv::Mat framePos =  Rcw * (SampleRwobj * Pos + twobj) + tcw;
        float inv_z = 1.0 / framePos.at<float>(2);
        float u = CurrentFrame.fx * framePos.at<float>(0) * inv_z + CurrentFrame.cx;
        float v = CurrentFrame.fy * framePos.at<float>(1) * inv_z + CurrentFrame.cy;
        points.emplace_back(u,v);

    }
    //calculate angle
    //O-X

    float angleX;
    if(points[0].x < points[1].x)
        angleX = atan2(points[1].y - points[0].y, points[1].x - points[0].x);
    else
        angleX = atan2(points[0].y - points[1].y, points[0].x - points[1].x);

    //O-Y
    float angleY;
    if(points[0].x < points[2].x)
        angleY = atan2(points[2].y - points[0].y, points[2].x - points[0].x);
    else
        angleY = atan2(points[0].y - points[2].y, points[0].x - points[2].x);

    //O-Z
    float angleZ;
    if(points[0].x < points[3].x)
        angleZ = atan2(points[3].y - points[0].y, points[3].x - points[0].x);
    else
        angleZ = atan2(points[0].y - points[3].y, points[0].x - points[3].x);


    float error = 0;
    int num = 0;
    //th = 5, Lines with an error of less than 5 degrees are considered relevant
    float th = 5;

    //associate lines, for optimizer rotation
    vector<int> AssLinesX, AssLinesY, AssLinesZ;

    for(int i=0; i < mlatestFrameLines.rows();i++)
    {
        double x1 = mlatestFrameLines(i,0);
        double y1 = mlatestFrameLines(i,1);
        double x2 = mlatestFrameLines(i,2);
        double y2 = mlatestFrameLines(i,3);

        float angle = atan2(y2 - y1, x2 - x1);

        //3 lines angle error  0 ~ Pi/2
        float angle_error_X = abs((angle - angleX) * 180.0 / M_PI);
        angle_error_X = min(angle_error_X ,180 - angle_error_X);
        float angle_error_Y = abs((angle - angleY) * 180.0 / M_PI);
        angle_error_Y = min(angle_error_Y ,180 - angle_error_Y);
        float angle_error_Z = abs((angle - angleZ) * 180.0 / M_PI);
        angle_error_Z = min(angle_error_Z ,180 - angle_error_Z);

        float minError = min(min(angle_error_X,angle_error_Y),angle_error_Z);
        //cout<<"line: "<<i<<" minError: " <<minError<<endl;

        if(minError < th)
        {
            error += minError;
            ++num;
            if(minError == angle_error_X)
                AssLinesX.push_back(i);
            else if (minError == angle_error_Y)
                AssLinesY.push_back(i);
            else
                AssLinesZ.push_back(i);
        }

    }
    
    if(num == 0)
        return 0;
    else
    {
        AssLines.push_back(AssLinesX);
        AssLines.push_back(AssLinesY);
        AssLines.push_back(AssLinesZ);

        //The more associated lines and the smaller the error, the better
        float score = (float(num) / mlatestFrameLines.rows()) * (5 - error/num);
        return score;
    }

}

// Calculate size
void Object_Map::CalculateObjectShape()
{
    if(mbBad)
        return;

    cv::Mat tobjw_shape = cv::Mat::zeros(3,1,CV_32F);

    Eigen::Matrix3d R = mTobjw.to_homogeneous_matrix().block(0,0,3,3);
    cv::Mat Robjw = Converter::toCvMat(R);

    unique_lock<mutex> lock(mMutexMapPoints);

    //calculate object center
    vector<float> Obj_X_axis,Obj_Y_axis,Obj_Z_axis;
    vector<Eigen::Vector3d> points;
    
    for(MapPoint* pMP : mvpMapPoints)
    {   
        if(pMP->isBad())
                continue;
        
        cv::Mat Pos = pMP->GetWorldPos();
        Eigen::Vector3d ObjPos = R * Converter::toVector3d(Pos);
        //cout<<"size:  "<<ObjPos(0)<<"  "<<ObjPos(1)<<"  "<<ObjPos(2)<<" "<<endl;
        Obj_X_axis.push_back(ObjPos(0));
        Obj_Y_axis.push_back(ObjPos(1));
        Obj_Z_axis.push_back(ObjPos(2));

        points.push_back(ObjPos);

    }
    
    sort(Obj_X_axis.begin(),Obj_X_axis.end());
    sort(Obj_Y_axis.begin(),Obj_Y_axis.end());
    sort(Obj_Z_axis.begin(),Obj_Z_axis.end());
    
    //pos after Robjw
    tobjw_shape.at<float>(0) = -(Obj_X_axis[0] + Obj_X_axis[Obj_X_axis.size()-1]) / 2;
    tobjw_shape.at<float>(1) = -(Obj_Y_axis[0] + Obj_Y_axis[Obj_Y_axis.size()-1]) / 2;
    tobjw_shape.at<float>(2) = -(Obj_Z_axis[0] + Obj_Z_axis[Obj_Z_axis.size()-1]) / 2;

    if(mbFirstInit)
    {
        Cuboid shape;
        shape.mTobjw = mTobjw;
        mShape = shape;
        mbFirstInit = false;
    }

    if(haveNeRF)
        return;

    mShape.mTobjw = SE3Quat(R,Converter::toVector3d(tobjw_shape));
    mShape.a1 = abs(Obj_X_axis[Obj_X_axis.size()-1] - Obj_X_axis[0]) / 2;
    mShape.a2 = abs(Obj_Y_axis[Obj_Y_axis.size()-1] - Obj_Y_axis[0]) / 2;
    mShape.a3 = abs(Obj_Z_axis[Obj_Z_axis.size()-1] - Obj_Z_axis[0]) / 2;
    mShape.mfMaxDist = sqrt(mShape.a1 * mShape.a1 + mShape.a2 * mShape.a2 + mShape.a3 * mShape.a3);

}

//step5. updata covisibility relationship
void Object_Map::UpdateCovRelation(const vector<Object_Map*>& CovObjs)
{
    if(mbBad)
        return;

    unique_lock<mutex> lock(mMutex);
    for(Object_Map* pObj : CovObjs)
    {
        if(pObj == this)
            continue;
        if(pObj->IsBad())
            continue;

        mmAppearSameTimes[pObj]++;
    }

}

//After associating the new MapPoints, whether the bbox projected into the image change greatly
bool Object_Map::whetherAssociation(const Object_Frame& ObjFrame,const Frame& CurrentFrame)
{
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    float fx = CurrentFrame.fx;
    float fy = CurrentFrame.fy;
    float cx = CurrentFrame.cx;
    float cy = CurrentFrame.cy;

    
    vector<float> xpt,ypt,mix_xpt,mix_ypt;

    // original
    for(MapPoint* pMP : mvpMapPoints)
    {
        if(pMP->isBad())
            continue;

        cv::Mat pos = pMP->GetWorldPos();
        pos = Rcw * pos + tcw;
        float inv_z = 1.0 / pos.at<float>(2);
        float u =  fx * pos.at<float>(0) * inv_z + cx;
        float v =  fy * pos.at<float>(1) * inv_z + cy;

        xpt.push_back(u);
        mix_xpt.push_back(u);
        ypt.push_back(v);
        mix_ypt.push_back(v);

    }

    //mix
    for(MapPoint* pMP : ObjFrame.mvpMapPoints)
    {
        if(pMP->isBad())
            continue;

        cv::Mat pos = pMP->GetWorldPos();
        pos = Rcw * pos + tcw;
        float inv_z = 1.0 / pos.at<float>(2);
        float u =  fx * pos.at<float>(0) * inv_z + cx;
        float v =  fy * pos.at<float>(1) * inv_z + cy;

        mix_xpt.push_back(u);
        mix_ypt.push_back(v);
    }
    
    sort(xpt.begin(),xpt.end());
    sort(mix_xpt.begin(),mix_xpt.end());
    sort(ypt.begin(),ypt.end());
    sort(mix_ypt.begin(),mix_ypt.end());
    
    cv::Rect origin(xpt[0],ypt[0],xpt[xpt.size()-1] - xpt[0],ypt[ypt.size()-1] - ypt[0]);
    cv::Rect mix(mix_xpt[0],mix_ypt[0],mix_xpt[mix_xpt.size()-1] - mix_xpt[0],mix_ypt[mix_ypt.size()-1] - mix_ypt[0]);
    
    float IoUarea = (origin & mix).area();
    IoUarea = IoUarea / (origin.area() + mix.area() - IoUarea);
    if(IoUarea < 0.4 )
        return false;
    else
        return true;

}

//Construct Bbox by reprojecting MapPoints, for data association
void Object_Map::ConstructBboxByMapPoints(const Frame& CurrentFrame)
{
    if(mbBad)
        return;
    unique_lock<mutex> lock(mMutexMapPoints);

    vector<float> v_u;
    vector<float> v_v;
    cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    float fx = CurrentFrame.fx;
    float fy = CurrentFrame.fy;
    float cx = CurrentFrame.cx;
    float cy = CurrentFrame.cy;
    for(MapPoint* pMP : mvpMapPoints)
    {   
        // world -> camera.
        cv::Mat FramePos = Rcw * pMP->GetWorldPos() + tcw;
        float invz = 1.0 / FramePos.at<float>(2);
        // camera -> image.
        float u = fx * FramePos.at<float>(0) * invz + cx;
        float v = fy * FramePos.at<float>(1) * invz + cy;
        v_u.push_back(u);
        v_v.push_back(v);
    }

    sort(v_u.begin(),v_u.end());
    sort(v_v.begin(),v_v.end());

    // make insure in the image
    float minU = max(CurrentFrame.mnMinX,v_u[0]);
    float minV = max(CurrentFrame.mnMinY,v_v[0]);
    float maxU = min(CurrentFrame.mnMaxX,v_u[v_u.size()-1]);
    float maxV = min(CurrentFrame.mnMaxY,v_v[v_v.size()-1]);

    mMPsProjectRect = cv::Rect(minU,minV,maxU-minU,maxV-minV);

}

void Object_Map::MergeObject(Object_Map* pObj,const double CurKeyFrameStamp)
{
    //cout << "MergeObject: "<<pObj->mnClass<<endl;
    if(pObj->IsBad())
        return;
    
    unique_lock<mutex> lock(mMutex);

    //update
    if(pObj->mnCreatFrameId < mnCreatFrameId)
        mnCreatFrameId = pObj->mnCreatFrameId;
    if(pObj->mnlatestObsFrameId > mnlatestObsFrameId)
    {
        mnlatestObsFrameId = pObj->mnlatestObsFrameId; 
        mLastBbox = pObj->mLastBbox;
        mLastLastBbox = pObj->mLastBbox;
        mlatestFrameLines = pObj->mlatestFrameLines; 
    }
    mnObs += pObj->mnObs;

    bool checkMPs = false;
    SE3Quat Tobjw;
    float Maxdist_x = 0;
    float Maxdist_y = 0;
    float Maxdist_z = 0;
    if(mvpMapPoints.size() > 10)
    {
        checkMPs = true;
        if(mbFirstInit)
        {
            Tobjw = mTobjw;
            Maxdist_x = mfLength;
            Maxdist_y = mfLength;
            Maxdist_z = mfLength;
        }
        else
        {   //more accurate
            Tobjw = mShape.mTobjw;
            Maxdist_x = mShape.a1;
            Maxdist_y = mShape.a2;
            Maxdist_z = mShape.a3;
        }
    }

    for(size_t j=0;j<pObj->mvpMapPoints.size();j++)
    {   
        MapPoint* pMP = pObj->mvpMapPoints[j];
        if(pMP->isBad())
            continue;

        // check position
        if(checkMPs)
        {
            Eigen::Vector3d ObjPos = Tobjw * Converter::toVector3d(pMP->GetWorldPos());
            if(abs(ObjPos(0)) > MergeMPsDistMultiple * Maxdist_x || abs(ObjPos(1)) > MergeMPsDistMultiple * Maxdist_y || abs(ObjPos(2)) > MergeMPsDistMultiple * Maxdist_z)
                continue;
        }
        
        //new MapPoint
        AddNewMapPoints(pMP);
    }
    UpdateMapPoints();   
    
    //Fiter outlier
    EIFFilterOutlier();

    //update history pos
    for(const cv::Mat& pos : pObj->mvHistoryPos)
        mvHistoryPos.push_back(pos);

    //update covisibility relationship
    map<Object_Map*,int>::iterator it;
    for(it = pObj->mmAppearSameTimes.begin();it!= pObj->mmAppearSameTimes.end();it++)
    {
        mmAppearSameTimes[it->first] += it->second;
    }

    //update nerf bbox
    for(const auto& it : pObj->mHistoryBbox)
    {   
        double stamp = it.first;
        if(mHistoryBbox.find(stamp) != mHistoryBbox.end())
        {
            mHistoryBbox[stamp] = it.second;
            mHistoryTwc[stamp] = pObj->mHistoryTwc[stamp];
            if(CurKeyFrameStamp == stamp)
            {
                mKeyFrameHistoryBbox[stamp] = it.second;
                mKeyFrameHistoryBbox_Temp[stamp] = it.second;
            }
        }
            
    }

}

Object_Map* Object_Map::GetReplaced()
{
    unique_lock<mutex> lock(mMutex);
    return mpReplaced;

}

void Object_Map::InsertHistoryBboxAndTwc(const Frame& CurrentFrame)
{
    unique_lock<mutex> lock(mMutex);
    mHistoryBbox[CurrentFrame.mTimeStamp] = mLastBbox;
    mHistoryTwc[CurrentFrame.mTimeStamp] = Converter::toMatrix4f(CurrentFrame.mTcw).inverse();

}

}
