/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/11/2022
* Author: Xiao Han
*/
#include "ObjectFrame.h"
#include "MapPoint.h"
#include "Frame.h"

namespace ORB_SLAM2
{
Bbox::Bbox()
{
    mnClass = -1;
    mfConfidence = 0.0f;
    mbEdge = false;
    mbEdgeAndSmall = false;
    
}

Object_Frame::Object_Frame() : mbBad(false)
{   
}

void Object_Frame::SetPose(const cv::Mat &Tcw)
{
    Tcw.copyTo(mTcw);

}

cv::Mat Object_Frame::GetPose()
{
    return mTcw.clone();
}

//This function comes from CubeSLAM, with some changes
// merge short edges into long. edges n*4  each edge should start from left to right! 
void Object_Frame::MergeLines()
{   
    double pre_merge_dist_thre = 20; 
	double pre_merge_angle_thre = 5; 
	double edge_length_threshold = 30;

    bool can_force_merge = true;
    int counter = 0;

    Eigen::MatrixXd merge_lines_out = mLines;
    int total_line_number = merge_lines_out.rows();  // line_number will become smaller and smaller, merge_lines_out doesn't change
    
    pre_merge_angle_thre = pre_merge_angle_thre/180.0*M_PI;

    while ((can_force_merge) && (counter<500))
    {
	    counter++;
	    can_force_merge=false;
	    Eigen::MatrixXd line_vector = merge_lines_out.topRightCorner(total_line_number,2) - merge_lines_out.topLeftCorner(total_line_number,2);

        // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
	    Eigen::VectorXd all_angles; 
        all_angles.resize(line_vector.rows());
	    for (int i=0;i<all_angles.rows();i++)
	        all_angles(i)=std::atan2(line_vector(i,1),line_vector(i,0)); 

        for (int seg1 = 0;seg1 < total_line_number - 1; seg1++) 
        {
		    for (int seg2 = seg1+1; seg2 < total_line_number; seg2++)
            {
                double diff = std::abs(all_angles(seg1) - all_angles(seg2));
                double angle_diff = std::min(diff, M_PI - diff);

                if (angle_diff < pre_merge_angle_thre)
                {
                    double dist_1ed_to_2 = (merge_lines_out.row(seg1).tail(2) - merge_lines_out.row(seg2).head(2)).norm();
                    double dist_2ed_to_1 = (merge_lines_out.row(seg2).tail(2) - merge_lines_out.row(seg1).head(2)).norm();

                    if ((dist_1ed_to_2 < pre_merge_dist_thre) || (dist_2ed_to_1 < pre_merge_dist_thre))
                    {
                        Eigen::Vector2d merge_start, merge_end;
                        if (merge_lines_out(seg1,0) < merge_lines_out(seg2,0))
                            merge_start = merge_lines_out.row(seg1).head(2);
                        else
                            merge_start = merge_lines_out.row(seg2).head(2);
                        if (merge_lines_out(seg1,2) > merge_lines_out(seg2,2))
                            merge_end = merge_lines_out.row(seg1).tail(2);
                        else
                            merge_end = merge_lines_out.row(seg2).tail(2);
                        
                        double merged_angle = std::atan2(merge_end(1)-merge_start(1),merge_end(0)-merge_start(0));
                        
                        double temp = std::abs(all_angles(seg1) - merged_angle);
                        double merge_angle_diff = std::min( temp, M_PI-temp );
                        
                        if (merge_angle_diff < pre_merge_angle_thre)
                        {
                            merge_lines_out.row(seg1).head(2) = merge_start;
                            merge_lines_out.row(seg1).tail(2) = merge_end;
                            merge_lines_out.row(seg2) = merge_lines_out.row(total_line_number-1);
                            total_line_number--;  //also decrease  total_line_number
                            can_force_merge = true;
                            break;
                        }
                    }
                }
		    }
            if (can_force_merge)
                break;			
	    }
    }
    // Filter lines with length less than threshold
    if (edge_length_threshold > 0)
    {
        Eigen::MatrixXd line_vectors = merge_lines_out.topRightCorner(total_line_number,2) - merge_lines_out.topLeftCorner(total_line_number,2);
        Eigen::VectorXd line_lengths = line_vectors.rowwise().norm();

        int long_line_number = 0;
        Eigen::MatrixXd long_merge_lines(total_line_number, 4);
        for (int i = 0; i < total_line_number; i++)
        {
            if (line_lengths(i) > edge_length_threshold)
            {
                long_merge_lines.row(long_line_number) = merge_lines_out.row(i);
                long_line_number++;
            }
        }
        merge_lines_out = long_merge_lines.topRows(long_line_number);
    }
    else
	    merge_lines_out.conservativeResize(total_line_number,Eigen::NoChange);

    mLines = merge_lines_out;
    
}

//Mappoints with different depth of field may be in the same detection box,
//which is easy to filter through the BoxPlot
void Object_Frame::FilterMPByBoxPlot(cv::Mat& FrameTcw)
{   
    vector<float> CFrameDepth;
    cv::Mat Rcw = FrameTcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = FrameTcw.rowRange(0,3).col(3);

    for(MapPoint* pMP : mvpMapPoints)
    {
        cv::Mat worldPos = pMP->GetWorldPos();
        cv::Mat FramePos = Rcw * worldPos + tcw;
        CFrameDepth.push_back(FramePos.at<float>(2));
    }
    
    sort(CFrameDepth.begin(),CFrameDepth.end());

    //calculate BoxPlot
    if(CFrameDepth.size() < 4)
        return;
    
    float Q1 = CFrameDepth[int(CFrameDepth.size() / 4 )];
    float Q3 = CFrameDepth[int(CFrameDepth.size() * 3 / 4 )];
    float IQR = Q3 - Q1;
    float minth = Q1 - 1.5 * IQR;
    float maxth = Q3 + 1.5 * IQR;

    for(vector<MapPoint*>::iterator pMP = mvpMapPoints.begin();pMP!=mvpMapPoints.end();)
    {
        cv::Mat worldPos = (*pMP)->GetWorldPos();
        cv::Mat FramePos = Rcw * worldPos + tcw;
        float z = FramePos.at<float>(2);
        //Filter
        if(z < minth || z > maxth)
        {
            *pMP = mvpMapPoints.back();
            mvpMapPoints.pop_back();
            mSumPointsPos -= worldPos;
        }
        else
            pMP++;    
    }

}

//Calculate the mean and standard deviation
void Object_Frame::CalculateMeanAndStandard()
{
    if(mbBad)
        return;
    
    mPosMean = cv::Mat::zeros(3,1,CV_32F);
    mPosMean = mSumPointsPos / mvpMapPoints.size();

    float meanX = mPosMean.at<float>(0);
    float meanY = mPosMean.at<float>(1);
    float meanZ = mPosMean.at<float>(2);

    float sumX = 0, sumY = 0, sumZ = 0;
    for(MapPoint* pMP : mvpMapPoints)
    {
        cv::Mat worldPos = pMP->GetWorldPos();
        sumX += (meanX - worldPos.at<float>(0)) * (meanX - worldPos.at<float>(0));
        sumY += (meanY - worldPos.at<float>(1)) * (meanY - worldPos.at<float>(1));
        sumZ += (meanZ - worldPos.at<float>(2)) * (meanZ - worldPos.at<float>(2));
    }
    mfStandardX = sqrt(sumX / mvpMapPoints.size());
    mfStandardY = sqrt(sumY / mvpMapPoints.size());
    mfStandardZ = sqrt(sumZ / mvpMapPoints.size());

}

//Construct Bbox by reprojecting MapPoints, for data association
void Object_Frame::ConstructBboxByMapPoints(const Frame& CurrentFrame)
{
    if(mbBad)
        return;

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

    mBboxByMapPoints = cv::Rect(minU,minV,maxU-minU,maxV-minV);

}

}


