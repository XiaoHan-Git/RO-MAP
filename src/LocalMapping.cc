/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
/*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/23/2022
* Author: Xiao Han
*/


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include"Converter.h"
#include <unistd.h>
#include<mutex>
#include "ObjectMap.h"

namespace ORB_SLAM2
{
float LocalMapping::mfAngleChange;

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):mnLastUpdateObjFramaeId(0),
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{

    //RO-MAP
    //read t-test data
    ifstream f;
    f.open("./lib/t_test.txt");
    if(!f.is_open())
        cerr<<"Can't read t-test data"<<endl;
    for(int i=0;i<101;i++)
    {
        for(int j=0;j<4;j++)
            f >> tTest[i][j];  
    }
    f.close();

}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;
    vector<float> vTimesLocalMapping;
    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);
        

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            //Close the loop closing thread, and of course, it can also be enabled
            //mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

            //object-nerf-slam------------------------------------------------------
            //NeRF data to GPU
            NewDataToGPU();
            
            //Estimate size and pose
            UpdateObjSizeAndPose();

            unique_lock<mutex> lock(mpMap->mMutexObjects);
            //Merge possible objects
            MergeObjects();
            MergeOverlapObjects();
            
            //object NeRF to GPU
            UpdateObjNeRF();
            //object-nerf-slam------------------------------------------------------

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	        vTimesLocalMapping.push_back(ttrack);

        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
            {
                //last deal objects
                vector<Object_Map*> vObjs = mpMap->GetAllObjectMap();
                Object_Map::mnCheckMPsObs = false;
                for(Object_Map* pOBJ : vObjs)
                {
                    if(pOBJ->IsBad())
                        continue;
                    pOBJ->EIFFilterOutlier();
                    pOBJ->CalculateObjectShape();
                }
                
                break;
            }
                
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();

    sort(vTimesLocalMapping.begin(),vTimesLocalMapping.end());
    float totaltime = 0;
    for(int ni=0; ni<vTimesLocalMapping.size(); ni++)
    {
        totaltime+=vTimesLocalMapping[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median LocalMapping time: " << vTimesLocalMapping[vTimesLocalMapping.size()/2] << endl;
    cout << "mean LocalMapping time: " << totaltime/vTimesLocalMapping.size() << endl;

}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}



bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::UpdateObjSizeAndPose()
{   
    vector<Object_Map*> vObjs;
    
    {
        unique_lock<mutex> lock(mpMap->mMutexObjects);
        vObjs = mpMap->GetAllObjectMap();
        if(vObjs.empty())
            return;
    }
    
    //Update changed Object
    mvUpdateObj.clear();

    for(Object_Map* pObj : vObjs)
    {
        if(pObj->IsBad())
            continue;
        
        //The object has new observations in the tracking thread
        if(pObj->mnlatestObsFrameId > mnLastUpdateObjFramaeId)
        {
            mvUpdateObj.insert(pObj);
        }
    }
        
    if(mvUpdateObj.empty())
        return;
    
    unsigned long int maxUpdateObjFramaeId = mnLastUpdateObjFramaeId;
    
    for(Object_Map* pObj : mvUpdateObj)
    {
        // Calculate size and pose
        pObj->CalculateObjectShape();  

        //KeyFrame and 2DBbox
        if(pObj->mHistoryBbox.find(mpCurrentKeyFrame->mTimeStamp) != pObj->mHistoryBbox.end())
        {
            //Store keyframes
            pObj->mKeyFrameHistoryBbox[mpCurrentKeyFrame->mTimeStamp] = pObj->mHistoryBbox[mpCurrentKeyFrame->mTimeStamp];
            pObj->mKeyFrameHistoryBbox_Temp[mpCurrentKeyFrame->mTimeStamp] = pObj->mHistoryBbox[mpCurrentKeyFrame->mTimeStamp];
        }
            
        if(pObj->mnlatestObsFrameId > maxUpdateObjFramaeId)
            maxUpdateObjFramaeId = pObj->mnlatestObsFrameId;
    }

    mnLastUpdateObjFramaeId = maxUpdateObjFramaeId;

}


void LocalMapping::MergeObjects()
{
    
    vector<Object_Map*> vObjs = mpMap->GetAllObjectMap();
    if(vObjs.empty())
        return;
    
    for(Object_Map* pOBJ : vObjs)
    {
        if(pOBJ->IsBad())
            continue;
        if(!pOBJ->mPossibleSameObj.empty())
        {

            cv::Mat samp1Pos = pOBJ->mHistoryPosMean;
            float samp1std_x = pOBJ->mfPosStandardX;
            float samp1std_y = pOBJ->mfPosStandardY;
            float samp1std_z = pOBJ->mfPosStandardZ;
            float n1 = pOBJ->mnObs;

            for(map<Object_Map*, int>::iterator it = pOBJ->mPossibleSameObj.begin();it!= pOBJ->mPossibleSameObj.end();it++)
            {
                Object_Map* pCandidateObj = it->first;
                if(pCandidateObj->IsBad())
                    continue;
                else if( pOBJ->mmAppearSameTimes.find(pCandidateObj) != pOBJ->mmAppearSameTimes.end())
                    continue;
                else
                {
                    //double t-test
                    cv::Mat samp2Pos = pCandidateObj->mHistoryPosMean;
                    float samp2std_x = pCandidateObj->mfPosStandardX;
                    float samp2std_y = pCandidateObj->mfPosStandardY;
                    float samp2std_z = pCandidateObj->mfPosStandardZ;
                    float n2 = pCandidateObj->mnObs;


                    //pooled standard deviation
                    float Sp_x = sqrt(((n1-1)*samp1std_x*samp1std_x + (n2-1)*samp2std_x*samp2std_x) / (n1+n2-2)*(1/n1+1/n2));
                    float Sp_y = sqrt(((n1-1)*samp1std_y*samp1std_y + (n2-1)*samp2std_y*samp2std_y) / (n1+n2-2)*(1/n1+1/n2));
                    float Sp_z = sqrt(((n1-1)*samp1std_z*samp1std_z + (n2-1)*samp2std_z*samp2std_z) / (n1+n2-2)*(1/n1+1/n2));

                    float t_test_x = abs(samp1Pos.at<float>(0) - samp2Pos.at<float>(0)) / Sp_x;
                    float t_test_y = abs(samp1Pos.at<float>(1) - samp2Pos.at<float>(1)) / Sp_y;
                    float t_test_z = abs(samp1Pos.at<float>(2) - samp2Pos.at<float>(2)) / Sp_z;

                    //0.001
                    float th = tTest[min(int(n1 + n2 - 2), 100)][4];
                    if(t_test_x < th && t_test_y < th && t_test_z < th)
                    {
                        //merge
                        if(pOBJ->mnObs > pCandidateObj->mnObs)
                        {
                            pOBJ->MergeObject(pCandidateObj,mpCurrentKeyFrame->mTimeStamp);
                            pOBJ->CalculateMeanAndStandard();
                            pOBJ->CalculatePosMeanAndStandard();
                            pCandidateObj->SetBad("double t-test merge");
                            pCandidateObj->mpReplaced = pOBJ;
                            mpMap->EraseObjectMap(pCandidateObj);
                            if(!mvUpdateObj.count(pOBJ))
                                mvUpdateObj.insert(pOBJ);
                        }
                        else
                        {
                            pCandidateObj->MergeObject(pOBJ,mpCurrentKeyFrame->mTimeStamp);
                            pCandidateObj->CalculateMeanAndStandard();
                            pCandidateObj->CalculatePosMeanAndStandard();
                            pOBJ->SetBad("double t-test merge");
                            pOBJ->mpReplaced = pCandidateObj;
                            mpMap->EraseObjectMap(pOBJ);
                            if(!mvUpdateObj.count(pCandidateObj))
                                mvUpdateObj.insert(pCandidateObj);
                            break;
                        }
                    }
                    
                }
            }

            pOBJ->mPossibleSameObj.clear();
            
        }
    }
}

void LocalMapping::MergeOverlapObjects()
{
    
    vector<Object_Map*> vObjs = mpMap->GetAllObjectMap();
    if(vObjs.empty())
        return;
    
    int nOBjs = vObjs.size();
    for(int i=0;i<nOBjs;i++)
    {
        Object_Map* pOBJ = vObjs[i];

        if(pOBJ->IsBad())
        {
            mpMap->EraseObjectMap(pOBJ);    
            continue;
        }
        
        //cout<<"CreatFrameId: "<<pOBJ->mnCreatFrameId<<" mnObs: "<<pOBJ->mnObs<<" mnClass: "<<pOBJ->mnClass<<endl;


        Eigen::Vector3d Pos1 = pOBJ->mShape.mTobjw.inverse().translation();
        float  lenth1_x =  pOBJ->mShape.a1;
        float  lenth1_y =  pOBJ->mShape.a2;
        float  lenth1_z =  pOBJ->mShape.a3;

        //get overlap Object 
        for(int j=0;j<nOBjs;j++)
        {
            Object_Map* pReObj = vObjs[j];

            if(pOBJ == pReObj || pReObj->IsBad())
                continue;
            
            Eigen::Vector3d Pos2 = pReObj->mShape.mTobjw.inverse().translation();

            //distance between two object center
            float  dist_x =  abs(Pos1(0) - Pos2(0));
            float  dist_y =  abs(Pos1(1) - Pos2(1));
            float  dist_z =  abs(Pos1(2) - Pos2(2));

            float  lenth_x =  pReObj->mShape.a1 + lenth1_x;
            float  lenth_y =  pReObj->mShape.a2 + lenth1_y;
            float  lenth_z =  pReObj->mShape.a3 + lenth1_z;

            if(dist_x < lenth_x && dist_y < lenth_y && dist_z < lenth_z)
            {   
                float volume1 = 8 * lenth1_x * lenth1_y * lenth1_z;
                float volume2 = 8 * pReObj->mShape.a1 * pReObj->mShape.a2 * pReObj->mShape.a3;
                float OverlapVolume = (lenth_x - dist_x) * (lenth_y - dist_y) * (lenth_z - dist_z);

                //overlap
                if(pOBJ->mnClass == pReObj->mnClass)
                {   
                    bool AppearSameTime = pOBJ->mmAppearSameTimes.find(pReObj) != pOBJ->mmAppearSameTimes.end();
                    
                    //Not appearing simultaneously, possibly multiple landmarks of the same object
                    if(!AppearSameTime)
                    {
                        if((OverlapVolume / volume1) > 0.3 || (OverlapVolume / volume2) > 0.3) 
                        {
                            // same object
                            if(pOBJ->mnObs >= pReObj->mnObs)
                            {
                                pOBJ->MergeObject(pReObj,mpCurrentKeyFrame->mTimeStamp);
                                pOBJ->CalculateMeanAndStandard();
                                pOBJ->CalculatePosMeanAndStandard();
                                pReObj->SetBad("same object");
                                pReObj->mpReplaced = pOBJ;
                                mpMap->EraseObjectMap(pReObj);
                                if(!mvUpdateObj.count(pOBJ))
                                    mvUpdateObj.insert(pOBJ);
                            }
                            else
                            {
                                pReObj->MergeObject(pOBJ,mpCurrentKeyFrame->mTimeStamp);
                                pReObj->CalculateMeanAndStandard();
                                pReObj->CalculatePosMeanAndStandard();
                                pOBJ->SetBad("same object");
                                pOBJ->mpReplaced = pReObj;
                                mpMap->EraseObjectMap(pOBJ);
                                if(!mvUpdateObj.count(pReObj))
                                    mvUpdateObj.insert(pReObj);
                                break;
                            }
                        }
                        else
                        {   
                            // false Object
            
                            if(pOBJ->mnObs >= pReObj->mnObs)
                            {   
                                pReObj->SetBad("false object");
                                mpMap->EraseObjectMap(pReObj);
                            }
                            else
                            {
                                pOBJ->SetBad("false object");
                                mpMap->EraseObjectMap(pOBJ);
                                break;
                            }
                        }
                    }  
                    
                }
                else if(Object_Map::MergeDifferentClass)
                {
                    if(pOBJ->mmAppearSameTimes.find(pReObj) == pOBJ->mmAppearSameTimes.end())
                    {
                        if((OverlapVolume / volume1) > 0.5 || (OverlapVolume / volume2) > 0.5) 
                        {
                            // same object
                            if(pOBJ->mnObs >= pReObj->mnObs)
                            {
                                pOBJ->MergeObject(pReObj,mpCurrentKeyFrame->mTimeStamp);
                                pOBJ->CalculateMeanAndStandard();
                                pOBJ->CalculatePosMeanAndStandard();
                                pReObj->SetBad("same object");
                                pReObj->mpReplaced = pOBJ;
                                mpMap->EraseObjectMap(pReObj);
                                if(!mvUpdateObj.count(pOBJ))
                                    mvUpdateObj.insert(pOBJ);
                            }
                            else
                            {
                                pReObj->MergeObject(pOBJ,mpCurrentKeyFrame->mTimeStamp);
                                pReObj->CalculateMeanAndStandard();
                                pReObj->CalculatePosMeanAndStandard();
                                pOBJ->SetBad("same object");
                                pOBJ->mpReplaced = pReObj;
                                mpMap->EraseObjectMap(pOBJ);
                                if(!mvUpdateObj.count(pReObj))
                                    mvUpdateObj.insert(pReObj);
                                break;
                            }
                        }
                    }


                }

                
            }

        }
        
    }
}

void LocalMapping::InsertKeyFrameAndImg(KeyFrame *pKF,const cv::Mat& img,const cv::Mat& Instance)
{
    unique_lock<mutex> lock(mNewDataNeRF);
    mlNewKeyFramesNeRF.push_back(pKF);
    mlNewImgNeRF.push_back(img.clone());
    mlNewInstanceNeRF.push_back(Instance.clone());

}

void LocalMapping::SetNeRFManager(nerf::NerfManagerOnline* pNeRFManager)
{
    mpNeRFManager = pNeRFManager;
}

void LocalMapping::NewDataToGPU()
{

    if(mlNewKeyFramesNeRF.empty())
        return;
    
    /* //----------------------------------------------------------------------
    //update old Frame pose ... Not use
    if(mvNeRFDataKeyFrames.size() > 10 && mvNeRFDataKeyFrames.size() % 10 == 0)
    {
        vector<Eigen::Matrix4f> updateTwc;
        size_t head = mvNeRFDataKeyFrames.size() - 10;
        for(size_t i=head; i<head+10;i++)
        {   
            KeyFrame* pKF = mvNeRFDataKeyFrames[i];
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            while(pKF->isBad())
            {
            //  cout << "bad parent" << endl;
                Tcw = Tcw*pKF->mTcp;
                pKF = pKF->GetParent();
            }
            Tcw = Tcw*pKF->GetPose();
            updateTwc.push_back(Converter::toMatrix4f(Tcw).inverse());
            //updateTwc.push_back(Converter::toMatrix4f(mvNeRFDataKeyFrames[i]->GetPoseInverse()));
        }
        mpNeRFManager->UpdateDataset(mCurImgId,10,updateTwc);
    }
    //---------------------------------------------------------------------- */

    //add new Frame pose
    KeyFrame* pKF;
    cv::Mat img;
    cv::Mat instance;
    {
        unique_lock<mutex> lock(mNewDataNeRF);
        pKF = mlNewKeyFramesNeRF.front();
        img = mlNewImgNeRF.front();
        instance = mlNewInstanceNeRF.front();
        mlNewKeyFramesNeRF.pop_front();
        mlNewImgNeRF.pop_front();
        mlNewInstanceNeRF.pop_front();
    }
    
    mvNeRFDataKeyFrames.push_back(pKF);
    Eigen::Matrix4f pose = Converter::toMatrix4f(pKF->GetPoseInverse());
    string timestamp = to_string(pKF->mTimeStamp);
    cv::Mat depth_img = cv::Mat::zeros(img.rows, img.cols,CV_32FC1);
    
    //Generate sparse depth maps
    if(mpNeRFManager->mbUseSparseDepth)
        pKF->GenerateSparseDepthImg(depth_img);
   
    mpNeRFManager->NewFrameToDataset(mCurImgId,timestamp,img,instance,depth_img,pose);
    //cout<<"KF_Id: "<<pKF->mnId<<" mCurImgId: "<<mCurImgId<<endl;
    
    mCurImgId += 1;
}

void LocalMapping::UpdateObjNeRF()
{
    if(mvUpdateObj.empty())
        return;
    
    for(auto it=mvUpdateObj.begin();it != mvUpdateObj.end();it++)
    {
        Object_Map* pObj = *it;

        if(pObj->IsBad())
            continue;

        //Calculate angle change
        if(pObj->mKeyFrameHistoryBbox_Temp.size() > 2 && pObj->twc_xy_last == Eigen::Vector2f::Zero())
        {   
            //first
            pObj->twc_xy_last = Converter::toVector3d(mpCurrentKeyFrame->GetCameraCenter()).cast<float>().head<2>();
            continue;
        }
        
        if(pObj->mKeyFrameHistoryBbox_Temp.size() < 10)
            continue;


        pObj->twc_xy = Converter::toVector3d(mpCurrentKeyFrame->GetCameraCenter()).cast<float>().head<2>();
        
        //Calculate angle changes
        g2o::SE3Quat Twobj = pObj->mShape.mTobjw.inverse();
        Eigen::Vector2f Twobj_t_xy = Twobj.translation().cast<float>().head<2>();
        Eigen::Vector2f v1 = pObj->twc_xy - Twobj_t_xy;
        Eigen::Vector2f v2 = pObj->twc_xy_last - Twobj_t_xy;
        float cosVal = v1.dot(v2) /(v1.norm()*v2.norm()); 
        float angle = acos(cosVal) * 180 / M_PI; 

        //No nerf created
        if(!pObj->haveNeRF)
        {   
            if(angle > 2 * mfAngleChange)
            {
                //create 
                int cls;
                Eigen::Matrix4f Tow;
                nerf::BoundingBox BBox;

                //NeRF Bbox attribute
                Tow = pObj->mShape.mTobjw.to_homogeneous_matrix().cast<float>();
                cls = pObj->mnClass;
                BBox.min = Eigen::Vector3f(-pObj->mShape.a1,-pObj->mShape.a2,-pObj->mShape.a3);
                BBox.max = Eigen::Vector3f(pObj->mShape.a1,pObj->mShape.a2,pObj->mShape.a3);

                size_t idx = mpNeRFManager->CreateNeRF(cls,Tow,BBox);
                
                pObj->haveNeRF = true;
                pObj->pNeRFIdx = idx;
                pObj->mTow_NeRF = Tow;
                pObj->BBox_NeRF = (cls == 41 || cls == 73) ? BBox.max * 1.2f : BBox.max * 1.1f;

                //2D bbox for sampling rays
                vector<nerf::FrameIdAndBbox> vFrameBbox;
                GetUpdateBbox(pObj,vFrameBbox);
                pObj->mKeyFrameHistoryBbox_Temp.clear();
                mpNeRFManager->UpdateNeRFBbox(idx,vFrameBbox,1);    

                cout<<"Create NeRF ... Id: "<<idx<<" Init Angle: "<<angle<<" Bbox Sizes: "<<vFrameBbox.size()<<endl;
                pObj->twc_xy_last = pObj->twc_xy;
            }
            else
            {
                continue;
            }
        
        }
        else
        {
            //Update 2D Bbox for sampling rays
            if(angle > mfAngleChange)
            {
                //update
                vector<nerf::FrameIdAndBbox> vFrameBbox;
                GetUpdateBbox(pObj,vFrameBbox);
                pObj->mKeyFrameHistoryBbox_Temp.clear();
                mpNeRFManager->UpdateNeRFBbox(pObj->pNeRFIdx,vFrameBbox,1);

                pObj->twc_xy_last = pObj->twc_xy;
            }
        }
    
    }

}

void LocalMapping::GetUpdateBbox(Object_Map* pObj, vector<nerf::FrameIdAndBbox>& vFrameBbox)
{
    vFrameBbox.clear();
    for(auto it = pObj->mKeyFrameHistoryBbox_Temp.begin();it != pObj->mKeyFrameHistoryBbox_Temp.end();it++)
    {
        double stamp = it->first;
        Bbox& box = it->second;

        int id = mpNeRFManager->GetFrameIdx(stamp);
        if(id == -1)
            continue;
        nerf::FrameIdAndBbox a;
        a.FrameId = uint32_t(id);
        a.x = box.x;
        a.y = box.y;
        a.w = box.width;
        a.h = box.height;
        vFrameBbox.push_back(a);
    }
}

} //namespace ORB_SLAM
