/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: SQ-SLAM
* Version: 1.0
* Created: 05/20/2022
* Author: Xiao Han
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    std::sort(vpKFs.begin(),vpKFs.end(),[](KeyFrame* pKF1, KeyFrame* pKF2){return pKF1->mnId<pKF2->mnId;});
    
    if(bDrawKF)
    {
        cv::Mat last_twc;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            cv::Mat ttwc = Twc.colRange(0,3).row(3);
            if(i==0)
                last_twc = ttwc;
            else
            {   
                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f,1.0f,0.0f);
                glBegin(GL_LINES);
                glVertex3f(last_twc.at<float>(0),last_twc.at<float>(1),last_twc.at<float>(2));
                glVertex3f(ttwc.at<float>(0),ttwc.at<float>(1),ttwc.at<float>(2));
                last_twc = ttwc;
                glEnd();
            }

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

//RO-MAP
void MapDrawer::DrawObject(bool drawPoints, bool drawMesh, bool drawBbox,bool drawObs)
{
    vector<Object_Map*> AllObjs = mpMap->GetAllObjectMap();
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    map<double,KeyFrame*> mappKFs;

    std::transform(
    vpKFs.begin(), vpKFs.end(),
    std::inserter(mappKFs, mappKFs.end()),
    [](KeyFrame* pKF){return std::make_pair(pKF->mTimeStamp,pKF);});


    if(AllObjs.empty())
        return;
    
    const float w = mKeyFrameSize;

    const float linewidth = mKeyFrameLineWidth;

    // color.
    vector<vector<GLint>> colors = { {135,0,248},
                                     {255,0,253},
                                     {4,254,119},
                                     {255,126,1},
                                     {0,112,255},
                                     {0,250,250},
                                     {88,180,27},
                                     {27,45,221}};

    for(int i=0;i<AllObjs.size();i++)
    {
        if(AllObjs[i]->IsBad() || AllObjs[i]->mbFirstInit || AllObjs[i]->mnObs <15)
            continue;

        glPushMatrix();
        Eigen::Matrix4d pose;
        
        pose =  AllObjs[i]->mShape.mTobjw.inverse().to_homogeneous_matrix();
        float shape_a1 = AllObjs[i]->mShape.a1;
        float shape_a2 = AllObjs[i]->mShape.a2;
        float shape_a3 = AllObjs[i]->mShape.a3;

        if(AllObjs[i]->haveNeRF)
        {
            pose = AllObjs[i]->mTow_NeRF.inverse().cast<double>();
            shape_a1 = AllObjs[i]->BBox_NeRF[0];
            shape_a2 = AllObjs[i]->BBox_NeRF[1];
            shape_a3 = AllObjs[i]->BBox_NeRF[2];
        }
      
        glMultMatrixd(pose.data());

        vector<GLint> color = colors[AllObjs[i]->mnClass % 8];
        if(drawBbox)
        {
            vector<GLint> color = colors[AllObjs[i]->mnClass % 8];
            glLineWidth(linewidth);
            glBegin(GL_LINES);
            glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0);
            Eigen::Vector3f point;
            float a1 = shape_a1;
            float a2 = shape_a2;
            float a3 = shape_a3;
            //1
            glVertex3f(a1, a2, a3);
            glVertex3f(-a1, a2, a3);
            //2
            glVertex3f(-a1, a2, a3);
            glVertex3f(-a1, -a2, a3);
            //3
            glVertex3f(-a1, -a2, a3);
            glVertex3f(a1, -a2, a3);
            //4
            glVertex3f(a1, -a2, a3);
            glVertex3f(a1, a2, a3);

            //5
            glVertex3f(a1, a2, -a3);
            glVertex3f(-a1, a2, -a3);
            //6
            glVertex3f(-a1, a2, -a3);
            glVertex3f(-a1, -a2, -a3);
            //7
            glVertex3f(-a1, -a2, -a3);
            glVertex3f(a1, -a2, -a3);
            //8
            glVertex3f(a1, -a2, -a3);
            glVertex3f(a1, a2, -a3);

            //1
            glVertex3f(a1, a2, a3);
            glVertex3f(a1, a2, -a3);
            //2
            glVertex3f(-a1, a2, a3);
            glVertex3f(-a1, a2, -a3);
            //3
            glVertex3f(-a1, -a2, a3);
            glVertex3f(-a1, -a2, -a3);
            //4
            glVertex3f(a1, -a2, a3);
            glVertex3f(a1, -a2, -a3);
            glEnd();

            //XYZ Coordinate
            glLineWidth(linewidth);
            glBegin ( GL_LINES );
            glColor3f ( 1.0f,0.f,0.f );
            glVertex3f( 0,0,0 );
            glVertex3f( w,0,0 );
            glColor3f( 0.f,1.0f,0.f);
            glVertex3f( 0,0,0 );
            glVertex3f( 0,w,0 );
            glColor3f( 0.f,0.f,1.f);
            glVertex3f( 0,0,0 );
            glVertex3f( 0,0,w );
            glEnd();
        }
        

        if(drawMesh && AllObjs[i]->haveNeRF)
        {
            size_t idx = AllObjs[i]->pNeRFIdx;
            mpNeRFManager->DrawMesh(idx);
        }

        glPopMatrix();

        if(drawPoints)
        {
            glPointSize(mPointSize+1);
            glBegin(GL_POINTS);
            glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0);

            vector<MapPoint*> pMPs = AllObjs[i]->mvpMapPoints;
            for(size_t i=0, iend=pMPs.size(); i<iend;i++)
            {
                //if(pMPs[i]->isBad())
                //    continue;
                cv::Mat pos = pMPs[i]->GetWorldPos();
                glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
            }
            glEnd();
        }

        //Connection between objects and training KFs
        if(drawObs && AllObjs[i]->haveNeRF)
        {
            vector<GLint> color = colors[AllObjs[i]->mnClass % 8];
            for(auto& p : AllObjs[i]->mKeyFrameHistoryBbox) 
            {
                if(mappKFs.count(p.first)) 
                {
                    KeyFrame* pKF =mappKFs[p.first];
                    cv::Mat Twc = pKF->GetPoseInverse().t();
                    cv::Mat ttwc = Twc.colRange(0,3).row(3);
                    
                    Eigen::Vector3f ttwc_eig;
                    ttwc_eig << ttwc.at<float>(0), ttwc.at<float>(1), ttwc.at<float>(2);

                    glLineWidth(0.1f*linewidth);
                    glBegin(GL_LINES);
                    glColor3f(color[0]/255.0, color[1]/255.0, color[2]/255.0);
                    glVertex3f( ttwc.at<float>(0),ttwc.at<float>(1),ttwc.at<float>(2));

                    Eigen::Vector3f two = pose.col(3).head<3>().cast<float>();
                    //two = (two - ttwc_eig) * 0.8 + ttwc_eig;
                    glVertex3f(two[0],two[1],two[2]);
                    glEnd();
                }
                
            }

        }
        
    }

}

void MapDrawer::SetNeRFManager(nerf::NerfManagerOnline* pNeRFManager)
{
    mpNeRFManager = pNeRFManager;
}

} //namespace ORB_SLAM
