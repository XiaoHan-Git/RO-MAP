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

//#include <opencv2/ximgproc.hpp>

#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <unistd.h>
#include <mutex>

namespace ORB_SLAM2
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}

Eigen::Matrix4f GenerateToc(const float theta,const float phi,const float r)
{   
    float z = r * sin(phi * M_PI / 180.0f);
    float x = r * cos(phi * M_PI / 180.0f) * cos(theta * M_PI / 180.0f);
    float y = r * cos(phi * M_PI / 180.0f) * sin(theta * M_PI / 180.0f);
    Eigen::Vector3f t(x,y,z);
    Eigen::Vector3f z_axis = -t;
    z_axis.normalize();
    float r_v= (theta + 90.0f) * M_PI / 180.0f;
    Eigen::Vector3f x_axis(cos(r_v),sin(r_v),0);
    x_axis.normalize();
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    Eigen::Matrix4f Toc = Eigen::Matrix4f::Identity();
    Toc.col(0).head<3>() = x_axis;
    Toc.col(1).head<3>() = y_axis;
    Toc.col(2).head<3>() = z_axis;
    Toc.col(3).head<3>() = t;

    return Toc;
}


void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    pangolin::CreateWindowAndBind("RO-MAP: Map Viewer",1024,768);
    if(glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph",false,true);
    pangolin::Var<bool> menuLocalizationMode("menu.Localization",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);

    //RO-MAP
    pangolin::Var<bool> menuShowObject("menu.Show Objects",true,true);
    pangolin::Var<bool> menuShowObjectMPs("menu.Show Obj MPs",false,true);
    pangolin::Var<bool> menuShowBbox("menu.Show Bbox",true,true);
    pangolin::Var<bool> menuShowMesh("menu.Show Mesh",false,true);
    pangolin::Var<bool> menuShowObs("menu.Show Obs",false,true);
    pangolin::Var<bool> menuRotation("menu.360 Rotation",false,true);

    pangolin::Var<bool> menuSaveImg("menu.Save Img",false,false);
    int saveimgnum = 0;


    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
                );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    cv::namedWindow("RO-MAP: Current Frame Bbox");
    cv::namedWindow("RO-MAP: Current Instance Frame");
    bool bFollow = true;
    bool bLocalizationMode = false;
    
    //test
    //360 rotation
    vector<Eigen::Matrix4f> vToc;
    int theta_num = 240;
    // 360 -- 450 Turn a bit more
    float theta = 450 / float(theta_num);
    float cur_theta = 0.0f;
    float phi = 25;
    for(int i=0;i<theta_num;i++)
    {   
        cur_theta += theta;
        //room 1.6  black 0.7
        Eigen::Matrix4f Toc = GenerateToc(cur_theta,phi,2);
        vToc.push_back(Toc);
    }
    int RotationIdx = 0;
    bool initTtoc = true;
    bool RatotionComplate = false;

    Eigen::Vector3f restoc;

    while(1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        if(menuLocalizationMode && !bLocalizationMode)
        {
            mpSystem->ActivateLocalizationMode();
            bLocalizationMode = true;
        }
        else if(!menuLocalizationMode && bLocalizationMode)
        {
            mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
        }


        if(menuRotation)
        {
            if(initTtoc)
            {
                //Obtain the average center of all objects
                vector<Object_Map*> AllObjs = mpMapDrawer->mpMap->GetAllObjectMap();
                
                vector<float> allx;
                vector<float> ally;
                vector<float> allz;
                for(Object_Map* pObj : AllObjs)
                {
                    if(pObj->haveNeRF)
                    {
                        Eigen::Vector3f toc = pObj->mTow_NeRF.inverse().col(3).head<3>();
                        allx.push_back(toc[0]);
                        ally.push_back(toc[1]);
                        allz.push_back(toc[2]);
                    }
                }
                sort(allx.begin(),allx.end());
                sort(ally.begin(),ally.end());
                sort(allz.begin(),allz.end());
                float x = (*(allx.end()-1) + *allx.begin()) / 2.0f;
                float y = (*(ally.end()-1) + *ally.begin()) / 2.0f;
                float z = (*(allz.end()-1) + *allz.begin()) / 2.0f;
                restoc = Eigen::Vector3f(x,y,z);
                initTtoc = false;
            }

            Eigen::Matrix4f Toc = vToc[RotationIdx];
            Toc.col(3).head<3>() += restoc;

            Twc.m[0] = Toc(0,0);
            Twc.m[1] = Toc(1,0);
            Twc.m[2] = Toc(2,0);
            Twc.m[3]  = 0.0;

            Twc.m[4] = Toc(0,1);
            Twc.m[5] = Toc(1,1);
            Twc.m[6] = Toc(2,1);
            Twc.m[7]  = 0.0;

            Twc.m[8] = Toc(0,2);
            Twc.m[9] = Toc(1,2);
            Twc.m[10] = Toc(2,2);
            Twc.m[11]  = 0.0;

            Twc.m[12] = Toc(0,3);
            Twc.m[13] = Toc(1,3);
            Twc.m[14] = Toc(2,3);
            Twc.m[15]  = 1.0;

            //s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            if(!RatotionComplate)
            {
                ++RotationIdx;   
            }
            if(RotationIdx == vToc.size())
            {
                RotationIdx = vToc.size() - 1;
                RatotionComplate = true;
            }
            
            s_cam.Follow(Twc); 
        }


        d_cam.Activate(s_cam);

        glClearColor(1.0f,1.0f,1.0f,1.0f);

        //ground XYZ
        /* glBegin ( GL_LINES );
        glColor3f ( 1.0f,0.f,0.f );
        glVertex3f( 0,0,0 );
        glVertex3f( 1,0,0 );
        glColor3f( 0.f,1.0f,0.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,1,0 );
        glColor3f( 0.f,0.f,1.f);
        glVertex3f( 0,0,0 );
        glVertex3f( 0,0,1 );
        glEnd(); */
        
        mpMapDrawer->DrawCurrentCamera(Twc);
        if(menuShowKeyFrames || menuShowGraph)
            mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph);
        if(menuShowPoints)
            mpMapDrawer->DrawMapPoints();
        
        bool drawMps = false;
        bool drawMesh = false;
        bool drawBbox = false;
        bool drawObs = false;
        if(menuShowObjectMPs)
            drawMps = true;
        if(menuShowMesh)
            drawMesh = true;
        if(menuShowBbox)
            drawBbox = true;
        if(menuShowObs)
            drawObs = true;
        if(menuShowObject)
            mpMapDrawer->DrawObject(drawMps,drawMesh,drawBbox,drawObs);
        
        if (pangolin::Pushed(menuSaveImg)) 
        { 
            pangolin::SaveWindowOnRender("./output/"+ to_string(saveimgnum)+".png");
            saveimgnum++;
        }
        pangolin::FinishFrame();

        //cv::Mat imgray = mpFrameDrawer->DrawFrame();
        cv::Mat imcolor = mpFrameDrawer->DrawFrameBboxAndLines();
        cv::Mat iminstance = mpFrameDrawer->FrameInstance();

        if(!iminstance.empty() && !imcolor.empty())
        {
            //cv::imshow("ORB-SLAM2: Current Frame",imgray);
            cv::imshow("RO-MAP: Current Frame Bbox",imcolor);
            cv::imshow("RO-MAP: Current Instance Frame",iminstance);
            cv::waitKey(mT);
        }
    
        if(menuReset)
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            menuLocalizationMode = false;
            if(bLocalizationMode)
                mpSystem->DeactivateLocalizationMode();
            bLocalizationMode = false;
            bFollow = true;
            menuFollowCamera = true;
            mpSystem->Reset();
            menuReset = false;
        }

        if(Stop())
        {
            while(isStopped())
            {
                usleep(3000);
            }
        }

        if(CheckFinish())
            break;
    }

    SetFinish();
}

void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

}
