/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: RO-MAP
* Version: 1.0
* Author: Xiao Han
*/

#include "Tracking.h"
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"
#include <unistd.h>
#include"Optimizer.h"
#include"PnPsolver.h"
#include "ObjectFrame.h"
#include "ObjectMap.h"
#include<iostream>
#include <chrono>
#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbInitObjectMap(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    //RO-MAP

    //Line detect--------------------------------------------------------
    cout<<endl<<"Load RO-MAP Parameters..."<<endl;
    int ExtendBox = fSettings["ExtendBox"];
    mbExtendBox = ExtendBox;
    cout<<"ExtendBox: "<<mbExtendBox<<endl;

    int CheckBoxEdge = fSettings["CheckBoxEdge"];
    mbCheckBoxEdge = CheckBoxEdge;
    cout<<"CheckBoxEdge: "<<CheckBoxEdge<<endl;

    cv::FileNode node = fSettings["IgnoreCategory"];
    cout<<"IgnoreCategory: ";
    for(auto it = node.begin();it!=node.end();it++)
    {
        int number = *it;
        mvIgnoreCategory.insert(number);
        cout<<number<<" ";
    }
    cout<<endl;
    mnBoxMapPoints = fSettings["BoxMapPoints"];
    cout<<"BoxMapPoints: "<<mnBoxMapPoints<<endl;
    if(mnBoxMapPoints < 1)
    {
        cerr<<"Failed to load RO-MAP parameters, Please add parameters to yaml file..."<<endl;
        exit(0);
    }

    mnMinimumContinueObs = fSettings["Minimum.continue.obs"];
    cout<<"MinimumContinueObs: "<<mnMinimumContinueObs<<endl;

    AddMPsDistMultiple = fSettings["Add.MPs.distance.multiple"];
    cout<<"AddMPsDistMultiple: "<<AddMPsDistMultiple<<endl;

    Object_Map::MergeMPsDistMultiple = fSettings["Merge.MPs.distance.multiple"];
    cout<<"MergeMPsDistMultiple: "<<Object_Map::MergeMPsDistMultiple<<endl;

    int MergeDifferentClass  = fSettings["Merge.Different.class"];
    Object_Map::MergeDifferentClass = MergeDifferentClass;
    cout<<"MergeDifferentClass: "<<Object_Map::MergeDifferentClass<<endl;

    Object_Map::mfEIFthreshold = fSettings["EIFthreshold"];
    cout<<"EIFthreshold: "<<Object_Map::mfEIFthreshold<<endl;

    int CheckMPsObs = fSettings["CheckMPsObs"];
    Object_Map::mnCheckMPsObs = CheckMPsObs;
    cout<<"CheckMPsObs: "<<Object_Map::mnCheckMPsObs<<endl;

    Object_Map::mnEIFObsNumbers = fSettings["EIFObsNumbers"];
    cout<<"EIFObsNumbers: "<<Object_Map::mnEIFObsNumbers<<endl;

    LocalMapping::mfAngleChange = fSettings["NeRF.AngleChange"];
    cout<<"AngleChange: "<<LocalMapping::mfAngleChange<<endl;

    int numoctaves = 1;
    float octaveratio = 2.0;
    bool use_LSD = false;  // use LSD detector or edline detector    
    float line_length_thres = 15;
    mpLineDetect = new line_lbd_detect(numoctaves,octaveratio);

    mpLineDetect->use_LSD = use_LSD;
    mpLineDetect->line_length_thres = line_length_thres;
    mpLineDetect->save_imgs = false;
    mpLineDetect->save_txts = false;
    
    //read t-test data
    ifstream f;
    f.open("./lib/t_test.txt");
    if(!f.is_open())
    {
        cerr<<"Can't read t-test data"<<endl;
        exit(0);
    }
    for(int i=0;i<101;i++)
    {
        for(int j=0;j<4;j++)
            f >> tTest[i][j];  
    }
    f.close();

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im,const cv::Mat &ImgInstance, const double &timestamp, const string &strDataset)
{
    mImGray = im;
    //RO-MAP
    mImColor = im;
    mImgInstance = ImgInstance;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    
    //object-nerf-SLAM-------------------------START----------------------------
    vector<Bbox> Bboxs;
    //TODO online object detection 
    if(0)
    {
    }
    else
    {
        //offline read object detection----------------------------
        string sYolopath = strDataset + "/" + "bbox/" + to_string(timestamp) + ".txt";
        
        ifstream f;
        f.open(sYolopath);
        if(!f.is_open())
        {
            cout << "yolo_detection file open fail" << endl;
            exit(0);
        }
        
        //Read txt
        string line;
        float num;
        vector<float> row;
        while(getline(f,line))
        {
            stringstream s(line);
            while (s >> num)
            {
                row.push_back(num);
            }
            Bbox newBbox;
            newBbox.mnClass = row[0];

            //extend box
            if(mbExtendBox)
            {
                newBbox.x = max(0.0f,row[1] - 10);
                newBbox.y = max(0.0f,row[2] - 10);
                newBbox.width = min(float(mImGray.cols-1) - newBbox.x,row[3] + 20);
                newBbox.height = min(float(mImGray.rows-1) - newBbox.y,row[4] + 20); 
            }
            else
            {
                newBbox.x = row[1];
                newBbox.y = row[2];
                newBbox.width = row[3];
                newBbox.height = row[4];
            }
            newBbox.mfConfidence = row[5];
            Bboxs.push_back(newBbox);
            row.clear();
        }
        f.close();

    }
    
    /*Filter bad Bbox. Including the following situations:
        Close to image edge
        Overlap each other
        Bbox is too large
        */
    if(!Bboxs.empty())
    {
        vector<int> resIdx(Bboxs.size(),1);
        vector<Bbox> resBbox;
        for(size_t i=0;i<Bboxs.size();i++)
        {
            if(!resIdx[i])
                continue;

            Bbox &box = Bboxs[i];

            // There will be errors in target detection, and error categories can be filtered here
            if(mvIgnoreCategory.find(box.mnClass) != mvIgnoreCategory.end())
            {
                resIdx[i] = 0;
                continue;
            }

            if(mbCheckBoxEdge)
            {
                //Close to image edge
                if(box.x < 20 || box.x+box.width > im.cols-20 || box.y < 20 || box.y+box.height > im.rows-20)
                {  
                    if(box.area() < im.cols * im.rows * 0.05)
                    {
                        resIdx[i] = 0;
                        continue;
                    }
                    box.mbEdge = true;
                    if(box.area() < im.cols * im.rows * 0.1)
                        box.mbEdgeAndSmall = true;
                }
            }

            //Bbox is large than half of img
            if(box.area() > im.cols * im.rows * 0.5)
            {
                resIdx[i] = 0;
                continue;
            }
            else if(box.area() < im.cols * im.rows * 0.005)
            {
                resIdx[i] = 0;
                continue;
            }

            //Overlap each other
            for(size_t j=0;j<Bboxs.size();j++)
            {
                if(i == j || resIdx[j] == 0)
                    continue;
                
                Bbox &box_j = Bboxs[j];
                float SizeScale = min(box_j.area(),box.area()) / max(box_j.area(),box.area());
                if(SizeScale > 0.25)
                {
                    float overlap = (box & box_j).area();
                    float IOU = overlap / (box.area() + box_j.area() - overlap);
                    if(IOU > 0.4)
                    {
                        resIdx[i] = 0;
                        resIdx[j] = 0;
                        break;
                    }
                }
            } 
        }

        for(size_t i=0;i<Bboxs.size();i++)
        {
            if(resIdx[i])
                resBbox.push_back(Bboxs[i]);
        }
        Bboxs = resBbox;
    }
    
    if(!Bboxs.empty())
    {
        mCurrentFrame.mbDetectObject = true;
        mCurrentFrame.mvBbox = Bboxs;
        mCurrentFrame.UndistortFrameBbox();
        
        //Line detect-----------------------------------------------
        //using distort Img
        cv::Mat undistortImg = mImGray.clone();
        if(mDistCoef.at<float>(0)!=0.0)
        {
            cv::undistort(mImGray,undistortImg,mK,mDistCoef);
        }
        mpLineDetect->detect_raw_lines(undistortImg,mCurrentFrame.mvLines);
        
        vector<KeyLine> FilterLines;
        mpLineDetect->filter_lines(mCurrentFrame.mvLines,FilterLines);
        mCurrentFrame.mvLines = FilterLines;
        Eigen::MatrixXd LinesEigen(FilterLines.size(),4);
        for(size_t i=0;i<FilterLines.size();i++)
        {
            LinesEigen(i,0)=FilterLines[i].startPointX;
            LinesEigen(i,1)=FilterLines[i].startPointY;
            LinesEigen(i,2)=FilterLines[i].endPointX;
            LinesEigen(i,3)=FilterLines[i].endPointY;
        }
        mCurrentFrame.mLinesEigen = LinesEigen;
        
        //creat object_Frame---------------------------------------------------
        for(size_t i=0;i<mCurrentFrame.mvBboxUn.size();i++)
        {
            Object_Frame object_frame;
            object_frame.mnFrameId = mCurrentFrame.mnId;
            object_frame.mBbox = mCurrentFrame.mvBboxUn[i];
            object_frame.mnClass = object_frame.mBbox.mnClass;
            object_frame.mfConfidence = object_frame.mBbox.mfConfidence;
            mCurrentFrame.mvObjectFrame.push_back(object_frame);
        }
    }
    
    //Assign feature points and lines to detected objects
    mCurrentFrame.AssignFeaturesToBbox(mImgInstance);
    
    mCurrentFrame.AssignLinesToBbox();

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            //Initialization
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                //object-nerf-SLAM=======The main functions are in TrackLocalMap===============
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            
            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
            
            //Make the global coordinate system parallel to the horizontal plane
            //using GT pose only at initialization
            string sTimestmap = to_string(mCurrentFrame.mTimeStamp);
            sTimestmap = sTimestmap.substr(0,sTimestmap.size()-4);
            Eigen::Matrix<double,7,1> GTiw;
            bool haveGTpose = false;
            //RO-MAP get current camera groundtruth by timestamp.
            for(size_t i=0;i<mvGroundtruthPose.size();i++)
            {
                
                string GTtimestamp = to_string(mvGroundtruthPose[i][0]);
                //GTtimestamp = GTtimestamp.substr(0,GTtimestamp.size()-2);
                GTtimestamp = GTtimestamp.substr(0,GTtimestamp.size()-4);
                if(sTimestmap == GTtimestamp)
                {
                    GTiw(0) = mvGroundtruthPose[i][1];
                    GTiw(1) = mvGroundtruthPose[i][2];
                    GTiw(2) = mvGroundtruthPose[i][3];
                    GTiw(3) = mvGroundtruthPose[i][4];
                    GTiw(4) = mvGroundtruthPose[i][5];
                    GTiw(5) = mvGroundtruthPose[i][6];
                    GTiw(6) = mvGroundtruthPose[i][7];
                    haveGTpose = true;
                    break;
                }
            }
            
            if(haveGTpose)
            {   
                //Only the rotation information is used to make the world coordinates parallel to the ground plane 
                cout<<"GTiw: "<<GTiw<<endl;
                g2o::SE3Quat SE3GroundtruthPose_iw(GTiw);
                cv::Mat Tiw = Converter::toCvMat(SE3GroundtruthPose_iw);
                cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                cv::Mat Rwi = Riw.t();
                cv::Mat Twi = cv::Mat::eye(4,4,CV_32F);
                Rwi.copyTo(Twi.rowRange(0,3).colRange(0,3));
                //Twi.rowRange(0,3).colRange(0,3) = Rwi;
                mInitialFrame.SetPose(mInitialFrame.mTcw * Twi);
                mCurrentFrame.SetPose(mCurrentFrame.mTcw * Twi);

                for(size_t i=0; i<mvIniMatches.size();i++)
                {             
                    if(mvIniMatches[i]<0)
                        continue;
                
                    cv::Mat worldPos(mvIniP3D[i]);
                    worldPos = Riw * worldPos;
                    mvIniP3D[i].x = worldPos.at<float>(0);
                    mvIniP3D[i].y = worldPos.at<float>(1);
                    mvIniP3D[i].z = worldPos.at<float>(2);
                }

            }
            else
            {
                cout<<"No GT pose Init"<<endl;
                //If there is no GT pose, 
                //the camera needs to be parallel to the ground plane during initialization
            } 

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 2.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    auto startTime = std::chrono::system_clock::now();


    //object-nerf-SLAM-----------------The main functions are as follows--------------------------------------
    bool FilterOutlier = true;
    mvPointsToDrawer.clear();

    if(mCurrentFrame.mbDetectObject)
    {
        vector<Object_Frame>& Objects = mCurrentFrame.mvObjectFrame;
        //1. After optimize pose, associate MapPoints with objects using KeyPoints;
        for(Object_Frame& obj : Objects)
        {   
            obj.mSumPointsPos = cv::Mat::zeros(3,1,CV_32F);
            
            for(const int& i : obj.mvIdxKeyPoints)
            {
                if(mCurrentFrame.mvpMapPoints[i])
                    if(FilterOutlier && mCurrentFrame.mvbOutlier[i])
                    {
                        continue;
                    }
                    else
                    {
                        obj.mvpMapPoints.push_back(mCurrentFrame.mvpMapPoints[i]);
                        cv::Mat MPworldPos = mCurrentFrame.mvpMapPoints[i]->GetWorldPos();
                        obj.mSumPointsPos += MPworldPos;

                        // to FrameDraw
                        cv::Point point = mCurrentFrame.mvKeysUn[i].pt;
                        mvPointsToDrawer.push_back(point);
                        
                    }
            }
            
            if(obj.mvpMapPoints.size() < mnBoxMapPoints)
            {
                obj.mbBad = true;
                continue;
            }

            //on edge 
            if(obj.mBbox.mbEdge)
                if(obj.mvpMapPoints.size() < mnBoxMapPoints * 2)
                {
                    obj.mbBad = true;
                    continue;
                }

            //2. filter mappoints using Boxplot
            if(!obj.mvpMapPoints.empty())
            {
                obj.FilterMPByBoxPlot(mCurrentFrame.mTcw);
            }

            //3. Calculate object  information
            //Calculate the mean and standard deviation
            obj.CalculateMeanAndStandard();

            //Construct Bbox by reprojecting MapPoints, for data association
            obj.ConstructBboxByMapPoints(mCurrentFrame);

        }

        //deal ObjectMap
        unique_lock<mutex> lock(mpMap->mMutexObjects);

        mvNewOrChangedObj.clear();
        //4. first init object map
        if(!mbInitObjectMap && mCurrentFrame.mnId >  mInitialFrame.mnId + 3)
        {
            if(InitObjectMap())
                mbInitObjectMap = true;
        } 
        else if (mbInitObjectMap)
        {
            //5. data association
            //Please refer to our previous work for theoretical.

            //Two strategies  1. Consecutive association    2. Non-consecutive association

            //update mLastFrame.mvObjectMap(backend merge object)
            for(size_t i=0;i<mLastFrame.mvObjectMap.size();i++)
            {
                if(mLastFrame.mvObjectMap[i])
                {
                    Object_Map* pObj = mLastFrame.mvObjectMap[i];

                    Object_Map* pReObj = pObj->GetReplaced();
                        if(pReObj)
                            mLastFrame.mvObjectMap[i] = pReObj;
                }
            }
            
            //Separate two types of Object_Map
            set<Object_Map*> sObjsIF;
            set<Object_Map*> sObjsNIF;
        
            for(Object_Map* pObj : mLastFrame.mvObjectMap)
            {
                if(pObj && !pObj->IsBad())
                    sObjsIF.insert(pObj);
            }
            for(Object_Map* pObj : mpMap->GetAllObjectMap())
            {   
                if(pObj->IsBad())
                    continue;
                if(sObjsIF.find(pObj) != sObjsIF.end())
                    continue;
                else
                {   
                    //Successful inter frame matching for at least three consecutive frames
                    if(pObj->mnObs < mnMinimumContinueObs)
                    {
                        pObj->SetBad("No inter-frame matching");
                        continue;
                    }    

                    sObjsNIF.insert(pObj);
                    //for Non interframe Data Association
                    pObj->ConstructBboxByMapPoints(mCurrentFrame);
                }
                    
            }
            
            int nObjFrame = mCurrentFrame.mvObjectFrame.size();
            mCurrentFrame.mvObjectMap = vector<Object_Map*>(nObjFrame,static_cast<Object_Map*>(NULL));

            auto startAssTime = std::chrono::system_clock::now();

            /* cout<<"sObjsIF.size(): "<<sObjsIF.size()<<endl;
            cout<<"sObjsNIF.size(): "<<sObjsNIF.size()<<endl;
            cout<<"Objects.size(): "<<Objects.size()<<endl; */
            //cout<<"------------------------------------------"<<endl;
            
            for(size_t i=0;i<Objects.size();i++)
            {
                Object_Frame& obj = Objects[i];
                if(obj.mbBad)
                    continue;

                bool bIFass = false;
                bool bNIFass = false;

                Object_Map* AssOBJ = static_cast<Object_Map*>(NULL);
                vector<Object_Map*> possibleOBJ;

                //1. Consecutive association

                Object_Map* IoUassOBJ = static_cast<Object_Map*>(NULL);
                float fMaxIoU = 0;

                Object_Map* MaxMPassOBJ = static_cast<Object_Map*>(NULL);
                int nMaxMPs = 0;
                set<MapPoint*> ObjMPs(obj.mvpMapPoints.begin(),obj.mvpMapPoints.end());
                
                for(Object_Map* pObjMap : sObjsIF)
                {
                    //Inter frame IOU data association
                    if(pObjMap->IsBad())
                        continue;
                    if(pObjMap->mnClass != obj.mnClass)
                        continue;
                    if(pObjMap->mnlatestObsFrameId == mCurrentFrame.mnId)
                        continue;

                    cv::Rect predictBbox;
                    //CurrentFrame Bbox prediction
                    if(pObjMap->mLastBbox != pObjMap->mLastLastBbox)
                    {
                        float top_left_x = pObjMap->mLastBbox.x * 2 - pObjMap->mLastLastBbox.x;
                        float top_left_y = pObjMap->mLastBbox.y * 2 - pObjMap->mLastLastBbox.y;
                
                        if(top_left_x < mCurrentFrame.mnMinX)
                            top_left_x = mCurrentFrame.mnMinX;
                        if(top_left_y < mCurrentFrame.mnMinY)
                            top_left_y = mCurrentFrame.mnMinY;  

                        float width = pObjMap->mLastBbox.width * 2 - pObjMap->mLastLastBbox.width;
                        float height = pObjMap->mLastBbox.height * 2 - pObjMap->mLastLastBbox.height;

                        if(width > mCurrentFrame.mnMaxX - top_left_x)
                            width = mCurrentFrame.mnMaxX - top_left_x;
                        
                        if(height > mCurrentFrame.mnMaxY - top_left_y)
                            height = mCurrentFrame.mnMaxY - top_left_y;
                        
                        predictBbox = cv::Rect(top_left_x,top_left_y,width,height);

                    }
                    else
                    {
                        predictBbox = pObjMap->mLastBbox;
                    }

                    float IoUarea = (predictBbox & obj.mBbox).area(); 
                    IoUarea = IoUarea / float((predictBbox.area() + obj.mBbox.area() - IoUarea));
                    if(IoUarea > 0.5 && IoUarea > fMaxIoU)
                    {
                        fMaxIoU = IoUarea;
                        IoUassOBJ = pObjMap;  
                    }
                    
                    //Inter frame MapPoints data association
                    int nShareMP = 0;
                    //Data association is not performed when there are too few MapPoints
                    if(ObjMPs.size() > 6)
                    {
                        for(MapPoint* pMP : pObjMap->mvpMapPoints)
                        {
                            if(ObjMPs.find(pMP) != ObjMPs.end())
                            ++nShareMP;
                        }
                    
                        if(nShareMP > ObjMPs.size() / 3 && nShareMP> nMaxMPs)
                        {
                            nMaxMPs = nShareMP;
                            MaxMPassOBJ = pObjMap;
                        }
                    }

                }
                
                //have association
                if(fMaxIoU > 0.7)
                {
                    if(IoUassOBJ->whetherAssociation(obj,mCurrentFrame))
                    {
                        AssOBJ = IoUassOBJ;
                        bIFass = true;
                    }
                    else
                        bIFass = false;
                }
                else if(fMaxIoU > 0 && nMaxMPs > 0)
                {   
                    //same association
                    if(IoUassOBJ == MaxMPassOBJ)
                    {
                        if(IoUassOBJ->whetherAssociation(obj,mCurrentFrame))
                        {
                            AssOBJ = IoUassOBJ;
                            bIFass = true;
                        }
                        else
                            bIFass = false;
                    }
                    else
                    {
                        // have association but not same
                        bIFass = false;
                        obj.mbBad = true;
                    }
                }
                else if(fMaxIoU == 0 && nMaxMPs==0)
                {
                    bIFass = false;
                }
                else
                {

                    if(fMaxIoU > 0)
                    {
                        if(IoUassOBJ->whetherAssociation(obj,mCurrentFrame))
                        {
                            AssOBJ = IoUassOBJ;
                            bIFass = true;
                        }
                        else
                            bIFass = false;
                    }
                    else
                    {
                        if(MaxMPassOBJ->whetherAssociation(obj,mCurrentFrame))
                        {
                            AssOBJ = MaxMPassOBJ;
                            bIFass = true;
                        }
                        else
                            bIFass = false;
                    }
                }
                
                // Non-consecutive Association
                for(Object_Map* pObjMap : sObjsNIF)
                {   
                    //cout<<"pObjMap IsBad: "<<pObjMap->IsBad()<<endl;
                    if(pObjMap->IsBad() || pObjMap->mnClass != obj.mnClass)
                        continue;

                    if(pObjMap->mnlatestObsFrameId == mCurrentFrame.mnId)
                        continue;

                    //Data association is not performed when there are too few MapPoints
                    int nShareMP = 0;
                    if(ObjMPs.size() > 6)
                    {
                        for(MapPoint* pMP : pObjMap->mvpMapPoints)
                        {
                            if(ObjMPs.find(pMP) != ObjMPs.end())
                                ++nShareMP;
                        }
                    
                        if(nShareMP > ObjMPs.size() / 3)
                        {
                            possibleOBJ.push_back(pObjMap);
                            continue;
                        }
                    }

                    int nobs = pObjMap->mnObs;
                    //t-test 
                    float tx,ty,tz;

                    tx = abs(pObjMap->mHistoryPosMean.at<float>(0) - obj.mPosMean.at<float>(0));
                    tx = sqrt(nobs) * tx / pObjMap->mfPosStandardX;
                    ty = abs(pObjMap->mHistoryPosMean.at<float>(1) - obj.mPosMean.at<float>(1));
                    ty = sqrt(nobs) * ty / pObjMap->mfPosStandardY;
                    tz = abs(pObjMap->mHistoryPosMean.at<float>(2) - obj.mPosMean.at<float>(2));
                    tz = sqrt(nobs) * tz / pObjMap->mfPosStandardZ;
                    // Degrees of freedom.
                    int deg = min(100,nobs-1);

                    if(pObjMap->mnObs > 6)
                    {
                        //0.05
                        float th = tTest[deg][2];
                        if(tx<th && ty<th && tz<th)
                        {
                            possibleOBJ.push_back(pObjMap);
                            continue;
                        }

                    }

                    //check IoU, reproject associate
                    float IoUarea = (pObjMap->mMPsProjectRect & obj.mBbox).area();
                    IoUarea = IoUarea / float((pObjMap->mMPsProjectRect.area() + obj.mBbox.area() - IoUarea));
                    if(IoUarea > 0.3)
                    {   
                        //0.001
                        float th = tTest[deg][4];
                        if(tx<th && ty<th && tz<th)
                        {
                            possibleOBJ.push_back(pObjMap);
                            continue;
                        }
                        else if( (tx+ty+tz) / 3 < 2 * th)
                        {
                            possibleOBJ.push_back(pObjMap);
                            continue;
                        }
                    }
                }

                //try possible object
                if(!bIFass && !possibleOBJ.empty())
                {
                    sort(possibleOBJ.begin(),possibleOBJ.end(),[](const Object_Map* left,const Object_Map* right){return left->mnObs < right->mnObs;});

                    for(int i=possibleOBJ.size()-1;i>=0;i--)
                    {
                        if(possibleOBJ[i]->whetherAssociation(obj,mCurrentFrame))
                        {
                            AssOBJ = possibleOBJ[i];
                            bNIFass = true;
                            break;
                        }
                    }
                    if(bNIFass)
                    {
                        for(int i=possibleOBJ.size()-1;i>=0;i--)
                        {
                            if(possibleOBJ[i] == AssOBJ)
                            {
                                continue;
                            }
                            //judge in the backend
                            AssOBJ->mPossibleSameObj[possibleOBJ[i]]++;
                        }
                    }
                }
                else if(!possibleOBJ.empty())
                {
                    for(int i=possibleOBJ.size()-1;i>=0;i--)
                        {
                            if(possibleOBJ[i] == AssOBJ)
                            {
                                continue;
                            }
                            //judge in the backend
                            AssOBJ->mPossibleSameObj[possibleOBJ[i]]++;
                        }
                }

                //update 2D information
                if(bIFass || bNIFass)
                {
                    //cout<<"bIFass: "<<bIFass<<" bNIFass: "<<bNIFass<<endl;
                    AssOBJ->mnlatestObsFrameId = mCurrentFrame.mnId;
                    AssOBJ->mnObs++;
                    if(bIFass)
                        AssOBJ->mLastLastBbox = AssOBJ->mLastBbox;
                    else
                        AssOBJ->mLastLastBbox = obj.mBbox;
                    AssOBJ->mLastBbox = obj.mBbox;
                    AssOBJ->mlatestFrameLines = obj.mLines;
                    AssOBJ->mvHistoryPos.push_back(obj.mPosMean);
                    
                    bool checkMPs = false;
                    SE3Quat Tobjw;
                    float Maxdist_x = 0;
                    float Maxdist_y = 0;
                    float Maxdist_z = 0;
                    if(AssOBJ->mvpMapPoints.size() > 20)
                    {   
                        checkMPs = true;
                        if(AssOBJ->mbFirstInit)
                        {
                            Tobjw = AssOBJ->mTobjw;
                            Maxdist_x = AssOBJ->mfLength;
                            Maxdist_y = AssOBJ->mfLength;
                            Maxdist_z = AssOBJ->mfLength;
                        }
                        else
                        {   //more accurate
                            Tobjw = AssOBJ->mShape.mTobjw;
                            Maxdist_x = AssOBJ->mShape.a1;
                            Maxdist_y = AssOBJ->mShape.a2;
                            Maxdist_z = AssOBJ->mShape.a3;
                        }
                    }

                    //associate ObjectMap and MapPoints
                    for(size_t j=0;j<obj.mvpMapPoints.size();j++)
                    {   
                        MapPoint* pMP = obj.mvpMapPoints[j];
                        if(pMP->isBad())
                            continue;

                        if(checkMPs)
                        {
                            // check position
                            Eigen::Vector3d ObjPos = Tobjw * Converter::toVector3d(pMP->GetWorldPos());
                            /* float dist = sqrt(ObjPos(0) * ObjPos(0) + ObjPos(1) * ObjPos(1) + ObjPos(2) * ObjPos(2));
                            if(dist > 1.1 * Maxdist)
                                continue; */
                            if(abs(ObjPos(0)) > AddMPsDistMultiple * Maxdist_x || abs(ObjPos(1)) > AddMPsDistMultiple * Maxdist_y || abs(ObjPos(2)) > AddMPsDistMultiple * Maxdist_z)
                                continue;

                        }
                        //new MapPoint
                        AssOBJ->AddNewMapPoints(pMP);
                    }

                    AssOBJ->UpdateMapPoints();
                    mCurrentFrame.mvObjectMap[i] = AssOBJ;
                    //cout <<"FrameId: "<<mCurrentFrame.mnId <<" ObjId: "<< AssOBJ->mnId<<endl;
                    mvNewOrChangedObj.push_back(AssOBJ);
                }
                else
                {
                    //cout<<"creat new ObjectMap: "<<obj.mnClass<<endl;
                    //creat new ObjectMap
                    Object_Map* newObjMap = new Object_Map(mpMap);
                    newObjMap->mnCreatFrameId = mCurrentFrame.mnId;
                    newObjMap->mnlatestObsFrameId = mCurrentFrame.mnId;
                    newObjMap->mnObs++;
                    newObjMap->mnClass = obj.mnClass;
                    newObjMap->mLastBbox = obj.mBbox;
                    newObjMap->mLastLastBbox = obj.mBbox;
                    newObjMap->mlatestFrameLines = obj.mLines;
                    newObjMap->mvHistoryPos.push_back(obj.mPosMean);

                    //associate ObjectMap and MapPoints
                    for(size_t j=0;j<obj.mvpMapPoints.size();j++)
                    {   
                        MapPoint* pMP = obj.mvpMapPoints[j];
                        if(pMP->isBad())
                            continue;
                        //new MapPoint
                        newObjMap->AddNewMapPoints(pMP);
                    }
                    
                    // Calculate the mean and standard deviation
                    newObjMap->UpdateMapPoints();

                    mCurrentFrame.mvObjectMap[i] = newObjMap;
                    mvNewOrChangedObj.push_back(newObjMap);
                    mpMap->AddObjectMap(newObjMap);
                    
                }
            }
            auto endAssTime = std::chrono::system_clock::now();
            //cout<<"ObjectMapTime: "<<std::chrono::duration_cast<chrono::microseconds>(endAssTime - startAssTime).count()<<endl;
            Asstime.push_back(std::chrono::duration_cast<chrono::microseconds>(endAssTime - startAssTime).count());
            
        }
    
        //End of association, update object_map
        auto startObjectMapTime = std::chrono::system_clock::now();
        //Has been initialized and has a new association
        if(mbInitObjectMap && !mvNewOrChangedObj.empty())
        {   
            
            //6. update ObjectMap
            for(size_t i=0;i < mvNewOrChangedObj.size();i++)
            {
                Object_Map* pObj = mvNewOrChangedObj[i];
                //step1. Filter outlier
                pObj->FilterOutlier(mCurrentFrame);
                pObj->EIFFilterOutlier();
                
                //step2. Calculate (pos) MeanAndStandard
                pObj->CalculateMeanAndStandard();

                pObj->CalculatePosMeanAndStandard();
                
                //step3. Calculate Translation and Rotation
                pObj->CalculateObjectPose(mCurrentFrame);
                
                //step4. update covisibility relationship
                pObj->UpdateCovRelation(mvNewOrChangedObj);

                //step5. History BBox
                pObj->InsertHistoryBboxAndTwc(mCurrentFrame);

            }
        }
        auto endObjectTime = std::chrono::system_clock::now();
        //cout<<"ObjectMapTime: "<<std::chrono::duration_cast<chrono::milliseconds>(endObjectTime - startObjectMapTime).count()<<endl;
        //cout<<"ObjectTime: "<<std::chrono::duration_cast<chrono::milliseconds>(endObjectTime - startTime).count()<<endl;
        //cout <<"--------------------------------------------------------" <<endl;
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    //object-nerf-slam
    if(!mvNewOrChangedObj.empty())
        mpLocalMapper->InsertKeyFrameAndImg(pKF,mImColor,mImgInstance);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

//Use only the first frame to align the ground plane
void Tracking::LoadGroundtruthPose(const string &strDataset)
{
    string GTpath = strDataset + "/groundtruth.txt";
    ifstream f;
        f.open(GTpath);
        if(!f.is_open())
        {
            cout << "groundtruth file open fail" << endl;
            exit(0);
        }
        string line;
        double num;
        vector<double> row;
        string s0;
        getline(f, s0);
        while(getline(f,line))
        {
            stringstream s(line);
            while (s >> num)
            {
                row.push_back(num);
            }
            
            mvGroundtruthPose.push_back(row);
            row.clear();
        }
        f.close();
}

bool Tracking::InitObjectMap()
{

    int nObjFrame = mCurrentFrame.mvObjectFrame.size();
    mCurrentFrame.mvObjectMap = vector<Object_Map*>(nObjFrame,static_cast<Object_Map*>(NULL));

    vector<Object_Frame>& ObjFrame = mCurrentFrame.mvObjectFrame;

    for(int i=0;i<nObjFrame;i++)
    {
        if(ObjFrame[i].mbBad)
            continue;
        //The mappoints required for initialization are doubled
        if(ObjFrame[i].mvpMapPoints.size() < 10)
        {
            ObjFrame[i].mbBad = true;
            continue;
        }
        
        //creat new ObjectMap
        Object_Map* newObjMap = new Object_Map(mpMap);
        newObjMap->mnCreatFrameId = mCurrentFrame.mnId;
        newObjMap->mnlatestObsFrameId = mCurrentFrame.mnId;
        newObjMap->mnObs++;
        newObjMap->mnClass = ObjFrame[i].mnClass;
        newObjMap->mLastBbox = ObjFrame[i].mBbox;
        newObjMap->mLastLastBbox = ObjFrame[i].mBbox;
        newObjMap->mlatestFrameLines = ObjFrame[i].mLines;
        newObjMap->mvHistoryPos.push_back(ObjFrame[i].mPosMean);
        
        //associate ObjectMap and MapPoints
        for(size_t j=0;j<ObjFrame[i].mvpMapPoints.size();j++)
        {
            if(ObjFrame[i].mvpMapPoints[j]->isBad())
                continue;
            MapPoint* pMP = ObjFrame[i].mvpMapPoints[j];

            //new MapPoint
            newObjMap->AddNewMapPoints(pMP);
        }

        // Calculate the mean and standard deviation
        newObjMap->UpdateMapPoints();

        mCurrentFrame.mvObjectMap[i] = newObjMap;
        mvNewOrChangedObj.push_back(newObjMap);
        mpMap->AddObjectMap(newObjMap);
        
    }

    if(!mvNewOrChangedObj.empty())
    {
        cout<< "Init Object Map successful"  <<endl;
        return true;
    }  
    else 
        return false;

}

} //namespace ORB_SLAM
