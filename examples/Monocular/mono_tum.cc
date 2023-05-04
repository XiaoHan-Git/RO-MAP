/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* Modification: RO-MAP
* Version: 1.0
* Created: 12/19/2022
* Author: Xiao Han
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
vector<string> &vstrInstanceFilenames, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_nerf_config path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrInstanceFilenames;
    vector<double> vTimestamps;
    const string strDataset = string(argv[3]);
    const string strFile = string(argv[3]) + "/img.txt";
    const string strDatasetConfig = string(argv[3]) + "/config.yaml";
    const string strNeRFConfig = string(argv[2]);

    LoadImages(strFile, vstrImageFilenames,vstrInstanceFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],strDatasetConfig,strNeRFConfig,strDataset,nImages,ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    cv::Mat ImgInstance;
 
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(strDataset+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        ImgInstance = cv::imread(strDataset+"/"+vstrInstanceFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strDataset << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }
        if(ImgInstance.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << strDataset << "/" << vstrInstanceFilenames[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,ImgInstance,tframe,strDataset);
        
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6); 
        
    }

    usleep(1*1e6);
    SLAM.FinishNeRFs();
    cout<<endl<<"Press Enter to save NeRFs (render images and obj.ply) or Crtl+C to quit ..." <<endl;
    getchar();

    SLAM.RenderNeRFsTest("./output");
    // Stop all threads
    SLAM.Shutdown();
    
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("FrameTrajectory.txt");
    SLAM.SaveObjects("./output/objects.txt");
    
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,vector<string> &vstrInstanceFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sName;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sName;
            string rgb = "rgb/" + sName;
            string instance = "instance/" +  sName;
            vstrImageFilenames.push_back(rgb);
            vstrInstanceFilenames.push_back(instance);
        }
    }
    f.close();       
}
