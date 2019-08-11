// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>

#include <svo/slamviewer.h>
#include<thread>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace svo {

bool readIntrinsic(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef, cv::Size &imageSize)
{
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    DistCoef.at<double>(0) = fs["Camera.k1"];
    DistCoef.at<double>(1) = fs["Camera.k2"];
    DistCoef.at<double>(2) = fs["Camera.p1"];
    DistCoef.at<double>(3) = fs["Camera.p2"];

    K_Mat.at<double>(0,0) = fs["Camera.fx"];
    K_Mat.at<double>(1,1) = fs["Camera.fy"];
    K_Mat.at<double>(0,2) = fs["Camera.cx"];
    K_Mat.at<double>(1,2) = fs["Camera.cy"];

    imageSize.height = fs["Camera.height"];
    imageSize.width = fs["Camera.width"];

    cout<<"________________________________________"<<endl;
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << K_Mat.at<double>(0,0) << endl;
    cout << "- fy: " << K_Mat.at<double>(1,1) << endl;
    cout << "- cx: " << K_Mat.at<double>(0,2) << endl;
    cout << "- cy: " << K_Mat.at<double>(1,2) << endl;
    cout << "- k1: " << DistCoef.at<double>(0) << endl;
    cout << "- k2: " << DistCoef.at<double>(1) << endl;
    cout << "- p1: " << DistCoef.at<double>(2) << endl;
    cout << "- p2: " << DistCoef.at<double>(3) << endl;
    cout<<"________________________________________"<<endl;
    return true;
}

void LoadImages(const std::string &strFile, vector<std::string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());
    // skip first three lines
    std::string s0;
    getline(f, s0);
    getline(f, s0);
    getline(f, s0);

    while (!f.eof())
    {
        std::string s;
        getline(f, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            std::string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
    f.close();
}

class BenchmarkNode
{
    svo::AbstractCamera* cam_;
    svo::PinholeCamera* cam_pinhole_;
    svo::FrameHandlerMono* vo_;

    SLAM_VIEWER::Viewer* viewer_;
    std::thread * viewer_thread_;
    cv::Mat K_Mat, DistCoef;
    cv::Size imageSize;

public:
    BenchmarkNode(string datasetFileStr);
    ~BenchmarkNode();
    void runFromFolder(string datasetFileStr);
};

BenchmarkNode::BenchmarkNode(string datasetFileStr)
{
    string file_path = datasetFileStr+"/camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    K_Mat = cv::Mat::eye(3, 3, CV_64F);
    DistCoef = cv::Mat(4, 1, CV_64F);
    if(!readIntrinsic(file_path, K_Mat, DistCoef, imageSize))
        return;

    cam_ = new svo::PinholeCamera(imageSize.width, imageSize.height, K_Mat.at<double>(0,0),
                                  K_Mat.at<double>(1,1), K_Mat.at<double>(0,2), K_Mat.at<double>(1,2),
                                  0,0,0,0,0);

    vo_ = new svo::FrameHandlerMono(cam_);
    vo_->start();

    viewer_ = new SLAM_VIEWER::Viewer(vo_);
    viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run,viewer_);
    viewer_thread_->detach();
}

BenchmarkNode::~BenchmarkNode()
{
    delete vo_;
    delete cam_;
    delete cam_pinhole_;

    delete viewer_;
    delete viewer_thread_;
}

void BenchmarkNode::runFromFolder(string datasetFileStr)
{
    cv::Mat mapx, mapy; //内部存的是重映射之后的坐标，不是像素值
    cv::Mat R = cv::Mat::eye(3,3, CV_32F);
    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, R, K_Mat,
                                         imageSize, CV_32FC1, mapx, mapy);


    string inputDir = datasetFileStr+"/cam0";
    std::string fileExtension = ".jpg";
    std::vector<std::string> imageFilenames;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
            continue;

        std::string filename = itr->path().filename().string();

        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
            continue;
        imageFilenames.push_back(itr->path().string());
    }
    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return;
    }

    auto cmp = [](const std::string &a, const std::string &b){
        if(a.size() < b.size())
            return true;
        else if(a.size() == b.size())
            return a<b;
        return false;
    };
    sort(imageFilenames.begin(), imageFilenames.end(), cmp);

    int nImages = imageFilenames.size();
    for(int i=323; i<nImages; i++)
    {
        cv::Mat imageRaw = cv::imread(imageFilenames[i], cv::IMREAD_GRAYSCALE);
        assert(!imageRaw.empty());

        double imageTimestamp = std::stod(imageFilenames[i].substr(imageFilenames[i].size()-4-16, 10) + "." +
                        imageFilenames[i].substr(imageFilenames[i].size()-4-6, 6));

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imageRaw, imageRaw);

        cout<<imageFilenames[i]<<endl;
        cv::Mat imageUndistort;
        cv::remap(imageRaw, imageUndistort, mapx, mapy, cv::INTER_LINEAR);

        vo_->addImage(imageUndistort, imageTimestamp);
        cv::waitKey(50);

        if(vo_->lastFrame() != NULL)
        {

            std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                      << "#Features: " << vo_->lastNumObservations() << " \n";

        }
    }

}

} // namespace svo

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << std::endl << "Usage: ./mono_undistort_images path_to_log " << std::endl;
        return 0;
    }
    svo::BenchmarkNode benchmark(argv[1]);
    benchmark.runFromFolder(argv[1]);

    return 0;
}

