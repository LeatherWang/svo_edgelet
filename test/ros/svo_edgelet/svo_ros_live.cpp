#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <svo/slamviewer.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <svo/slamviewer.h>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;

class SvoRosShell
{
public:
    SvoRosShell(string paramFileStr);
    ~SvoRosShell();
    void spin();

private:
    ros::NodeHandle n;
    const static int rate = 200;

    svo::AbstractCamera* cam_;
    svo::PinholeCamera* cam_pinhole_;
    svo::FrameHandlerMono* vo_;

    SLAM_VIEWER::Viewer* viewer_;
    std::thread * viewer_thread_;

    cv::Mat mapx, mapy;
    bool readIntrinsic(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef, cv::Size &imageSize);

    ros::Subscriber subImageMsg;
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
};

SvoRosShell::SvoRosShell(string paramFileStr)
{
    string file_path = paramFileStr+"/camera.yaml"; // camera_fisheye_ankobot
    cout<<"intrinsic file: "<<file_path<<endl;
    cv::Mat K_Mat = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat DistCoef = cv::Mat(4, 1, CV_64F);
    cv::Size imageSize;
    if(!readIntrinsic(file_path, K_Mat, DistCoef, imageSize))
        return;

    cv::Mat R = cv::Mat::eye(3,3, CV_32F);
    cv::initUndistortRectifyMap(K_Mat, DistCoef, R, K_Mat,
                                         imageSize, CV_32FC1, mapx, mapy);
//    cv::fisheye::initUndistortRectifyMap(K_Mat, DistCoef, R, K_Mat,
//                                         imageSize, CV_32FC1, mapx, mapy);

    cam_ = new svo::PinholeCamera(imageSize.width, imageSize.height, K_Mat.at<double>(0,0),
                                  K_Mat.at<double>(1,1), K_Mat.at<double>(0,2), K_Mat.at<double>(1,2),
                                  0,0,0,0,0);

    vo_ = new svo::FrameHandlerMono(cam_);
    vo_->start();

    viewer_ = new SLAM_VIEWER::Viewer(vo_);
    viewer_thread_ = new std::thread(&SLAM_VIEWER::Viewer::run, viewer_);
    viewer_thread_->detach();

    subImageMsg = n.subscribe("/cam0/image_raw", 10, &SvoRosShell::GrabImage, this);
}

SvoRosShell::~SvoRosShell()
{
    delete vo_;
    delete cam_;
    delete cam_pinhole_;

    delete viewer_;
    delete viewer_thread_;
}

bool SvoRosShell::readIntrinsic(const string &file_path, cv::Mat &K_Mat, cv::Mat &DistCoef, cv::Size &imageSize)
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

void SvoRosShell::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imageUndistort;
    cv::remap(cv_ptr->image, imageUndistort, mapx, mapy, cv::INTER_LINEAR);

    if(imageUndistort.channels() == 3)
        cv::cvtColor(imageUndistort, imageUndistort, CV_BGR2GRAY);
    vo_->addImage(imageUndistort, ros::Time::now().toSec());

    if(vo_->lastFrame() != NULL)
    {

        std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                  << "#Features: " << vo_->lastNumObservations() << " \n";
    }
}

void SvoRosShell::spin()
{
    ros::Rate loop_rate(rate);

    while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "svo_ros");
    SvoRosShell mysvo("/home/leather/lxdata/leather_temp/calib/UI-1221E_2019-1-24");
    mysvo.spin();
    return 0;
}
