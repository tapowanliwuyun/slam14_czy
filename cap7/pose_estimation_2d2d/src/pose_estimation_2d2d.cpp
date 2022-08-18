#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d//features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cstring>
// #include "extra.h" // use this if in OpenCV2

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

string first_file = "../src/1.png";
string second_file = "../src/2.png";

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2,
                          std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches);

void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches,
                          cv::Mat& R, cv::Mat& t);
// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam (const cv::Point2d& p, const cv::Mat& K);


int main(int argc, char ** argv)
{
    //-- 读取图像
    cout << "读取src文件下的图片" << endl;
    cv::Mat img_1 = cv::imread(first_file,CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(second_file, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    //-- 估计两张图像间运动
    cv::Mat R,t;
    pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);

    //-- 验证E=t^R*scale
    cv::Mat t_x = (cv::Mat_<double>(3,3) <<
                                         0,                       -t.at<double>(2,0),t.at<double>(1,0),
            t.at<double>(2,0), 0,                       -t.at<double>(0,0)
                                                        -t.at<double>(1,0),t.at<double>(0,0),0
    );

    cout << "t^R = " << endl << t_x * R << endl;

    //-- 验证对极约束      // 相机内参
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0,0,1);
    for(cv::DMatch m:matches)
    {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);
        cv::Mat y1 = (cv::Mat_<double>(3,1) << pt1.x,pt1.y,1);
        cv::Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
        cv::Mat y2 = ( cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        cv::Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2,
                          std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches)
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<cv::DMatch> match;
    matcher->match(descriptors_1,descriptors_2,match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f\n", max_dist);
    printf("-- Min dist : %f\n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(match[i].distance <= max(2*min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam (const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2d(   //at是内参数矩阵
            (p.x - K.at<double> (0,2)) / K.at<double>(0,0),
            (p.y - K.at<double> (1,2)) / K.at<double>(1,1)
    );
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches,
                          cv::Mat& R, cv::Mat& t)
{
    // 相机内参,TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for(int i = 0; i < (int) matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);//匹配点对中第一张图片上的点
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);//匹配点对中第二张图片上的点
    }
    //-- 计算基础矩阵
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2,CV_FM_8POINT);//计算给定一组对应点的基本矩阵 八点法
    cout << "fundamental_matrix is" << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    cv::Point2d principal_point (325.1, 249.7); 	//相机光心, TUM dataset标定值
    double focal_length = 521; 	//相机焦距, TUM dataset标定值
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1,points2,focal_length,principal_point);
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose(essential_matrix, points1, points2, R,t,
                    focal_length,principal_point);
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
}
