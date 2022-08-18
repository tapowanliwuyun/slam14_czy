
// Created by czy on 2022/4/7
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>

using namespace std;

void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2,
                          std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches);


// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);

int main(int argc, char ** argv) {
    cv::Mat img_1,img_2,d1,d2;
    if(argc!=5)
    {
        //读取图片
        img_1 = cv::imread("../src/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread("../src/2.png",1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        d1 = cv::imread("../src/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        d2 = cv::imread("../src/2_depth.png", -1);
    }
    else
    {
        //读取图片
        img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread(argv[2],1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        d1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        d2 = cv::imread(argv[4], -1);
    }
    assert(img_1.data != nullptr && img_2.data != nullptr);//若读取的图片没有内容，就终止程序

    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);//得到两个图片的特征匹配点
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point3f> pts_3d;//创建容器pts_3d存放3d点（图1对应的特征点的相机坐标下的3d点）
    vector<cv::Point2f> pts_2d;//创建容器pts_2d存放图2的特征点

    for (cv::DMatch m: matches) {
        //把对应的图1的特征点的深度信息拿出来
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0;//用dd存放换算过尺度的深度信息
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//p1里面放的是图1特征点在相机坐标下的归一化坐标（只包含 x,y）
        pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));//得到图1特征点在相机坐标下的3d坐标
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);//得到图2特张点的像素坐标
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;//3d-2d配对个数得用pts_3d的size


    cout << "使用cv_PnP求解 位姿" << endl;

    cv::Mat r, t;
    cv::Mat R;
    chrono::steady_clock::time_point t1,t2;
    chrono::duration<double> time_used;

    cout << "***********************************SOLVEPNP_ITERATIVE***********************************" << endl;
    t1 = chrono::steady_clock::now();
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false,cv::SOLVEPNP_ITERATIVE); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "calling bundle adjustment" << endl;
    cout << "***********************************SOLVEPNP_ITERATIVE***********************************" << endl;

    cout << "***********************************SOLVEPNP_EPNP***********************************" << endl;
    t1 = chrono::steady_clock::now();
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false,cv::SOLVEPNP_EPNP); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "calling bundle adjustment" << endl;
    cout << "***********************************SOLVEPNP_EPNP***********************************" << endl;

    cout << "***********************************SOLVEPNP_UPNP***********************************" << endl;
    t1 = chrono::steady_clock::now();
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false,cv::SOLVEPNP_UPNP); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "calling bundle adjustment" << endl;
    cout << "***********************************SOLVEPNP_UPNP***********************************" << endl;

    cout << "***********************************SOLVEPNP_DLS***********************************" << endl;
    t1 = chrono::steady_clock::now();
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false,cv::SOLVEPNP_DLS); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "calling bundle adjustment" << endl;
    cout << "***********************************SOLVEPNP_DLS***********************************" << endl;

    cout << "***********************************SOLVEPNP_P3P***********************************" << endl;

    vector<cv::Point3f> pts_p3p_3d;//创建容器pts_3d存放3d点（图1对应的特征点的相机坐标下的3d点）
    vector<cv::Point2f> pts_p3p_2d;//创建容器pts_2d存放图2的特征点

    //取出其中的4个点对
    for (int i = 0; i < 4; i++)
    {
        pts_p3p_3d.push_back(pts_3d[i]);
        pts_p3p_2d.push_back(pts_2d[i]);
    }

/*    cv::Mat pts_p3p = (cv::Mat_<double>(4,3) <<
            pts_p3p_3d[0].x,pts_p3p_3d[0].y,pts_p3p_3d[0].z,
            pts_p3p_3d[1].x,pts_p3p_3d[1].y,pts_p3p_3d[1].z,
            pts_p3p_3d[2].x,pts_p3p_3d[2].y,pts_p3p_3d[2].z,
            pts_p3p_3d[3].x,pts_p3p_3d[3].y,pts_p3p_3d[3].z
        );*/

    t1 = chrono::steady_clock::now();
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP(pts_p3p_3d, pts_p3p_2d, K, cv::Mat(), r, t, false,cv::SOLVEPNP_P3P); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
    cout << "calling bundle adjustment" << endl;
    cout << "***********************************SOLVEPNP_P3P***********************************" << endl;
}



//实现特征匹配
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
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows;i++)
    {
        double dist = match[i].distance;
        if(dist > max_dist) max_dist = dist;
        if(dist < min_dist) min_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);;
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ))
        {
            matches.push_back ( match[i] );
        }
    }
}

//实现像素坐标到相机坐标的转换（求出来的只是包含相机坐标下的x,y的二维点）
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2d(
            ((p.x - K.at<double>(0,2))/K.at<double>(0,0)),
            ((p.y - K.at<double>(1,2))/K.at<double>(1,1))
    );
}

