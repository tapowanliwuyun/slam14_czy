#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <Eigen/SVD>
#include <chrono>

#include "rotation.h"


using namespace std;
using namespace cv;


//第一步：定义Cost Function函数
struct cost_function
{
    cost_function(Point3f p_3D,Point2f p_p):_p_3D(p_3D),_p_p(p_p) {}//3D-2D：知道n个3D空间点以及其投影位置，然后估计相机位姿
    //计算残差
    template <typename T>//模板：使得在定义时可以模糊类型
    bool operator() (
            const T* const r, const T* const t,//r,t为待优化的参数
            const T* K,//k为待优化的参数
            T* residual ) const //殘差
    {
        T p_3d[3];
        T p_Cam[3];//相机坐标系下空间点的坐标
        p_3d[0]=T(_p_3D.x);
        p_3d[1]=T(_p_3D.y);
        p_3d[2]=T(_p_3D.z);
//         通过tool文件夹中的rotation.h中的AngleAxisRotatePoint()函数
//         计算在相机仅旋转的情况下，新坐标系下的坐标
//         也就是p_Cam=R*p_3d
        //cout<<"point_3d: "<<p_3d[0]<<" "<<p_3d[1]<<"  "<<p_3d[2]<<endl;
        AngleAxisRotatePoint(r,p_3d,p_Cam);

        p_Cam[0]=p_Cam[0]+t[0];
        p_Cam[1]=p_Cam[1]+t[1];
        p_Cam[2]=p_Cam[2]+t[2];
        //R,t计算T
        //Eigen::Isometry3d T_w_c;
//        T_w_c.rotate(r);

        const T x=p_Cam[0]/p_Cam[2];
        const T y=p_Cam[1]/p_Cam[2];
        //3D点投影后的像素坐标
//         const T u=x*520.9+325.1;
//         const T v=y*521.0+249.7;
        const T u=x*K[0]+K[1];
        const T v=y*K[2]+K[3];

        //观测到的投影位置的像素坐标
        T u1=T(_p_p.x);
        T v1=T(_p_p.y);

        //残差
        residual[0]=u-u1;
        residual[1]=v-v1;
        return true;
    }
    Point3f _p_3D;
    Point2f _p_p;
};

void find_feature_matches (
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment(const vector<Point3f> points_3d,
                      const vector<Point2f> points_2d, Mat K, Mat &r, Mat &t);

int main ( int argc, char** argv )
{
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

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/1000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
//    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false,cv::SOLVEPNP_EPNP ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    // solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false,CV_ITERATIVE );
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"optional before: "<<endl;
    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl<<endl;

    cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment(pts_3d,pts_2d,K,r,t);
}
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
            (
                    ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
                    ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
            );
}

//构建最小二乘问题
void bundleAdjustment(const vector<Point3f> points_3d,
                      const vector<Point2f> points_2d,Mat K, Mat &r, Mat &t)
{
//    cout<<"R = "<<endl<<R<<endl;
    cout<<"start:"<<endl;
    double rotation_vector[3],tranf[3];//旋转向量r，平移t
    double k[4];
    rotation_vector[0]=r.at<double>(0,0);
    rotation_vector[1]=r.at<double>(0,1);
    rotation_vector[2]=r.at<double>(0,2);

    tranf[0]=t.at<double>(0,0);
    tranf[1]=t.at<double>(1,0);
    tranf[2]=t.at<double>(2,0);

    k[0]=520.9;
    k[1]=325.1;
    k[2]=521.0;
    k[3]=249.7;

    ceres::Problem problem;
    for(int i=0;i<points_3d.size();++i)
    {
        ceres::CostFunction* costfunction=new ceres::AutoDiffCostFunction<cost_function,2,3,3,4>(new cost_function(points_3d[i],points_2d[i]));
        problem.AddResidualBlock(costfunction,NULL,rotation_vector,tranf,k);
    }
//    cout<<rotation_vector[0]<<" "<<rotation_vector[1]<<" "<<rotation_vector[2]<<endl;
    //配置求解器
    ceres::Solver::Options option;
    option.linear_solver_type=ceres::DENSE_QR;//DENSE_SCHUR
    //true:迭代信息输出到屏幕.false:不输出
    option.minimizer_progress_to_stdout=true;

    ceres::Solver::Summary summary;//优化信息
    //可以和g2o优化做对比
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    //开始优化
    ceres::Solve(option,&problem,&summary);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"solve time cost = "<<time_used.count()<<" second."<<endl;

    //输出结果
    cout<<summary.BriefReport()<<endl;
    Mat Rotaion_vector=(Mat_<double>(3,1)<<rotation_vector[0],rotation_vector[1],rotation_vector[2]);
//    cout<<rotation_vector[0]<<" "<<rotation_vector[1]<<""<<rotation_vector[2]<<endl;//0,1,2
    Mat Rotation_matrix;
    Rodrigues(Rotaion_vector,Rotation_matrix);//r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"after optional:"<<endl;
    cout<<"R = "<<endl<<Rotation_matrix<<endl;
//    cout<<"R = "<<endl<<<<endl;
    cout<<"t = "<<tranf[0]<<" "<<tranf[1]<<" "<<tranf[2]<<endl;

    cout << "k = " << k[0] << " " << k[1] << " " << k[2] << " " << k[3] << endl;
}
