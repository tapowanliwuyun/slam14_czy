#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <chrono>

#include "rotation.h"

using namespace std;
using namespace cv;

void find_feature_matches(const Mat& img_1,const Mat& img_2,
                          vector<KeyPoint>& keypoints_1,
                          vector<KeyPoint>& keypoints_2,
                          vector<DMatch>& matches);

//像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d& p,const Mat& K);

void pose_estimation_3d3d(const vector<Point3f>& pts1,
                          const vector<Point3f>& pts2,
                          Mat& r,Mat& t_inv);

void bundleAdjustment(const vector<Point3f>& points_3f,
                      const vector<Point3f>& points_2f,
                      Mat& R, Mat& t_inv);//test 声明的行参和定义的不同是否可行:可以的!

struct cost_function_define
{
    cost_function_define(Point3f p1,Point3f p2):_p1(p1),_p2(p2){}
    template<typename T>
    bool operator()(const T* const cere_r,const T* const cere_t,T* residual) const
    {
        T p_1[3];
        T p_2[3];
        p_1[0]=T(_p1.x);
        p_1[1]=T(_p1.y);
        p_1[2]=T(_p1.z);
        AngleAxisRotatePoint(cere_r,p_1,p_2);
        p_2[0]=p_2[0]+cere_t[0];
        p_2[1]=p_2[1]+cere_t[1];
        p_2[2]=p_2[2]+cere_t[2];
        const T x=p_2[0]/p_2[2];
        const T y=p_2[1]/p_2[2];
        const T u=x*520.9+325.1;
        const T v=y*521.0+249.7;
        T p_3[3];
        p_3[0]=T(_p2.x);
        p_3[1]=T(_p2.y);
        p_3[2]=T(_p2.z);

        const T x1=p_3[0]/p_3[2];
        const T y1=p_3[1]/p_3[2];

        const T u1=x1*520.9+325.1;
        const T v1=y1*521.0+249.7;

        residual[0]=u-u1;
        residual[1]=v-v1;
        return true;
    }
    Point3f _p1,_p2;
};

int main(int argc,char** argv)
{
    cv::Mat img_1,img_2,depth_1,depth_2;
    if(argc!=5)
    {
        //读取图片
        img_1 = cv::imread("../src/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread("../src/2.png",1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        depth_1 = cv::imread("../src/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        depth_2 = cv::imread("../src/2_depth.png", -1);
    }
    else
    {
        //读取图片
        img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread(argv[2],1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        depth_1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        depth_2 = cv::imread(argv[4], -1);
    }
    assert(img_1.data != nullptr && img_2.data != nullptr);//若读取的图片没有内容，就终止程序

    vector<KeyPoint> keypoints_1,keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    cout<<"一共找到了"<<matches.size()<<"组匹配点"<<endl;

    Mat K=(Mat_<double>(3,3)<<520.9,0,325.1,
            0,521.0,249.7,
            0,0,1);
    vector<Point3f> pts1,pts2;
    for(DMatch m:matches)
    {
        ushort d1=depth_1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2=depth_2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if(d1==0 || d2==0)//bad depth
            continue;
        Point2d p1=pixel2cam(keypoints_1[m.queryIdx].pt,K);
        Point2d p2=pixel2cam(keypoints_2[m.trainIdx].pt,K);
        float dd1=float(d1)/5000.0;
        float dd2=float(d2)/5000.0;
        pts1.push_back(Point3f(p1.x*dd1,p1.y*dd1,dd1));
        pts2.push_back(Point3f(p2.x*dd2,p2.y*dd2,dd2));
    }

    cout<<"3d-3d pairs: "<<pts1.size()<<endl;
    Mat R,t;
    pose_estimation_3d3d(pts1,pts2,R,t);
    cout<<"ICP via SVD results: "<<endl;
    cout<<"R ="<<endl<<R<<endl;
    cout<<"t ="<<endl<<t<<endl;
    Mat R_inv=R.t();
    Mat t_inv=-R.t()*t;
    cout<<"R_inv ="<<endl<<R_inv<<endl;//R^(-1)
    cout<<"t_inv ="<<endl<<t_inv<<endl;

    Mat r;
    Rodrigues(R_inv,r);//R_inv->r
    cout<<"r= "<<endl<<r<<endl;

    for(int i=0;i<5;++i)
    {
        cout<<"p1= "<<pts1[i]<<endl;//??
        cout<<"p2= "<<pts2[i]<<endl;//??
        cout<<"(R*p2+t) = "<<
            R*(Mat_<double>(3,1)<<pts2[i].x,pts2[i].y,pts2[i].z)+t
            <<endl;cout<<endl;
    }

    cout<<"calling bundle adjustment"<<endl;
    bundleAdjustment(pts1,pts2,r,t_inv);

    return 0;
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
void pose_estimation_3d3d(const vector<cv::Point3f> &pts1,
                          const vector<cv::Point3f> &pts2,
                          cv::Mat &R, cv::Mat &t)
{
    int N = pts1.size();//匹配的3d点个数
    cv::Point3f p1,p2;//质心
    for (int i = 0; i < N; i++)
    {
        p1+=pts1[i];
        p2+=pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1)/N);//得到质心
    p2 = cv::Point3f(cv::Vec3f(p2)/N);
    vector<cv::Point3f> q1(N), q2(N);

    for(int i = 0; i < N; i++)
    {
        //去质心
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    //计算 W+=q1*q2^T(求和)
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();//初始化
    for(int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x,q1[i].y,q1[i].z) * Eigen::Vector3d(q2[i].x,q2[i].y,q2[i].z).transpose();
    }
    cout << "W = " << endl << W << endl;

    //利用svd分解 W=U*sigema*V
    //Eigen::ComputeFullU ： 在JacobiSVD中使用，表示要计算方阵U
    //Eigen::ComputeFullV ： 在JacobiSVD中使用，表示要计算方阵V
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();//得到U矩阵
    Eigen::Matrix3d V = svd.matrixV();//得到V矩阵
    cout << "U=" << U << endl;
    cout << "V=" << V << endl;
    Eigen::Matrix3d R_ = U * (V.transpose());
    if(R_.determinant() < 0)//若旋转矩阵R_的行列式<0 则取负号
    {
        R_ = -R_;
    }
    //得到平移向量
    Eigen::Vector3d t_ = Eigen::Vector3d (p1.x,p1.y,p1.z) - R_ * Eigen::Vector3d(p2.x,p2.y,p2.z);
    //把 Eigen形式的 r 和 t_ 转换为CV 中的Mat格式
    R = (cv::Mat_<double>(3,3) <<
                               R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

}


void bundleAdjustment(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &r, Mat &t_inv)
{

    double cere_rot[3], cere_tranf[3];
    //关于初值的选取有疑问，随便选择一个初值时，结果不正确，why??
    cere_rot[0]=r.at<double>(0,0);
    cere_rot[1]=r.at<double>(1,0);
    cere_rot[2]=r.at<double>(2,0);

    cere_tranf[0]=t_inv.at<double>(0,0);
    cere_tranf[1]=t_inv.at<double>(1,0);
    cere_tranf[2]=t_inv.at<double>(2,0);


    ceres::Problem problem;
    for(int i=0;i<pts1.size();++i)
    {
        ceres::CostFunction* costfunction=new ceres::AutoDiffCostFunction<cost_function_define,2,3,3>(new cost_function_define(pts1[i],pts2[i]));
        problem.AddResidualBlock(costfunction,NULL,cere_rot,cere_tranf);
    }

    ceres::Solver::Options option;
    option.linear_solver_type=ceres::DENSE_SCHUR;
    option.minimizer_progress_to_stdout=true;
    ceres::Solver::Summary summary;
    ceres::Solve(option,&problem,&summary);
    cout<<summary.BriefReport()<<endl;

    //std::cout << summary.FullReport() << "\n";

    cout<<"optional after: "<<endl;
    Mat cam_3d=(Mat_<double>(3,1)<<cere_rot[0],cere_rot[1],cere_rot[2]);
    //    cout<<"cam_3d : "<<endl<<cam_3d<<endl;
    Mat cam_9d;
    Rodrigues(cam_3d,cam_9d);

    cout<<"cam_9d: "<<endl<<cam_9d<<endl;
    cout<<"cam_t: "<<endl<<cere_tranf[0]<<" "<<cere_tranf[1]<<" "<<cere_tranf[2]<<endl;
    Mat tranf_3d=(Mat_<double>(3,1)<<cere_tranf[0],cere_tranf[1],cere_tranf[2]);

    for(int i=0;i<5;++i)
    {
        cout<<"p1= "<<pts1[i]<<endl;
        cout<<"p2= "<<pts2[i]<<endl;
        cout<<"(R*p1+t)= "<<
            cam_9d*(Mat_<double>(3,1)<<pts1[i].x,pts1[i].y,pts1[i].z)+tranf_3d<<endl;
        cout<<endl;
    }
}
