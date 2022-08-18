//
// Created by czy on 2022/4/6.
//
#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <sophus/se3.hpp>
#include <chrono>

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint>& keypoints_1,
                          std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch> &matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void pose_estimation_3d3d(const vector<cv::Point3f> &pts1,
                          const vector<cv::Point3f> &pts2,
                          cv::Mat &R, cv::Mat &t);

void bundleAdjustment(const vector<cv::Point3f> &pts1,
                      const vector<cv::Point3f> &pts2,
                      cv::Mat &R, cv::Mat &t);

int main(int argc, char** argv)
{
    cv::Mat img_1,img_2,depth1,depth2;
    if(argc!=5)
    {
        //读取图片
        img_1 = cv::imread("../src/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread("../src/2.png",1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        depth1 = cv::imread("../src/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        depth2 = cv::imread("../src/2_depth.png", -1);
    }
    else
    {
        //读取图片
        img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread(argv[2],1);
        //接下来的是建立3d点 利用深度图可以获取深度信息
        //depth1是图1对应的深度图 depth2是图2对应的深度图
        depth1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);// 深度图为16位无符号数，单通道图像
        depth2 = cv::imread(argv[4], -1);
    }

    vector<cv::KeyPoint> keypoints_1, keypoints_2;//容器keypoints_1, keypoints_2分别存放图1和图2的特征点
    vector<cv::DMatch> matches;
    //得到图1与图2的特征匹配点
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //内参矩阵
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3f> pts1,pts2;

    for(cv::DMatch m:matches)
    {
        //先把两图特征匹配点对应的深度拿出来
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_1[m.trainIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if(d1 == 0 || d2 == 0)//深度无效
            continue;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);//得到图1的特征匹配点在其相机坐标下的x ,y
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt,K);//得到图2的特征匹配点在其相机坐标下的x ,y
        //对深度进行尺度变化得到真正的深度
        float dd1 = float(d1)/5000.0;
        float dd2 = float(d2)/5000.0;
        //容器 pts_1与pts_2分别存放 图1中的特征匹配点其相机坐标下的3d点 和 图2中的特征匹配点其相机坐标下的3d点
        pts1.push_back(cv::Point3f(p1.x*dd1, p1.y*dd1,dd1));
        pts2.push_back(cv::Point3f(p2.x*dd2, p2.y*dd2,dd2));
    }
    //这样就可以得到 3d-3d的匹配点
    cout << "3d-3d pairs: " << pts1.size() << endl;
    cv::Mat R,t;
    cout<<" ************************************ICP 3d-3d by SVD**************************************** "<<endl;
    pose_estimation_3d3d(pts1,pts2,R,t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R^T = " << R.t() << endl;
    cout << "t^T = " << -R.t() * t << endl;
    cout<<" ************************************ICP 3d-3d by SVD**************************************** "<<endl;
    cout<<endl;
    cout<<" ************************************ICP 3d-3d by g2o**************************************** "<<endl;
    bundleAdjustment(pts1, pts2, R, t);
    cout<<"R= \n"<<R<<endl;
    cout<<"t = "<< t.t() <<endl;
    cout<<"验证 p2 = R*P1 +t "<<endl;
    for (int i = 0; i < 5; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R*p1+t) = " <<
            R * (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, pts1[i].z) + t
            << endl;
        cout << endl;
    }
    cout<<" ************************************ICP 3d-3d by g2o**************************************** "<<endl;


    return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
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
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

//实现像素坐标到相机坐标的转换（求出来的只是包含相机坐标下的x,y的二维点）
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
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
        p2+=pts2[2];
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

//对于用g2o来进行优化的话，首先要定义顶点和边的模板
//顶点，也就是咱们要优化的pose 用李代数表示它 6维
class Vertexpose : public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//必须写，我也不知道为什么
    //重载setToOriginImpl函数 这个应该就是把刚开的待优化的pose放进去
    virtual void setToOriginImpl() override
    {
        _estimate = Sophus::SE3d();
    }
    //重载oplusImpl函数，用来更新pose（待优化的系数）
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double, 6,1> update_eigen;//更新的量，就是增量呗，dx
        update_eigen <<  update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;//更新pose 李代数要转换为李群，这样才可以左乘
    }
    //存盘 读盘 ：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};
//定义边
class EdgeProjectXYZRGBD : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, Vertexpose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBD(const Eigen::Vector3d &point) : _point(point) {}//赋值这个是图1坐标下的3d点
    //计算误差
    virtual void computeError() override
    {
        const Vertexpose *v = static_cast<const Vertexpose*>(_vertices[0]);//顶点v
        _error = _measurement - v->estimate() * _point;
    }
    //计算雅克比
    virtual void linearizeOplus() override
    {
        const Vertexpose *v = static_cast<const Vertexpose *>(_vertices[0]);//顶点v
        Sophus::SE3d T = v->estimate();//把顶点的待优化系数拿出来
        Eigen::Vector3d xyz_trans = T * _point;//变换到图2下的坐标点
        //课本2  p86 and p199
        _jacobianOplusXi.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3,3>(0,3) = Sophus::SO3d::hat(xyz_trans);
    }

    //存盘 读盘 ：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}

protected:
    Eigen::Vector3d _point;
};

//利用g2o
void bundleAdjustment(const vector<cv::Point3f> &pts1,
                      const vector<cv::Point3f> &pts2,
                      cv::Mat &R, cv::Mat &t)
{
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;// pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block (unique_ptr<Block::LinearSolverType>(linearSolver));  // 矩阵块求解器
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(unique_ptr<Block>(solver_ptr));

    g2o::SparseOptimizer optimizer;// 图模型
    optimizer.setAlgorithm(solver);// 设置求解器
    optimizer.setVerbose(false);       // 调试输出

    //加入顶点
    Vertexpose *v = new Vertexpose();
    v->setEstimate(Sophus::SE3d());
    v->setId(0);
    optimizer.addVertex(v);
    //加入边
    int index = 1;
    for (size_t i = 0; i < pts1.size();i++)
    {
        EdgeProjectXYZRGBD *edge = new EdgeProjectXYZRGBD(Eigen::Vector3d(pts1[i].x,pts1[i].y,pts1[i].z));
        edge->setId(index);//边的编号
        edge->setVertex(0,v);//设置顶点  顶点编号
        edge->setMeasurement(Eigen::Vector3d(pts2[i].x,pts2[i].y,pts2[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());//set信息矩阵为单位矩阵 信息矩阵是3维方阵，因为误差是3维
        optimizer.addEdge(edge);//加入边
        index++;//边的编号++
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();//开始  初始化
    optimizer.optimize(10);//迭代次数

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;


    cout << endl << "after optimization:" << endl;
    cout << "T=\n" << v->estimate().matrix() << endl;

    // 把位姿转换为Mat类型
    Eigen::Matrix3d R_ = v->estimate().rotationMatrix();
    Eigen::Vector3d t_ = v->estimate().translation();

    R = (cv::Mat_<double>(3,3) << R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2));
    t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

}

