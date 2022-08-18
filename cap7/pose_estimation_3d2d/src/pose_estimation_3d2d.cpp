// Created by nnz on 2022/4/5
#include <iostream>
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
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <sophus/se3.hpp>
#include <chrono>
//该程序用了三种方法实现位姿估计
//第一种，调用cv的函数pnp求解 R ,t
//第二种，手写高斯牛顿进行位姿优化
//第三种，利用g2o进行位姿优化
using namespace cv;
using namespace std;

//VecVector2d可以定义存放二维向量的容器
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
//VecVector3d可以定义存放三维向量的容器
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2,
                          std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches);

// 像素坐标转相机归一化坐标
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);

// 利用g2o优化位姿和特征点
void bundleAdjustment(const vector<cv::Point3f> points_3d,
                      const vector<cv::Point2f> points_2d,
                      const cv::Mat& K,
                      cv::Mat& R, cv::Mat& t);

// BA by gauss-newton 手写高斯牛顿进行位姿优化
void bundleAdjustmentGaussNewton(const VecVector3d& points_3d,
                                 const VecVector2d& points_2d,
                                 const cv::Mat & K,
                                 Sophus::SE3d& pose);


//对于用g2o来进行优化的话，首先要定义顶点和边的模板
//顶点，也就是咱们要优化的pose 用李代数表示它 6维
class Vertexpose: public g2o::BaseVertex<6,Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW//必须写，我也不知道为什么
    //重载setToOriginImpl函数 这个应该就是把刚开的待优化的pose放进去
    virtual void setToOriginImpl() override
    {
        _estimate = Sophus::SE3d();
    }
    //重载oplusImpl函数，用来更新pose（待优化的系数）
    virtual void oplusImpl(const double *update) override
    {
        Eigen::Matrix<double,6,1> update_eigen;//更新的量，就是增量呗，dx
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate=Sophus::SE3d::exp(update_eigen)* _estimate;//更新pose 李代数要转换为李群，这样才可以左乘
    }
    //存盘 读盘 ：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};
//定义边模板 边也就是误差，二维 并且把顶点也放进去
class EdgeProjection : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,Vertexpose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//必须写，我也不知道为什么
    //有参构造，初始化 图1中的3d点 以及相机内参K
    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos),_K(K) {}
    //计算误差
    virtual void computeError() override
    {
        const Vertexpose *v=static_cast<const Vertexpose *>(_vertices[0]);
        Sophus::SE3d T=v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);//T * _pos3d是图2的相机坐标下的3d点
        pos_pixel /= pos_pixel[2];//得到了像素坐标的齐次形式
        _error = _measurement - pos_pixel.head<2>();
    }
    //计算雅克比矩阵
    virtual void linearizeOplus() override
    {
        const Vertexpose *v = static_cast<Vertexpose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam=T*_pos3d;//图2的相机坐标下的3d点
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        //雅克比矩阵见 书 p187 公式7.46
        _jacobianOplusXi
                << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
                0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;


    }
    //存盘 读盘 ：留空
    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};


//利用g2o优化pose
void bundleAdjustmentG2O(const VecVector3d& points_3d,
                         const VecVector2d& points_2d,
                         const cv::Mat & K,
                         Sophus::SE3d &pose);


int main(int argc, char ** argv)
{
    cv::Mat img_1,img_2,d1;
    //读取图片
    if(argc != 5)
    {

        img_1 = cv::imread("../src/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread("../src/2.png", CV_LOAD_IMAGE_COLOR);

        d1 = cv::imread("../src/1_depth.png", -1);// 深度图为16位无符号数，单通道图像

        //cout << "usage : pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        //return 1;
    }
    else
    {
        img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);//读取彩色图
        img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

        d1 = cv::imread(argv[3], -1);// 深度图为16位无符号数，单通道图像
    }

    //-- 读取图像
    //img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);//读取彩色图
    //img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);//若读取的图片没有内容，就终止程序

    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches (img_1, img_2, keypoints_1,keypoints_2, matches);//得到两个图片的特征匹配点
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点，把深度图信息读进来，构造三维点
    //d1 = cv::imread(argv[3], -1);// 深度图为16位无符号数，单通道图像
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    vector<cv::Point3f> pts_3d;//创建容器pts_3d存放3d点（图1对应的特征点的相机坐标下的3d点）
    vector<cv::Point2f> pts_2d;//创建容器pts_2d存放图2的特征点

    for(cv::DMatch m : matches)
    {
        //把对应的图1的特征点的深度信息拿出来
        ushort d = d1.ptr<unsigned short> (int (keypoints_1[m.queryIdx].pt.y))[int (keypoints_1[m.queryIdx].pt.x)];
        if(d == 0) // bad depth
            continue;
        float dd = d/5000.0;//用dd存放换算过尺度的深度信息
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//p1里面放的是图1特征点在相机坐标下的归一化坐标（只包含 x,y）
        pts_3d.push_back(cv::Point3f (p1.x *dd, p1.y*dd,dd));//得到图1特征点在相机坐标下的3d坐标
        pts_2d.push_back( keypoints_2[m.trainIdx].pt );//得到图2特张点的像素坐标
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;//3d-2d配对个数得用pts_3d的size

    cout<<"使用cv求解 位姿"<<endl;
    cout<<"***********************************opencv***********************************"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat r,t;
    //Mat()这个参数指的是畸变系数向量
    cv::solvePnP( pts_3d, pts_2d, K, cv::Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Mat R;
    cv::Rodrigues(r, R);//r为旋转向量形式，利用cv的Rodrigues()函数将旋转向量转换为旋转矩阵

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;
    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;
    bundleAdjustment(pts_3d, pts_2d, K, R, t);

    cout<<"***********************************opencv***********************************"<<endl;
    cout<<"手写高斯牛顿优化位姿"<<endl;
    cout<<"***********************************手写高斯牛顿***********************************"<<endl;
    VecVector3d pts_3d_eigen; //存放3d点（图1对应的特征点的相机坐标下的3d点）
    VecVector2d pts_2d_eigen; //存放图2的特征点
    for (size_t i =0; i < pts_3d.size(); i++)//size_t
    {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y,pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    Sophus::SE3d pose_gn;//位姿（李群）
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen,K,pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;
    cout << "R = \n" << pose_gn.rotationMatrix() << endl;
    cout << "t = " << pose_gn.translation().transpose() << endl;
    cout<<"***********************************手写高斯牛顿***********************************"<<endl;


    cout<<"g2o优化位姿"<<endl;
    cout << "***********************************g2o***********************************" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen,pts_2d_eigen,K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;
    cout << "***********************************g2o***********************************" << endl;

    return 0;
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


//手写高斯牛顿
// BA by gauss-newton 手写高斯牛顿进行位姿优化
void bundleAdjustmentGaussNewton(const VecVector3d& points_3d,
                                 const VecVector2d& points_2d,
                                 const cv::Mat & K,
                                 Sophus::SE3d& pose)
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iters = 10;//迭代次数
    double cost = 0, lastcost = 0;//代价函数（目标函数）
    //拿出内参
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    //进入迭代
    for (int iter = 0; iter < iters; iter++)
    {
        Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();//初始化H矩阵
        Vector6d b = Vector6d::Zero();//对b矩阵初始化

        cost = 0;
        // 遍历所有的特征点  计算cost
        for(int i=0;i<points_3d.size();i++)
        {
            Eigen::Vector3d pc=pose*points_3d[i];//利用待优化的pose得到图2的相机坐标下的3d点
            double inv_z=1.0/pc[2];//得到图2的相机坐标下的3d点的z的倒数，也就是1/z
            double inv_z2 = inv_z * inv_z;//(1/z)^2
            //定义投影
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);
            //定义误差
            Eigen::Vector2d e=points_2d[i]-proj;
            cost += e.squaredNorm();//cost=e*e
            //定义雅克比矩阵J
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        //出了这个内循环，表述结束一次迭代的计算，接下来，要求pose了
        Vector6d dx;//P129页 公式6.33 计算增量方程 Hdx=b
        dx = H.ldlt().solve(b);//算出增量dx
        //判断dx这个数是否有效
        if(isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }
        //如果我们进行了迭代，且最后的cost>=lastcost的话，那就表明满足要求了，可以停止迭代了
        if(iter > 0 && cost >= lastcost)
        {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastcost << endl;
            break;
        }
        //优化pose 也就是用dx更新pose
        pose = Sophus::SE3d::exp(dx) * pose; //dx是李代数，要转换为李群
        lastcost=cost;
        cout << "iteration " << iter << " cost="<< std::setprecision(12) << cost << endl;
        //std::setprecision(12)浮点数控制位数为12位
        //如果误差特别小了，也结束迭代

        if (dx.norm() < 1e-6)
        {
            // converge
            break;
        }
    }
    cout<<"pose by g-n \n"<<pose.matrix()<<endl;

}


//利用g2o优化pose
void bundleAdjustmentG2O(const VecVector3d& points_3d,
                         const VecVector2d& points_2d,
                         const cv::Mat & K,
                         Sophus::SE3d &pose)
{
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;// pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block (unique_ptr<Block::LinearSolverType>(linearSolver));     // 矩阵块求解器
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));

    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器 算法g-n
    optimizer.setVerbose(false);       // 取消调试输出
    //加入顶点
    Vertexpose *v=new Vertexpose();
    v->setEstimate(Sophus::SE3d());
    v->setId(0);
    optimizer.addVertex(v);
    //K
    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    //加入边
    int index=1;
    for(size_t i=0;i<points_2d.size();++i)
    {
        //遍历 把3d点和像素点拿出来
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);//有参构造
        edge->setId(index);
        edge->setVertex(0,v);
        edge->setMeasurement(p2d);//设置观测值，其实就是图2 里的匹配特征点的像素位置
        edge->setInformation(Eigen::Matrix2d::Identity());//信息矩阵是二维方阵，因为误差是二维
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
    cout << "pose estimated by g2o =\n" << v->estimate().matrix() << endl;
    pose = v->estimate();
}

void bundleAdjustment(const vector<cv::Point3f> points_3d,
                      const vector<cv::Point2f> points_2d,
                      const cv::Mat& K,
                      cv::Mat& R, cv::Mat& t)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;// pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block (unique_ptr<Block::LinearSolverType>(linearSolver));     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));

    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm( solver );   // 设置求解器

    // vertex 有两个顶点，一个是优化位姿，一个是优化特征点的空间位置
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
            R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
            R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0))));
    optimizer.addVertex(pose);

    int index = 1;
    for (const cv::Point3f p:points_3d) // landmarks
    {
        g2o::VertexPointXYZ* point = new g2o::VertexPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);// g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
            K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const cv::Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        // 信息矩阵：协方差矩阵之逆 这里为1表示加权为1
        edge->setInformation ( Eigen::Matrix2d::Identity() );//这里的信息矩阵可以参考:http://www.cnblogs.com/gaoxiang12/p/5244828.html 里面有说明
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( false );// 打开调试输出，也就是输出迭代信息的设置
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}

