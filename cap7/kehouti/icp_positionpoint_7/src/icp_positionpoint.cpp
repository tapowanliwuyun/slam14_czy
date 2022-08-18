#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <g2o/core/optimization_algorithm_levenberg.h>

using namespace std;
using namespace cv;
/*+++++++++++++++++++++++++++++++++++++++++配置信息+++++++++++++++++++++++++++++++++++++++++++++***/

#define  MyselfOptimizerMethod  1   //  1: 课后习题7代码结果
//  0: 3d-3d书上例子
#define  ISBAProvideInitValue   0   //  1: 用ICP结果为BA提供初值
//  0: 用单位矩阵I和0平移向量作为初始值

#define  BAOptimizer            1   //  1: 加入BA优化
//  0: 不加入BA优化

/*+++++++++++++++++++++++++++++++++++++++++End配置信息+++++++++++++++++++++++++++++++++++++++++++++*/

void find_feature_matches (
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void pose_estimation_3d3d (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
);

#if BAOptimizer
void bundleAdjustment (
        const vector< Point3f >& pts1,
        const vector< Point3f >& pts2,
        Mat& R, Mat& t );

#if MyselfOptimizerMethod
//课后习题7题
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseBinaryEdge<3, Eigen::Vector3d,g2o::VertexPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( ) {}

    virtual void computeError()
    {
        const g2o::VertexPointXYZ * point  = dynamic_cast< const g2o::VertexPointXYZ *> ( _vertices[0] );
        const g2o::VertexSE3Expmap * pose = dynamic_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
        // measurement is p, point is p'
        //pose->estimate().map( _point );中用estimate（）估计一个值后，然后用映射函数 就是旋转加平移 将其_point映射到另一个相机坐标系下去
        //观测值 - 映射值 因为我们做的试验是3D-3D 所以这里把第一帧的3D坐标当做测量值，然后把第二帧坐标映射到第一帧坐标系中
        _error = _measurement - pose->estimate().map( point->estimate() );
    }
    //表示线性化 误差函数 就是要计算雅克比矩阵
    virtual void linearizeOplus()override final                //这里override 表示override覆盖基类的同名同参函数， final表示派生类的某个函数不能覆盖这个函数
    {
        g2o::VertexSE3Expmap* pose = dynamic_cast<g2o::VertexSE3Expmap *> (_vertices[1]);
        g2o::VertexPointXYZ * point = dynamic_cast< g2o::VertexPointXYZ * > (_vertices[0] );
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map( point->estimate() );//映射到第二帧相机坐标系下的坐标值
        double x = xyz_trans[0];                               //第一帧到第二帧坐标系下变换后的坐标值
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        //关于空间点的雅克比矩阵-R, 误差对空间点，也就是第二张图片的空间点求导，de/dp'，可以看书上p187
        _jacobianOplusXi = -T.rotation().toRotationMatrix();

        //3x6的关于优化变量的雅克比矩阵 可以看书上p187/p86页 自己推到的结果
        _jacobianOplusXj(0,0) = 0;
        _jacobianOplusXj(0,1) = -z;
        _jacobianOplusXj(0,2) = y;
        _jacobianOplusXj(0,3) = -1;
        _jacobianOplusXj(0,4) = 0;
        _jacobianOplusXj(0,5) = 0;

        _jacobianOplusXj(1,0) = z;
        _jacobianOplusXj(1,1) = 0;
        _jacobianOplusXj(1,2) = -x;
        _jacobianOplusXj(1,3) = 0;
        _jacobianOplusXj(1,4) = -1;
        _jacobianOplusXj(1,5) = 0;

        _jacobianOplusXj(2,0) = -y;
        _jacobianOplusXj(2,1) = x;
        _jacobianOplusXj(2,2) = 0;
        _jacobianOplusXj(2,3) = 0;
        _jacobianOplusXj(2,4) = 0;
        _jacobianOplusXj(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
};
#else
// g2o edge 边代表误差  所以在里面要override一个computerError函数
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = dynamic_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        //pose->estimate().map( _point );中用estimate（）估计一个值后，然后用映射函数 就是旋转加平移 将其_point映射到另一个相机坐标系下去
        //观测值 - 映射值 因为我们做的试验是3D-3D 所以这里把第一帧的3D坐标当做测量值，然后把第二帧坐标映射到第一帧坐标系中
        _error = _measurement - pose->estimate().map( _point );
    }
    //表示线性化 误差函数 就是要计算雅克比矩阵
    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = dynamic_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);//映射到第二帧相机坐标系下的坐标值
        double x = xyz_trans[0];                  //第一帧到第二帧坐标系下变换后的坐标值
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        //3x6的关于优化变量的雅克比矩阵 可以看书上p179页 自己推到的结果
        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;                     //设立误差点 之后将其与测量值进行相减 求得误差
};
#endif

#endif
//自己设置的打印配置信息
void printConfigInfo(){
    cout<<"This is the example from GaoXiang's book : ICP结果是第二帧到第一帧的RT."<<endl;
#if BAOptimizer

    cout<<"BA Optimizer is Provided!"<<endl;
#if ISBAProvideInitValue
    cout<<"The RT from ICP's result is Provided for BA as a initialization."<<endl;
#else
    cout<<" But,Not provide initialization for BA!"<<endl;
#endif

#if  MyselfOptimizerMethod
    cout<<"优化了空间点和位姿节点！"<<endl;
#else
    cout<<"未对空间点进行优化！"<<endl;
#endif

#endif
}
int main ( int argc, char** argv )
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
    assert(img_1.data != nullptr && img_2.data != nullptr);//若读取的图片没有内容，就终止程序

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts1, pts2;

    for ( DMatch m:matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /5000.0;
        float dd2 = float ( d2 ) /5000.0;

        //存储特征点的3D坐标
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

    cout<<"3d-3d pairs: "<<pts1.size() <<endl;
    Mat R, t;
    pose_estimation_3d3d ( pts1, pts2, R, t );                      //ICP方式的位姿估计
    cout<<"ICP via SVD results: "<<endl;
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;
    //实际上是第一帧到第二帧的R T
    cout<<"第一帧到第二帧的旋转和平移:" << endl << "R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

    cout<<"calling bundle adjustment"<<endl;

#if  BAOptimizer
    bundleAdjustment( pts1, pts2, R, t );                          //BA优化估计相机位姿  RT 是提供优化的初始值
#endif

    // verify p1 = R*p2 + t
/*    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
            <<endl;
        cout<<endl;
    }
  */
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
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
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

void pose_estimation_3d3d (
        const vector<Point3f>& pts1,
        const vector<Point3f>& pts2,
        Mat& R, Mat& t
)
{
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center 去质心坐标
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );//因为知道矩阵W的类型 所以在分解的时候直接是FullU | FullV
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    //例程本身的实现方式 求出的R T是第二帧到第一帧的
    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
                               R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
            R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
            R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
    );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}
#if  BAOptimizer
void bundleAdjustment (
        const vector< Point3f >& pts1,
        const vector< Point3f >& pts2,
        Mat& R, Mat& t )                //实际上 R t在这里并不是必要的 这个只是用来提供BA初始值
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose维度为 6, landmark 维度为 3
    std::unique_ptr<Block::LinearSolverType>linearSolver = g2o::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>();   //线性方程求解器
    std::unique_ptr<Block>solver_ptr = g2o::make_unique<Block>( std::move(linearSolver) );                                      //矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );            //LM法

/*  //另一种修改错误的方式
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();  // 线性方程求解器
    Block* solver_ptr = new Block( std::unique_ptr<Block::LinearSolverType>(linearSolver) );      // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr <g2o::Solver>(solver_ptr) ); //LM法
*/
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // vertex   这里仅仅添加了第二帧节点
    auto pose = new g2o::VertexSE3Expmap();                  // camera pose
    pose->setId(0);                                          // 位姿节点编号为0
#if ISBAProvideInitValue                                     // 为图优化提供ICP结果作为初值
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
            R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
            R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setEstimate(  g2o::SE3Quat(
            R_mat,
            Eigen::Vector3d( t.at<double> ( 0,0 ),t.at<double> ( 0,1 ),t.at<double> ( 0,2 ) )
                                    )
                     );
#else                                                       // 提供默认初值
    pose->setEstimate( g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d( 0,0,0 )
    ) );
#endif
    optimizer.addVertex( pose );                            //向优化器中添加节点
#if MyselfOptimizerMethod                                   //添加空间节点  并且按照书上的方式进行优化的
    int pointIndex = 1;                                     //因为上面的位姿节点就1个  所以我们这里标号为1
    for( auto &p: pts2 ){
        auto point = new g2o::VertexPointXYZ();
        point->setId( pointIndex++ );
        point->setMarginalized( true );                     //设置边缘化(必须设置 否则出错)
        point->setEstimate( Eigen::Vector3d( p.x, p.y, p.z ) );
        optimizer.addVertex( point );
    }
#endif
    // edges
    int index = 0;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
#if MyselfOptimizerMethod
        auto edge = new EdgeProjectXYZRGBDPoseOnly( );      //在课后习题中修改的EdgeProjectXYZRGBDPoseOnly可以去掉_point成员变量
#else
        auto edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
#endif
        edge->setId( index );
#if  MyselfOptimizerMethod
        edge->setVertex( 0 , dynamic_cast< g2o::VertexPointXYZ *> ( optimizer.vertex(index)) );
        edge->setVertex( 1 , dynamic_cast< g2o::OptimizableGraph::Vertex *> ( optimizer.vertex(0) ) );
#else   //参考ORB_SLAM:这里将原来的g2o::VertexSE3Expmap* 替换为g2o::OptimizableGraph::Vertex *
        edge->setVertex( 0 , dynamic_cast< g2o::OptimizableGraph::Vertex *> ( optimizer.vertex(0) ) );
#endif
        //这里以第一帧为测量值 说明优化的是第二帧到第一帧的旋转r和平移t
        edge->setMeasurement( Eigen::Vector3d (
                pts1[i].x, pts1[i].y, pts1[i].z)
        );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);                            //向优化器中添加边
        index++;
        edges.push_back(edge);                              //存储边指针
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
/*
    cout<<"输出优化后的point值:"<<endl;
    for (int j = 0; j <3 ; ++j) {
        cout<< dynamic_cast< g2o::VertexPointXYZ * > ( optimizer.vertex(j+1) )->estimate()<<endl<<endl;//Eigen::Vector3d
    }
    cout<<endl<<"优化前的点:"<<endl;
    for (int i = 0; i <3 ; ++i) {
        cout<<pts2[i]<<endl<<endl;
    }
*/

}
#endif

