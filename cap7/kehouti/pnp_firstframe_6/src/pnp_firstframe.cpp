#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
using namespace std;
using namespace cv;
#define  MyselfBAFunc  1        //1 课后习题6需要的BA优化函数
//0 例程用的
void find_feature_matches (
        const Mat& img_1, const Mat& img_2,
        std::vector<KeyPoint>& keypoints_1,
        std::vector<KeyPoint>& keypoints_2,
        std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );
#if MyselfBAFunc
void MyselfBAFun(
        const vector< cv::Point3f> &points1_3d,  //第一帧3d点(匹配好且有深度信息的点)
        const vector< cv::Point2f> &points1_2d,  //第一帧像素平面2d点(匹配好的点)
        const vector< cv::Point2f> &points2_2d,  //第二帧像素平面2d点(匹配好的点)
        const Mat&K,                             //因为里面没有修改相应的值所以用const
        const Mat&R,
        const Mat&t
);
#else
void bundleAdjustment (
    const vector<Point3f> &points_3d,
    const vector<Point2f> &points_2d,
    const Mat& K,
    Mat& R, Mat& t
);
#endif
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
#if MyselfBAFunc
    vector<Point2f> pts1_2d;                                  //第一帧下的像素坐标
#endif
    for ( DMatch m:matches )
    {
        //可以参考书上101页对应的程序，表示获取对应位置的深度图片的深度值
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );//表示的是 相机坐标系下的3D坐标 //这里是通过RGBD获取的深度信息，但是我们可以用三角测量获得第一帧下的3D坐标
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );      //第二帧 匹配好的点 2D像素坐标
#if MyselfBAFunc
        pts1_2d.push_back( keypoints_1[m.queryIdx].pt );      //第一帧 匹配好的点 2D像素坐标
#endif
    }
    //上面已经获得了第一帧坐标系的下的3d坐标 相当于世界坐标系下的坐标 （因为仅仅有两针图像帧 所以 以第一帧为世界坐标，也就是说 世界坐标到第一帧图像的R=I T=0 ）
    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;
    Mat r, t; //定义旋转和平移变量
    //参数信息: 第一帧3D 第二帧像素2D 内参矩阵k 无失真补偿  旋转向量r 平移向量t false表示输入的r t不作为初始化值 如果是true则此时会把t r作为初始值进行迭代
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false,SOLVEPNP_EPNP ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法SOLVEPNP_EPNP
    Mat R;
    cv::Rodrigues ( r, R );                                 // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;
    cout<<"calling bundle adjustment"<<endl;
#if MyselfBAFunc
    MyselfBAFun( pts_3d, pts1_2d, pts_2d, K, R, t);
#else
    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
#endif

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

#if MyselfBAFunc
void MyselfBAFun(
        const vector< cv::Point3f> &points1_3d,  //第一帧3d点(匹配好且有深度信息的点)
        const vector< cv::Point2f> &points1_2d,  //第一帧像素平面2d点(匹配好的点)
        const vector< cv::Point2f> &points2_2d,  //第二帧像素平面2d点(匹配好的点)
        const Mat&K,                             //因为里面没有修改相应的值所以用const
        const Mat&R,
        const Mat&t
){
#define PoseVertexNum 2                          //定义位姿节点个数 本试验仅仅有2帧图

//设置优化器
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  //优化位姿6维  优化路标点3维
    std::unique_ptr<Block::LinearSolverType> linearSolver=g2o::make_unique < g2o::LinearSolverCSparse<Block::PoseMatrixType> >();//线性求解设为CSparse
    std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver) ) );
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );

/*  Block::LinearSolverType *linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();  //设置线性求解器类型
    Block *solver_ptr = new Block( std::unique_ptr<Block::LinearSolverType>(linearSolver) );  //矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg( std::unique_ptr<g2o::Solver>(solver_ptr) ); //LM优化算法
*/
    g2o::SparseOptimizer optimizer;     //设置稀疏优化器
    optimizer.setAlgorithm(solver);     //设置优化算法

//向优化器中添加节点和边
    //添加节点 Vertex
    //（1）添加位姿节点 第一帧作为世界坐标系（不优化） 同时也是相机坐标系
    int poseVertexIndex = 0;                                       //位姿节点索引为0  总共两个位姿态节点 第一帧和第二帧
    Eigen::Matrix3d R2Init;
    R2Init <<
           R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ) ,
            R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ) ,
            R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 ) ;
    for( ; poseVertexIndex < PoseVertexNum ; poseVertexIndex++ )
    {
        auto pose = new g2o::VertexSE3Expmap();  //相机位姿
        pose->setId( poseVertexIndex );                            //设置节点标号
        pose->setFixed( poseVertexIndex == 0 );                    //如果是第一帧 则固定住 不优化
        if( poseVertexIndex == 1 )                                 //第二帧时让RT估计值为pnp得到的值 加快优化速度!
            pose->setEstimate(
                    g2o::SE3Quat( R2Init,
                                  Eigen::Vector3d( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                    )
            );                                    //两帧图像的位姿预设值都为 r=单位矩阵 t=0(当然这里可以填写自己设定的预设值 比如Pnp估计值)
        else
            pose->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( pose );                               //位姿节点加入优化器
    }
    //（2）添加路标节点
    int landmarkVertexIndex = PoseVertexNum ;
    for( int i = 0;  i < points1_3d.size() ; i ++ ){
        auto point = new g2o::VertexPointXYZ();                 //路标点
        point->setId( landmarkVertexIndex + i );                   //设置路标点节点标号
        point->setMarginalized( true );                            //设置边缘化
        point->setEstimate( Eigen::Vector3d( points1_3d[i].x, points1_3d[i].y, points1_3d[i].z ) ); //设置估计值 实际上就是第一帧坐标下的3d点坐标（也是世界坐标系下的坐标）
        optimizer.addVertex( point );                              //路标节点加入优化器
    }
    //加入相机参数（当然有另一种方式：查看笔记两种边的不同点）(最后一项为0 默认fx = fy 然后优化位姿 与g2o::EdegeSE3ProjectXYZ不同 笔记以记载 )
    g2o::CameraParameters *camera = new g2o::CameraParameters(
            K.at< double >(0,0), Eigen::Vector2d( K.at< double >(0,2), K.at< double >(1,2) ),0
    );
    camera->setId( 0 );
    optimizer.addParameter( camera );

    //加入边edge
    //世界坐标下路标点连接到第一帧位姿节点（因为以第一帧坐标系当做世界坐标系 所以我们前面没有优化第一帧的RT  仅仅优化第一帧到第二帧的RT）
    for(int i= 0 ;i < points1_2d.size() ; i++ ){
        auto edge = new g2o::EdgeProjectXYZ2UV();               //设置连接到第一帧的边
        //二元边 连接节点
        edge->setVertex( 0, dynamic_cast< g2o::VertexPointXYZ * >( optimizer.vertex( landmarkVertexIndex + i ) ) ); //世界坐标系下的路标节点
        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap * >( optimizer.vertex(0) ) );
        edge->setMeasurement( Eigen::Vector2d ( points1_2d[i].x, points1_2d[i].y ) );   //设置测量值为第一帧下的相机归一化平面坐标
        edge->setParameterId(0,0); //最后一位设置使用的相机参数(因为上面仅仅输入了一个相机参数id=0, 对应上面camer->setId(0),第一个参数0不知道是什么，但是必须为0)
        edge->setInformation ( Eigen::Matrix2d::Identity() );   //信息矩阵2x2的单位阵
        optimizer.addEdge( edge );
    }
    //第一帧路标点链接到第二帧位姿节点
    for( int i=0 ;i < points1_2d.size() ; i++){
        auto edge = new g2o::EdgeProjectXYZ2UV();   //设置链接到第二帧的边
        edge->setVertex( 0, dynamic_cast< g2o::VertexPointXYZ * >( optimizer.vertex( landmarkVertexIndex + i) ) ); //第一帧坐标系下的路标点
        edge->setVertex( 1, dynamic_cast< g2o::VertexSE3Expmap *> ( optimizer.vertex(1) ) ); //连接到第二个位姿节点
        edge->setMeasurement( Eigen::Vector2d ( points2_2d[i].x, points2_2d[i].y ) );        //设置测量值为第二帧下的相机归一化平面坐标
        edge->setInformation( Eigen::Matrix2d::Identity() ); //信息矩阵为2x2 实际上就是误差的加权为1：1的
        edge->setParameterId(0,0);
        optimizer.addEdge( edge );
    }

//run 算法
    cout<<"开始优化!"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );          //设置详细信息
    optimizer.initializeOptimization( );    //优化器初始化
    optimizer.optimize( 100 );              //进行最多100次的优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    cout<<"优化结束!"<<endl;
    chrono::duration< double > time_used = chrono::duration_cast< chrono::duration<double> >( t2 -t1 );
    cout<<"optimization costs time: "<<time_used.count() << "seconds."<<endl;
    cout<<endl<<"after optimization:"<<endl;
    //输出优化节点的位姿 estimate()输出的是SE3类型   Eigen::Isometry3d 实际上就是4x4的一个矩阵表示变换矩阵 这个类初始化可以是李代数
    //这里有一点不明白的是 Eigen::Isometry3d()为什么可以用estimate()返回的SE3Quat类型初始化？？？？？？？
    cout<<"T="<<endl<<Eigen::Isometry3d ( dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1))->estimate()).matrix() <<endl;
/*    g2o::SE3Quat  a();
    Eigen::Isometry3d( a);*/
}
#else
void bundleAdjustment (
    const vector< Point3f > &points_3d,
    const vector< Point2f > &points_2d,//这里加不加引用 都是会报错
    const Mat& K,
    Mat& R, Mat& t )
{
      // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    std::unique_ptr<Block::LinearSolverType> linearSolver( new g2o::LinearSolverCSparse<Block::PoseMatrixType>() );
    std::unique_ptr<Block> solver_ptr ( new Block ( std::move(linearSolver) ) );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr) );
/*    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( std::unique_ptr<Block::LinearSolverType>(linearSolver) );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr<g2o::Solver>(solver_ptr) );*/
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    //设置顶点估计值 然后才会用BA再次优化
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f &p:points_3d )   // landmarks 在执行这里的时候，实际上是所有空间点（匹配好的）组成的顶点
    {
        g2o::VertexPointXYZ* point = new g2o::VertexPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容   待注释？？
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2f &p:points_2d )//每个点都会与位姿节点链接 所以就有那么多的边
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        //下行转化 要用动态类型转换
        //将2元边连接上顶点
        edge->setVertex ( 0, dynamic_cast<g2o::VertexPointXYZ*> ( optimizer.vertex ( index ) ) );//空间点类型指针
        edge->setVertex ( 1, pose );                                                                //位姿类型指针
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );                                      //设置测量值
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() ); //因为误差向量为2维，所以信息矩阵也是2维，这里设置加权为1 即单位阵
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}
#endif
