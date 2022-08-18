#include <g2o/core/base_vertex.h>//g2o顶点（Vertex）头文件 视觉slam十四讲p141用顶点表示优化变量，用边表示误差项
#include <g2o/core/base_binary_edge.h>//g2o边（edge）头文件
#include <g2o/core/block_solver.h>//求解器头文件
#include <g2o/core/optimization_algorithm_levenberg.h>//列文伯格——马尔夸特算法头文件
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>//鲁棒核函数
#include <iostream>

#include "common.h"//使用common.h中定义的BALProblem类读入该文件的内容
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}
    
    // camera : 12 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length in x direction , [7] second order radial distortion , [8] forth order radial distortion
    // [9-11] : camera parameter, [9] focal length in y direction, [10] first coefficient tangential distortion, [11] scond coefficient tangential distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    
    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        fx = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
	fy = data_addr[9];
	p1 = data_addr[10];
        p2 = data_addr[11];
    }

    /// 将估计值放入内存
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = fx;
        data_addr[7] = k1;
        data_addr[8] = k2;
	    data_addr[9] = fy;
        data_addr[10] = p1;
        data_addr[11] = p2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double fx = 0, fy = 0;//初始化为0
    double k1 = 0, k2 = 0;//初始化为0
    double p1 = 0, p2 = 0;//初始化为0
};

/// 位姿加相机内参的顶点，12维，前三维为so3，接下去为t, fx, k1, k2, fy, p1, p2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<12, PoseAndIntrinsics> {
public://以下定义的成员变量和成员函数都是公有的
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题

    VertexPoseAndIntrinsics() {}
    
    // 重置
    virtual void setToOriginImpl() override//virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
    {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);//更新量累加
        _estimate.fx += update[6];//fx更新量累加
        _estimate.k1 += update[7];//k1更新量累加
        _estimate.k2 += update[8];//k2更新量累加
	_estimate.fy += update[9];//fy更新量累加
	_estimate.p1 += update[10];//p1更新量累加
	_estimate.p2 += update[11];//p2更新量累加
    }

    /// 根据估计值投影一个点
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
	double xp = pc[0], yp = pc[1];
	double xy = xp * yp;
	double x2 = xp * xp;
	double y2 = yp * yp;
	double r2 = pc.squaredNorm();	
        
    double radical_distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
	double tangential_distortion_x = 2.0 * _estimate.p1 * xy + _estimate.p2 * (r2 + 2.0 * x2);
	double tangential_distortion_y = _estimate.p1 * (r2 + 2.0 * y2) + 2.0 * _estimate.p2 * xy;
	
        return Vector2d(_estimate.fx * (radical_distortion * xp + tangential_distortion_x),
                        _estimate.fy * (radical_distortion * yp + tangential_distortion_y));
    }

   // 存盘和读盘：留空
   virtual bool read(istream &in) {} //istream类是c++标准输入流的一个基类
   //可参照C++ Primer Plus第六版的6.8节
   virtual bool write(ostream &out) const {} //ostream类是c++标准输出流的一个基类
   //可参照C++ Primer Plus第六版的6.8节
};

class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public://以下定义的成员变量和成员函数都是公有的
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题

    VertexPoint() {}
    // 重置
    virtual void setToOriginImpl() override //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
    {
        _estimate = Vector3d(0, 0, 0);
    }
    // 更新
    virtual void oplusImpl(const double *update) override 
    {
        _estimate += Vector3d(update[0], update[1], update[2]);//更新量累加
    }

    virtual bool read(istream &in) {}//istream类是c++标准输入流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
  virtual bool write(ostream &out) const {} //ostream类是c++标准输出流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
};
// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public://以下定义的成员变量和成员函数都是公有的
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题
    // 计算误差
    virtual void computeError() override  //virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
    {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives
    virtual bool read(istream &in) {}//istream类是c++标准输入流的一个基类
    //可参照C++ Primer Plus第六版的6.8节
    virtual bool write(ostream &out) const {}//ostream类是c++标准输出流的一个基类
    //可参照C++ Primer Plus第六版的6.8节

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;//输出使用方法
        return 1;
    }

    BALProblem bal_problem(argv[1]);
   bal_problem.Normalize();//归一化 将所有路标点的中心置零，然后做一个合适尺度的缩放
    bal_problem.Perturb(0.1, 0.5, 0.5);//通过Perturb函数给数据加入噪声
    bal_problem.WriteToPLYFile("initial.ply");//存储最初点云
    SolveBA(bal_problem);//BA求解
    bal_problem.WriteToPLYFile("final_g2o.ply");//存储最终点云


    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // pose dimension 12, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<12, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        //c++中的make_unique表示智能指针类型
    g2o::SparseOptimizer optimizer;// 图模型
    optimizer.setAlgorithm(solver);// 设置求解器
    optimizer.setVerbose(true);// 打开调试输出

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));//camera表示优化变量
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(100);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}
