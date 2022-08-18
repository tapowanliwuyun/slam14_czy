#include <g2o/core/base_vertex.h>//g2o顶点（Vertex）头文件 视觉slam十四讲p141用顶点表示优化变量，用边表示误差项
#include <g2o/core/base_binary_edge.h>//g2o边（edge）头文件
#include <g2o/core/block_solver.h>//求解器头文件
#include <g2o/core/optimization_algorithm_levenberg.h>//列文伯格——马尔夸特算法头文件
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>//鲁棒核函数
#include <iostream>

#include "common.h"//使用common.h中定义的BALProblem类读入该文件的内容
#include "sophus/se3.hpp"

//using namespace Sophus;
using namespace Eigen;
using namespace std;

/// 姿态和内参的结构
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr)
    {
        //每个相机一共有一共有6维的姿态
        rotation = Sophus::SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));//旋转位姿
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);//转换矩阵
        focal = data_addr[6];//1维焦距
        //2维畸变参数
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// 将估计值放入内存
    void set_to(double *data_addr)
    {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];//旋转位姿
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];//转换矩阵
        data_addr[6] = focal;//1维焦距
        data_addr[7] = k1;//2维畸变参数
        data_addr[8] = k2;//2维畸变参数
    }

    Sophus::SO3d rotation;
    Vector3d translation = Vector3d::Zero();//置0
    double focal = 0;//初始化为0
    double k1 = 0, k2 = 0;//将2维畸变参数初始化为0
};

/// 位姿加相机内参的顶点，9维，前三维为so3，接下去为t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public://以下定义的成员变量和成员函数都是公有的
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题

    VertexPoseAndIntrinsics() {}

    // 重置
    virtual void setToOriginImpl() override //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
    {
        _estimate = PoseAndIntrinsics();
    }
    // 更新
    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = Sophus::SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);//更新量累加
        _estimate.focal += update[6];//1维焦距更新量累加
        _estimate.k1 += update[7];//2维畸变参数更新量累加
        _estimate.k2 += update[8];//2维畸变参数更新量累加
    }

    /// 根据估计值投影一个点
    Vector2d project(const Vector3d &point)
    {
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
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

    // 存盘和读盘：留空
    virtual bool read(istream &in) {} //istream类是c++标准输入流的一个基类
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
    virtual void computeError() override //virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
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

void SolveBA(BALProblem &bal_problem);//定义SolveBA函数

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
    bal_problem.WriteToPLYFile("final.ply");//存储最终点云

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;// 每个误差项优化变量维度为9，误差值维度为3
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
        optimizer.addVertex(v);// 设置连接的顶点
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);// g2o 中必须设置 marg 参见第十讲内容，//设置边缘化
        optimizer.addVertex(v);//将新增的顶点加入到图模型中
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());//信息矩阵
        edge->setRobustKernel(new g2o::RobustKernelHuber());//核函数 鲁棒核函数
        optimizer.addEdge(edge);//添加边
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);//设置迭代次数为40

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        //camera_block_size = bal_problem.camera_block_size();
        //*camera = cameras + bal_problem.camera_block_size() * bal_problem.camera_index()[i]
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