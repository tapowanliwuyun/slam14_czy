#include <iostream>
#include <ceres/ceres.h>
#include "common.h"//使用common.h中定义的BALProblem类读入该文件的内容
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);//定义SolveBA函数

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;//输出使用方法
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();//归一化 将所有路标点的中心置零，然后做一个合适尺度的缩放
    bal_problem.Perturb(0.1, 0.5, 0.5);//通过Perturb函数给数据加入噪声
    bal_problem.WriteToPLYFile("initial.ply");//存储最初点云
    SolveBA(bal_problem);//BA求解
    bal_problem.WriteToPLYFile("final_Ceres.ply");//存储最终点云

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        //camera_block_size = bal_problem.camera_block_size();
        //*camera = cameras + bal_problem.camera_block_size() * bal_problem.camera_index()[i]
        double *point = points + point_block_size * bal_problem.point_index()[i];
        //point_block_size = bal_problem.point_block_size();
        //*point = points + bal_problem.point_block_size() * bal_problem.point_index()[i]
        problem.AddResidualBlock(cost_function, loss_function, camera, point);// 向问题中添加误差项
        //CostFunction* : 描述最小二乘的基本形式即代价函数
        //LossFunction* : 描述核函数的形式
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;//输出bal problem file loaded...
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;//bal_problem.num_cameras()表示相机位姿个数
    //bal_problem.num_points()表示路标点数
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;//BA求解
    ceres::Solver::Options options;// 这里有很多配置项可以填Options类嵌入在Solver类中 ，在Options类中可以设置关于求解器的参数
    options.linear_solver_type = ceres::SPARSE_SCHUR;  //增量方程如何求解 这里的linear_solver_type 是一个Linear_solver_type的枚举类型的变量
    //使用Schur消元
    options.minimizer_progress_to_stdout = true;


    //SetOrdering
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras
    // 增加的部分
    for(int i = 0; i < bal_problem.num_points(); ++i)
        if(i%2 == 0){
            ordering->AddElementToGroup(points + point_block_size * i, 0);
        }
        else{
            ordering->AddElementToGroup(points + point_block_size * i, 1);
        }

    for(int i = 0; i < bal_problem.num_cameras(); ++i)
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);

    options.linear_solver_ordering.reset(ordering);

    ceres::Solver::Summary summary;// 优化信息
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

