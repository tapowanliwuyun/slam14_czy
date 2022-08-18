#include <iostream>
#include <fstream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


#include <Eigen/Core>
#include <Eigen/Dense>

#include "sophus/so3.h"
#include "sophus/se3.h"

// 需要设置新的参数域，因此需要继承LocalParameterization
// 具体可以参考http://www.ceres-solver.org/nnls_modeling.html?highlight=localparameterization#localparameterization
class SE3Parameterization : public ceres::LocalParameterization
{
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}

    // 如果更新，此处使用李代数更新，而不是简单的加减
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const
    {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

        Sophus::SE3 T = Sophus::SE3::exp(lie);
        Sophus::SE3 delta_T = Sophus::SE3::exp(delta_lie);

        // 李代数左乘更新
        Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

        for(int i = 0; i < 6; ++i)
            x_plus_delta[i] = x_plus_delta_lie(i, 0);

        return true;
    }

    // 如果计算Jacobian，此处大概是局部求导；不过由于我已经在残差函数中仔细计算了导数，所以这里单位矩阵就好
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const
    {
        ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
        return true;
    }
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

// 自己求导数的类，需要重构SizedCostFunction
class PosegraphBA: public ceres::SizedCostFunction<6,6,6>
{
public:
    Sophus::SE3 deltaSE3Inv;

    PosegraphBA(double x,double y,double z,
                double s,double vx,double vy,double vz)
    {
        Eigen::Quaterniond q( s,vx, vy, vz );
        q.normalize();
        deltaSE3Inv = Sophus::SE3( q,Eigen::Vector3d( x, y, z ) ).inverse();
    }

    //  Evaluate是很重要的一个函数
    virtual bool Evaluate(double const* const* pose,
                          double *residual,
                          double **jacobians) const
    {
        // Get Pose A
        Eigen::Map<const Eigen::Matrix<double,6,1>> poseAVec6d(pose[0]);
        Sophus::SE3 poseASE3 = Sophus::SE3::exp(poseAVec6d);

        // Get Pose B
        Eigen::Map<const Eigen::Matrix<double,6,1>> poseBVec6d(pose[1]);
        Sophus::SE3 poseBSE3 = Sophus::SE3::exp(poseBVec6d);

        // Compute Error
        Sophus::SE3 errorSE3 = deltaSE3Inv*poseASE3.inverse()*poseBSE3;
        Eigen::Matrix<double,6,1> errorVec6d = errorSE3.log();

        // 残差项
        residual[0] = errorVec6d(0);
        residual[1] = errorVec6d(1);
        residual[2] = errorVec6d(2);
        residual[3] = errorVec6d(3);
        residual[4] = errorVec6d(4);
        residual[5] = errorVec6d(5);

        if( !jacobians )
            return true;

        if( !jacobians[0] && !jacobians[1] )
            return true;

        // 以下都是jacobian的公式，具体也对应了公式
        // 可以参考http://www.ceres-solver.org/nnls_modeling.html?highlight=evaluate#_CPPv2N5ceres12CostFunction8EvaluateEPPCdPdPPd
        {
            // 公式11.10 J_r^{-1}
            Eigen::Matrix<double,6,6> J;
            J.block(0,0,3,3) = Sophus::SO3::hat(errorSE3.so3().log());
            J.block(0,3,3,3) = Sophus::SO3::hat(errorSE3.translation());
            J.block(3,0,3,3) = Eigen::Matrix3d::Zero(3,3);
            J.block(3,3,3,3) = Sophus::SO3::hat(errorSE3.so3().log());
            J = J*0.5 + Eigen::Matrix<double,6,6>::Identity();

            // 公式11.8
            // row correspond with error
            // col correspond with parameterA
            Eigen::Matrix<double,6,6> jacA = - J * poseBSE3.inverse().Adj();

            jacobians[0][ 0] = jacA(0,0); jacobians[0][ 1] = jacA(1,0); jacobians[0][ 2] = jacA(2,0); jacobians[0][ 3] = jacA(3,0); jacobians[0][ 4] = jacA(4,0); jacobians[0][ 5] = jacA(5,0);
            jacobians[0][ 6] = jacA(0,1); jacobians[0][ 7] = jacA(1,1); jacobians[0][ 8] = jacA(2,1); jacobians[0][ 9] = jacA(3,1); jacobians[0][10] = jacA(4,1); jacobians[0][11] = jacA(5,1);
            jacobians[0][12] = jacA(0,2); jacobians[0][13] = jacA(1,2); jacobians[0][14] = jacA(2,2); jacobians[0][15] = jacA(3,2); jacobians[0][16] = jacA(4,2); jacobians[0][17] = jacA(5,2);
            jacobians[0][18] = jacA(0,3); jacobians[0][19] = jacA(1,3); jacobians[0][20] = jacA(2,3); jacobians[0][21] = jacA(3,3); jacobians[0][22] = jacA(4,3); jacobians[0][23] = jacA(5,3);
            jacobians[0][24] = jacA(0,4); jacobians[0][25] = jacA(1,4); jacobians[0][26] = jacA(2,4); jacobians[0][27] = jacA(3,4); jacobians[0][28] = jacA(4,4); jacobians[0][29] = jacA(5,4);
            jacobians[0][30] = jacA(0,5); jacobians[0][31] = jacA(1,5); jacobians[0][32] = jacA(2,5); jacobians[0][33] = jacA(3,5); jacobians[0][34] = jacA(4,5); jacobians[0][35] = jacA(5,5);

            // 公式11.9
            Eigen::Matrix<double,6,6> jacB =   J * poseBSE3.inverse().Adj();

            jacobians[1][ 0] = jacB(0,0); jacobians[1][ 1] = jacB(1,0); jacobians[1][ 2] = jacB(2,0); jacobians[1][ 3] = jacB(3,0); jacobians[1][ 4] = jacB(4,0); jacobians[1][ 5] = jacB(5,0);
            jacobians[1][ 6] = jacB(0,1); jacobians[1][ 7] = jacB(1,1); jacobians[1][ 8] = jacB(2,1); jacobians[1][ 9] = jacB(3,1); jacobians[1][10] = jacB(4,1); jacobians[1][11] = jacB(5,1);
            jacobians[1][12] = jacB(0,2); jacobians[1][13] = jacB(1,2); jacobians[1][14] = jacB(2,2); jacobians[1][15] = jacB(3,2); jacobians[1][16] = jacB(4,2); jacobians[1][17] = jacB(5,2);
            jacobians[1][18] = jacB(0,3); jacobians[1][19] = jacB(1,3); jacobians[1][20] = jacB(2,3); jacobians[1][21] = jacB(3,3); jacobians[1][22] = jacB(4,3); jacobians[1][23] = jacB(5,3);
            jacobians[1][24] = jacB(0,4); jacobians[1][25] = jacB(1,4); jacobians[1][26] = jacB(2,4); jacobians[1][27] = jacB(3,4); jacobians[1][28] = jacB(4,4); jacobians[1][29] = jacB(5,4);
            jacobians[1][30] = jacB(0,5); jacobians[1][31] = jacB(1,5); jacobians[1][32] = jacB(2,5); jacobians[1][33] = jacB(3,5); jacobians[1][34] = jacB(4,5); jacobians[1][35] = jacB(5,5);

        }

        return true;
    }
};



int main( int argc, char *argv[] )
{
    google::InitGoogleLogging(argv[0]);

    if(argc<2)
    {
        std::cerr<<"./pose_graph_ceres_SE3 sphere.g2o"<<std::endl;
        return -1;
    }

    std::cout<<"Input g2o file: "<<argv[1]<<std::endl;


    std::ifstream g2oFile( argv[1] );
    if ( !g2oFile )
    {
        std::cout<<"file "<<argv[1]<<" does not exist."<<std::endl;
        return -1;
    }
    // Count Pose and Edge
    int poseCount = 0;
    int edgeCount = 0;
    std::string fileLine;
    while( std::getline(g2oFile,fileLine) )
    {
        if(fileLine[0]=='V')
        {
            poseCount++;
        }
        if(fileLine[0]=='E')
        {
            edgeCount++;
        }
    }
    g2oFile.clear();
    g2oFile.seekg(std::ios::beg);


    std::cout<<poseCount<<std::endl;
    std::cout<<edgeCount<<std::endl;


    // Init Ceres
    ceres::Problem problem;

    // Load Data
    double *poseData = new double[poseCount*6];
    for( int i=0; i<poseCount; i++ )
    {
        std::string flag;
        int id;
        double x,y,z,s,vx,vy,vz;
        g2oFile>>flag>>id>>x>>y>>z>>vx>>vy>>vz>>s;
        Eigen::Quaterniond q( s,vx, vy, vz );
        q.normalize();
        Eigen::Matrix<double,6,1> poseVec6d = Sophus::SE3( q,Eigen::Vector3d( x, y, z ) ).log();

        poseData[6*i+0] = poseVec6d(0);
        poseData[6*i+1] = poseVec6d(1);
        poseData[6*i+2] = poseVec6d(2);
        poseData[6*i+3] = poseVec6d(3);
        poseData[6*i+4] = poseVec6d(4);
        poseData[6*i+5] = poseVec6d(5);
    }

    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();

    // Add Residual
    for( int i=0; i<edgeCount; i++ )
    {
        std::string flag;
        int idA,idB;
        double x,y,z,s,vx,vy,vz;
        double temp;
        g2oFile>>flag>>idA>>idB>>x>>y>>z>>vx>>vy>>vz>>s;
        // I dont't know how to use info
        for( int j=0; j<21; j++ )
        {
            g2oFile>>temp;
        }

        ceres::CostFunction *costFunction = new PosegraphBA(x,y,z,s,vx,vy,vz);

        problem.AddResidualBlock (  costFunction,
                                    nullptr,
                                    poseData+6*idA,
                                    poseData+6*idB
        );

        problem.SetParameterization(poseData+6*idA,local_parameterization);
        problem.SetParameterization(poseData+6*idB,local_parameterization);

    }
    g2oFile.close();


    // Set Ceres
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve ( options, &problem, &summary );

    // Report
    std::cout<<summary.FullReport() <<std::endl;
    std::ofstream txt("1.txt");
    for( int i=0; i<poseCount; i++ )
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>> poseAVec6d( poseData+6*i );
        Sophus::SE3 poseSE3 = Sophus::SE3::exp(poseAVec6d);

        txt<<poseSE3.translation().transpose()<<std::endl;
    }

    txt.close();


    // Release Data
    delete []poseData;

    return 0;
}

