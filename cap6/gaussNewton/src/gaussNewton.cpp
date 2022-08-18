#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;

int main(int argc,char ** argv) {
    double ar = 1.0, br = 2.0, cr = 1.0;  // true parameter value 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameter value 估计参数值
    int N = 100; // the data points 数据点
    double w_sigma = 1.0; //sigma of noise 噪声sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器   OpenCV random number generator

    vector<double> x_data, y_data; //data 数据
    for (int i = 0; i < N; i++)
    {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x+br*x+cr) + rng.gaussian(w_sigma*w_sigma) );
    }

    //开始高斯牛顿迭代    start gauss-Newton iteration
    int iterations = 100;// the number of iterations 迭代次数
    double cost = 0, lastCost = 0;//本次迭代的cost和上一次迭代的cost   cost of the current iteration and cost of the previous iteration
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    for(int iter = 0; iter < iterations; iter++)
    {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); // Hession = J^T W^{-1} J in Gauss-Newton
        Eigen::Vector3d b = Eigen::Vector3d::Zero(); // bias
        cost = 0;

        for(int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // the ith data point 第i个数据点
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Eigen::Vector3d J;//Jacobian matrix 雅可比矩阵

            J[0] = -xi * xi * exp( ae * xi * xi + be * xi + ce );
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = -exp(ae * xi * xi + be * xi + ce);

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * J * error;

            cost += error * error;
        }

        //求解线性方程Hx = b;  Solve the linear equation Hx = b;
        Eigen::Vector3d dx = H.ldlt().solve(b);

        if(isnan(dx[0]))
        {
            cout << " result is nan!" << endl;
            break;
        }

        if(iter > 0 && cost >= lastCost)
        {
            cout << "cost: " << cost << ">=last_cost: " << lastCost << ", break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << iter <<"total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae <<"," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << "seconds." << endl;
    cout << "estimated abc：" <<  ae << "," << be << "," << ce << endl;
    return 0;
}