#include <iostream>
using namespace std;

#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 100

int main(int argc, char** argv)
{
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic> A;
    A = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    A = A.transpose() * A;//乔利斯基分解需要正定矩阵
    Eigen::Matrix<double, Eigen::Dynamic,1> B;
    B = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
    Eigen::Matrix<double, Eigen::Dynamic,1> X;
    X = A.llt().solve(B);
    cout << "Cholesly's = " << X << endl;
    X = A.colPivHouseholderQr().solve(B);
    cout << "QR's = " << X << endl;
    return 0;

}