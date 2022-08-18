#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW//表示在利用Eigen库的数据结构时new的时候 需要对齐，所以加入EIGEN特有的宏定义即可实现
    //下面几个虚函数都是覆盖了基类的对应同名同参数的函数
    virtual void setToOriginImpl() // 重置  这个虚函数override 覆盖了Vertex类的对应函数 函数名字和参数都是一致的，是多态的本质
    {
        _estimate << 0,0,0;//输入优化变量初始值
    }

    virtual void oplusImpl( const double* update ) // 更新 对于拟合曲线这种问题，这里更新优化变量仅仅是简单的加法，
    // 但是到了位姿优化的时候，旋转矩阵更新是左乘一个矩阵 此时这个更新函数就必须要重写了
    { //更新参数估计值
        _estimate += Eigen::Vector3d(update);
    }
    // 存盘和读盘：留空
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型  //这里观测值维度是1维，如果是124页6.12式，则观测值维度是2
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW//表示在利用Eigen库的数据结构时new的时候 需要对齐，所以加入EIGEN特有的宏定义即可实现
    CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型误差
    void computeError()
    {
        /*       _vertices是std::vector<Vertex *>类型的变量，我们这里把基类指针_vertices【0】强制转换成const CurveFittingVertex* 自定义子类的常量指针
        这里的转换是上行转换（子类指针转换到基类），对于static_cast 和dynamic_cast两种的结果都是一样的，但是对于这种下行转换则dynamic_cast比static_cast多了类型检查功能
        更安全些，但是dynamic_cast只能用在类类型的指针 引用，static_cast则不限制,即可以用在类型也可以用在其他类型，所以这里应该更改为dynamic_cast
        const CurveFittingVertex* v = dynamic_cast<const CurveFittingVertex*> (_vertices[0]);//但是这里我没有修改，因为我不懂这块不敢乱该，如果你觉得有道理你 就修改试试，改了也是正常运行的
        */
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        //获取此时待估计参数的当前更新值 为下面计算误差项做准备
        const Eigen::Vector3d abc = v->estimate();
        //这里的error是1x1的矩阵，因为误差项就是1个 _measurement是测量值yi
        _error(0,0) = _measurement - std::exp( abc(0,0)*_x*_x + abc(1,0)*_x + abc(2,0) ) ;
    }

/*
    //计算雅可比矩阵，没有用到,ke yi yong
   virtual void linearizeOplus() override{
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x *_x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
 }
 */
    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}
public:
    double _x;  // x 值， y 值为 _measurement
};

int main( int argc, char** argv )
{
    double ar=1.0, br=2.0, cr=1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; //估计参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    double abc[3] = {0,0,0};            // abc参数的估计值

    vector<double> x_data, y_data;      // 数据

    //cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
                exp ( ar*x*x + br*x + cr ) + rng.gaussian ( w_sigma )
        );
        // cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

/*
原版错误方式 : 这样会出错
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1  后面的那个参数与误差变量无关 仅仅表示路标点的维度 这里因为没有用到路标点 所以为什么值都可以
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );//LM法
*/

//第一种实现方法start
/*
第一种解决方式: 将普通指针强制转换成智能指针 需要注意的是 转化之后 原来的普通指针指向的内容会有变化
 普通指针可以强制转换成智能指针，方式是通过智能指针的一个构造函数来实现的， 比如下面的Block( std::unique_ptr<Block::LinearSolverType>( linearSolver ) );
 这里面就是将linearSolver普通指针作为参数用智能指针构造一个临时的对象，此时原来的普通指针就无效了，一定不要再次用那个指针了，否则会有意想不到的错误，如果还想保留原来的指针
 那么就可以利用第二种方式 定义的时候就直接用智能指针就好，但是就如第二种解决方案那样，也会遇到类型转换的问题。详细见第二种方式说明
 */
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1  后面的那个参数与误差变量无关 仅仅表示路标点的维度 这里因为没有用到路标点 所以为什么值都可以
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
    //Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器
    Block* solver_ptr = new Block( unique_ptr<Block::LinearSolverType>(linearSolver) );      // 矩阵块求解器
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( unique_ptr<Block>(solver_ptr) );
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(  unique_ptr<Block>(solver_ptr)  );
    //g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(  unique_ptr<Block>(solver_ptr)  );
//第一种实现方法end

/*
//第二种实现方法start
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > BlockSolverType;  // 每个误差项优化变量维度为3，误差值维度为1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; //线性求解器类型
    // 梯度下降方法，从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//第二种实现方法end
*/

/*

/*第三种解决方案: 定义变量时就用智能指针 需要注意的是 需要std::move移动
 *下面可以这样做 std::make_unique<>是在c++14中引进的 而std::make_shared<>是在c++11中引进的,都是为了解决用new为智能指针赋值的操作。这种更安全。
 *  对于（2）将linearSovler智能指针的资源利用移动构造函数转移到新建立的Block中，此时linearSolver这个智能指针默认不能够访问以及使用了。
 *  对于(3)来说，因为solver_ptr是一个指向Block类型的智能指针，但是g2o::OptimizationAlgorithmLevenberg 构造函数接受的是std::unique_ptr<Solver>的参数，引起冲突，但是智能指针指向不同的类型时，
 *  不能够通过强制转换，所以此时应该用一个std::move将一个solver_ptr变为右值，然后调用std::unique_ptr的移动构造函数，而这个函数的本身并没有限制指针
 *  指向的类型，只要是std::unique_ptr类的对象，我们就可以调用智能指针的移动构造函数进行所属权的移动。
 *
 * */
/*
//第三种实现方法start
    std::unique_ptr<Block::LinearSolverType>linearSolver( new g2o::LinearSolverDense<Block::PoseMatrixType>() );// 线性方程求解器(1)
    std::unique_ptr<Block> solver_ptr ( new  Block( std::move(linearSolver) ) );// 矩阵块求解器 (2)
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( std::move(solver_ptr) );//(3) LM法
//第三种实现方法end
    */


    //梯度下降方法，从GN, LM, DogLeg 中选(下面的两种方式要按照上面的两种解决方案对应修改，否则会编译出错 )
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    // g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg( solver_ptr );
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm( solver );   // 设置求解器
    optimizer.setVerbose( true );       // 打开调试输出

    // 往图中增加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate( Eigen::Vector3d(ae,be,ce) );//迭代5次满足迭代阈值 //增加顶点的初始值，如果是位姿 则初始值是用ICP PNP来提供初始化值
    v->setId(0);//增加顶点标号 多个顶点要依次增加编号
    optimizer.addVertex( v );//将新增的顶点加入到图模型中

    // 往图中增加边 N个
    for ( int i=0; i<N; i++ )
    {
        CurveFittingEdge* edge = new CurveFittingEdge( x_data[i] );
        edge->setId(i);
        edge->setVertex( 0, v );                // 设置连接的顶点
        edge->setMeasurement( y_data[i] );      // 观测数值  经过高斯噪声的
        //这里的信息矩阵可以参考:http://www.cnblogs.com/gaoxiang12/p/5244828.html 里面有说明
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma) ); // 信息矩阵：协方差矩阵之逆 这里为1表示加权为1
        optimizer.addEdge( edge );
    }

    // 执行优化
    cout<<"start optimization"<<endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(7);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout<<"estimated model: "<<abc_estimate.transpose()<<endl;

    return 0;
}
