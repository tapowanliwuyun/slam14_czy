#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// path to trajectory file
string trajectory_file = "../src/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {

    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
    ifstream fin(trajectory_file);
    if (!fin) {
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }

    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz));
        poses.push_back(Twr);
    }
    cout << "read total " << poses.size() << " pose entries" << endl;

    // draw trajectory in pangolin
    DrawTrajectory(poses);
    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
    // create pangolin window and plot the trajectory
    /*
     * 接下来，我们使用CreateWindowAndBind命令创建了一个视窗对象，
     * 函数的入口的参数依次为视窗的名称、宽度和高度，
     * 该命令类似于OpenCV中的namedWindow，即创建一个用于显示的窗体。
     */
    // 创建名称为“Trajectory Viewer”的GUI窗口，尺寸为640×640
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    //启动深度测试。同时，我们启动了深度测试功能，
    // 该功能会使得pangolin只会绘制朝向镜头的那一面像素点，避免容易混淆的透视关系出现，因此在任何3D可视化中都应该开启该功能。
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 创建一个观察相机视图
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    // 创建一个观察相机视图
    // ProjectMatrix(int w, int h, int fu, int fv, int cu, int cv, int znear, int zfar)
    //      参数依次为观察相机的图像宽度、高度、4个内参以及最近和最远视距
    // ModelViewLookAt(double x, double y, double z,double lx, double ly, double lz, AxisDirection Up)
    //      参数依次为相机所在的位置，以及相机所看的视点位置(一般会设置在原点)

    //在完成视窗的创建后，我们需要在视窗中“放置”一个摄像机（注意这里的摄像机是用于观察的摄像机而非SLAM中的相机），
    // 我们需要给出摄像机的内参矩阵ProjectionMatrix从而在我们对摄像机进行交互操作时，
    // Pangolin会自动根据内参矩阵完成对应的透视变换。
    // 此外，我们还需要给出摄像机初始时刻所处的位置，摄像机的视点位置（即摄像机的光轴朝向哪一个点）以及摄像机的本身哪一轴朝上。

    // 创建交互视图
    //pangolin::Handler3D handler(s_cam); //交互相机视图句柄
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));//.SetHandler(&handler);
    //接下来我们需要创建一个交互式视图（view）用于显示上一步摄像机所“拍摄”到的内容，这一步类似于OpenGL中的viewport处理。
    // setBounds()函数前四个参数依次表示视图在视窗中的范围（下、上、左、右），可以采用相对坐标（0~1）以及绝对坐标（使用Attach对象）。


    while (pangolin::ShouldQuit() == false) {
        // 清空颜色和深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);//激活
        //在完成了上述所有准备工作之后，我们就可以开始绘制我们需要的图形了，
        // 首先我们使用glclear命令分别清空色彩缓存和深度缓存
        // 并激活之前设定好的视窗对象（否则视窗内会保留上一帧的图形，这种“多重曝光”效果通常并不是我们需要的）


        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//设定背景的颜色，此处为白色

        glLineWidth(2);//定义其中线条的宽度
        for (size_t i = 0; i < poses.size(); i++) {
            // 画每个位姿的三个坐标轴
            Vector3d Ow = poses[i].translation();
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }
        // 画出连线
        for (size_t i = 0; i < poses.size(); i++) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        //在绘制完成后，需要使用FinishFrame命令刷新视窗。
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
