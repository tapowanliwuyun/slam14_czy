#include "DBoW3/DBoW3.h"//词袋支持头文件
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/highgui/highgui.hpp>//gui模块
#include <opencv2/features2d/features2d.hpp>//特征点头文件
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    String directoryPath = "/home/bupo/my_study/slam14/slam14_my/cap11/rgbd_dataset_freiburg1_desk2/rgb";//图像路径
    vector<String> imagesPath;
    cv::glob(directoryPath, imagesPath);
    // string dataset_dir = argv[1];
    // ifstream fin ( dataset_dir+"/home/liqiang/slambook2/ch11/rgbd_dataset_freiburg1_desk2/rgb" );
    // if ( !fin )
    // {
    //     cout<<"please generate the associate file called associate.txt!"<<endl;
    //     return 1;
    // }

    // vector<string> rgb_files, depth_files;
    // vector<double> rgb_times, depth_times;
    // while ( !fin.eof() )
    // {
    //     string rgb_time, rgb_file, depth_time, depth_file;
    //     fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
    //     rgb_times.push_back ( atof ( rgb_time.c_str() ) );
    //     depth_times.push_back ( atof ( depth_time.c_str() ) );
    //     rgb_files.push_back ( dataset_dir+"/"+rgb_file );
    //     depth_files.push_back ( dataset_dir+"/"+depth_file );

    //     if ( fin.good() == false )
    //         break;
    // }
    // fin.close();

    cout<<"generating features ... "<<endl;//输出generating features (正在检测ORB特征)...
    vector<Mat> descriptors;//描述子
    Ptr< Feature2D > detector = ORB::create();
    int index = 1;
    for ( String path : imagesPath )
    {
        Mat image = imread(path);
        vector<KeyPoint> keypoints; //关键点
        Mat descriptor;//描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
        cout<<"extracting features from image " << index++ <<endl;//输出extracting features from image(从图像中提取特征)
    }
    cout<<"extract total "<<descriptors.size()*500<<" features."<<endl;

    // create vocabulary
    cout<<"creating vocabulary, please wait ... "<<endl;//输出creating vocabulary, please wait (创建词典，请稍等)...
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "vocab_larger.yml.gz" );//保存词典
    cout<<"done"<<endl;

    return 0;
}