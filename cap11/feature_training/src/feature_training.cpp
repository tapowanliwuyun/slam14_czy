#include "DBoW3/DBoW3.h"//词袋支持头文件
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/highgui/highgui.hpp>//gui模块
#include <opencv2/features2d/features2d.hpp>//特征点头文件
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main( int argc, char** argv )
{
    // read the image
    cout<<"reading images... "<<endl;//输出reading images...
    vector<Mat> images; //图像
    for ( int i=0; i<10; i++ )//遍历读取十张图像
    {
        string path = "./data/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;//输出detecting ORB features(正在检测ORB特征) ...
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;//描述子
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; //关键点
        Mat descriptor;//描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );//检测和计算
        descriptors.push_back( descriptor );
    }

    // create vocabulary (创建字典)
    cout<<"creating vocabulary ... "<<endl;//输出(creating vocabulary ...)创建字典
    DBoW3::Vocabulary vocab;//默认构造函数 k=10,d=5
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;//字典信息
    vocab.save( "vocabulary.yml.gz" );//保存字典压缩包
    cout<<"done"<<endl;//输出done

    return 0;
}