#include <iostream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

int main(int argc,char ** argv)
{
    cv::Mat img_1,img_2;
    if(argc != 3)
    {
        img_1 = cv::imread("../src/1.png",CV_LOAD_IMAGE_COLOR);
        img_2 = cv::imread("../src/2.png",CV_LOAD_IMAGE_COLOR);
        // 判断图像文件是否正确读取
        if (img_1.data == nullptr || img_2.data == nullptr) { //数据不存在,可能是文件不存在
            cout << "usage: feature_extraction ima1 img2" << endl;
            return 0;
        }
    }
    else
    {
        //读取图像
        img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
        assert(img_1.data != nullptr && img_2.data != nullptr);
    }

    //初始化
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //第一步：检测Oriented FAST角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1,keypoints_1,descriptors_1);
    descriptor->compute(img_2,keypoints_2,descriptors_2);

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds." << endl;

    cv::Mat outimg1;
    cv::drawKeypoints(img_1,keypoints_1,outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg1);
    cv::waitKey(0);//

    if(cv::imwrite("../src/ORB_features.png",outimg1) == false)
    {
        cout << "Failed to save the image" << endl;
    }


    //--第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<cv::DMatch> matches;  //cv::DMatch    用于匹配关键点描述子的类
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1,descriptors_2,matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "match ORB cost = " << time_used.count() << " seconds." << endl;


    //--第四步：匹配点对筛选
    //计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(),matches.end(),
                                  [](const cv::DMatch &m1, const cv::DMatch &m2){return m1.distance<m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    cout << "--Min dist : " << min_dist << endl;
    cout << "--Max dist : " << max_dist << endl;

    //当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时最小距离会非常小，所以要设置一个经验值30作为下限
    std::vector<cv::DMatch> good_matches;
    for(int i = 0; i < descriptors_1.rows;i++)
    {
        if(matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    //--第五步：绘制匹配结果
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img_1, keypoints_1, img_2,keypoints_2,matches,img_match);
    cv::drawMatches(img_1, keypoints_1, img_2,keypoints_2,good_matches,img_goodmatch);

    if(cv::imwrite("../src/all_matches.png",img_match) == false)
    {
        cout << "Failed to save the image" << endl;
    }
    if(cv::imwrite("../src/good_matches.png",img_goodmatch) == false)
    {
        cout << "Failed to save the image" << endl;
    }

    cv::imshow("all matches", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);
    return 0;
}

