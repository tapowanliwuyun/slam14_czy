//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {
// 7
Frontend::Frontend() {
    //构造GFTT特征点提取方式
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();//初始化双目
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();//跟踪
            break;
        case FrontendStatus::LOST:
            Reset();//丢失重置
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::Track() {
    if (last_frame_) {//如果存在上一帧
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose()); // 括号里参数为：Tcc_w = Tcc_lc * Tlcw
    }

    int num_track_last = TrackLastFrame();//使用LK光流估计上一帧左图特征点在当前帧的像素坐标位置，然后返回追踪成功的点的数目
    tracking_inliers_ = EstimateCurrentPose();//使用g2o计算当前帧的位姿，返回的得到的内点的数目

    if (tracking_inliers_ > num_features_tracking_) {//如果内点的数目大于所要求的内点的要求数目，设定此次追踪是好的
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {//如果内点的数目大于所要求的内点的最小数目，设定此次追踪是不好的
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {//否则，判断此次追踪lost
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();

    //relative_motion_：当前帧与上一帧的相对运动，用于估计下一个当前帧pose初值
    // Tcc_lc = Tcc_w * Tw_lc
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {//当前帧位姿优化后的内点（有效点），小于所设定的阈值则不更新
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();//设定当前帧为关键帧并分配关键帧id
    map_->InsertKeyFrame(current_frame_);//将当前帧插入到地图类的关键帧集合中

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame(); // 将当前帧的左图特征点 设定给其对应的地图点作为观测的到该地图点的观测点
    DetectFeatures();  // detect new features 检测新的特征 // 注意：
                                                         // 这里前面已经使用LK光流法追踪过上一帧左图中的特征点，节省检测时间，
                                                         // 此处的检测就会避免掉已经追踪成功的特征点的区域，检测是为了在新的一帧中会出现新的区域

    // track in right image
    FindFeaturesInRight();//使用LK流估计右图像中的特征点
    // triangulate map points
    TriangulateNewPoints(); //使用当前帧的左右图对当前帧中的二维点进行三角测量
    // update backend because we have a new keyframe
    backend_->UpdateMap(); //由于有了新的关键帧，所以更新后端，触发地图更新，启动优化

    if (viewer_) viewer_->UpdateMap();//更新地图，触发地图更新

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {//获取当前帧左图中各特征点
        auto mp = feat->map_point_.lock();//获取该特征点所关联的路标点
        if (mp) mp->AddObservation(feat);//
    }
}
    //使用当前帧的左右图对当前帧中的二维点进行三角测量
int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};//获取相机的位姿
    SE3 current_pose_Twc = current_frame_->Pose().inverse();//获取当前帧位姿
    int cnt_triangulated_pts = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {//获取当前帧左图中各特征点
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
            // 保证左图的特征点未关联地图点且存在右图匹配点，尝试三角化
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Vec3 pworld = Vec3::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0) {//三角测量得到该点在相机坐标下的3D坐标
                auto new_map_point = MapPoint::CreateNewMappoint();//创建新的路标点
                pworld = current_pose_Twc * pworld;// Twp = Twc * Tcp
                new_map_point->SetPos(pworld);//设定新的路标点在世界坐标系下的坐标位置
                new_map_point->AddObservation(
                    current_frame_->features_left_[i]);//将该特征点对（左图中）作为路标点的观测点
                new_map_point->AddObservation(
                    current_frame_->features_right_[i]);

                current_frame_->features_left_[i]->map_point_ = new_map_point;//为该特征点对设置所关联的路标点
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                map_->InsertMapPoint(new_map_point);//为地图类插入新的路标点
                cnt_triangulated_pts++;
            }
        }
    }
    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {//仅仅优化当前帧的位姿，这里不优化各路标点
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());//Tcw 当前帧的位姿的估计值
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {//取出当前帧左图中的各特征点
        auto mp = current_frame_->features_left_[i]->map_point_.lock();//取出该特征点所关联的地图点
        if (mp) {
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(
                toVec2(current_frame_->features_left_[i]->position_.pt));//测量值是该特征点的2D像素坐标
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);//Huber核函数
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers 估计姿态并确定异常值
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());//设定当前帧位姿的估计值
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];//获取每一个优化边
            if (features[i]->is_outlier_) {//没看懂这一步的意义
                e->computeError();
            }
            if (e->chi2() > chi2_th) {//如果该边的误差大于阈值，则对应的特征点设定为外点，且变得层数设置为1
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            } else {//如果该边的误差小于等于阈值，则对应的特征点设定为内点，且变得层数设置为0
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            };

            if (iteration == 2) {//当迭代两次之后，不再使用核函数
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate()); //为当前帧设定优化后的位姿

    LOG(INFO) << "Current Pose SE3 = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {//取出当前帧左图中各和路标点有关联的特征点
        if (feat->is_outlier_) {//如果该特征点是外点
            feat->map_point_.reset();//将其与路标点断开连接
            feat->is_outlier_ = false;  // maybe we can still use it in future 将其设置为内点
        }
    }
    return features.size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    for (auto &kp : last_frame_->features_left_) {//获取上一帧的特征点
        if (kp->map_point_.lock()) {//获取该特征点所对应的路标点
            // use project point
            auto mp = kp->map_point_.lock();
            auto px =
                camera_left_->world2pixel(mp->pos_, current_frame_->Pose());//获取当前该路标点在左相机下的像素坐标的估计值
            kps_last.push_back(kp->position_.pt);//获取当前特征点在上一帧中的像素坐标
            kps_current.push_back(cv::Point2f(px[0], px[1]));//估计当前特征点在当前帧中的像素坐标
        } else {
            kps_last.push_back(kp->position_.pt);
            kps_current.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);//使用LK光流追踪法计算上一帧中特征点在当前帧中的像素坐标

    int num_good_pts = 0;//累计追踪成功的点

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {//判断是否追踪成功
            cv::KeyPoint kp(kps_current[i], 7);//创建关键点，2d像素坐标
            Feature::Ptr feature(new Feature(current_frame_, kp));//创建新的特征点
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;//该特征点所关联的地图点= 上一帧左图特征点所关联的特征点
            current_frame_->features_left_.push_back(feature);//将该特征点加入到当前帧的左图特征点中
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    if (num_coor_features < num_features_init_) {
        return false;
    }

    bool build_map_success = BuildInitMap();//初始化地图点
    if (build_map_success) {//如果初始化成功，则改变前端类状态
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {//如果可视化类非空
            viewer_->AddCurrentFrame(current_frame_);//增加一个当前帧
            viewer_->UpdateMap();//以当前帧更新可视化类
        }
        return true;
    }
    return false;
}

int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }
    ///在已经存在特征点的区域不进行特征点提取
    std::vector<cv::KeyPoint> keypoints;
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_left_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    // 使用LK流估计右图像中的特征点
    std::vector<cv::Point2f> kps_left, kps_right;
    for (auto &kp : current_frame_->features_left_) {
        kps_left.push_back(kp->position_.pt);//将当前帧的作图特征点放入容器
        auto mp = kp->map_point_.lock();//获取该特征点对应的地图点
        if (mp) {
            // use projected points as initial guess
            // 使用投影点作为初始猜测
            auto px =
                camera_right_->world2pixel(mp->pos_, current_frame_->Pose());//地图点在右图中像素坐标的估计
            kps_right.push_back(cv::Point2f(px[0], px[1]));
        } else {
            // use same pixel in left iamge
            // 在左图像中使用相同的像素
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);//LK光流追踪

    int num_good_pts = 0;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {//如果追踪成功
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));//创建新的特征点，以及其所在的帧
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};//获取相机的位姿，相机在世界坐标系下的位姿
    size_t cnt_init_landmarks = 0;
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr) continue;//如果右图的特征点没有LK光流中追踪到左图中的点，会被赋予空指针，没办法使用三角测量求深度
        // create map point from triangulation
        std::vector<Vec3> points{ //两个3D点的vector，第一个是左图中特征点的2D坐标，第二个是右图中特征点的2D坐标
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {//确保深度是正确的值，这个三角测量方法没看东，之后再来看
            auto new_map_point = MapPoint::CreateNewMappoint();//创建一个新的地图点
            new_map_point->SetPos(pworld);//为新的地图点设定其在世界坐标系中的坐标
            new_map_point->AddObservation(current_frame_->features_left_[i]);//设定当前帧左图中与当前地图点对应的特征点
            new_map_point->AddObservation(current_frame_->features_right_[i]);//设定当前帧右图中与当前地图点对应的特征点
            current_frame_->features_left_[i]->map_point_ = new_map_point;//并为各特征点设置所关联的地图点
            current_frame_->features_right_[i]->map_point_ = new_map_point;//并为各特征点设置所关联的地图点
            cnt_init_landmarks++;//累计初始的地图点数目
            map_->InsertMapPoint(new_map_point);//地图类中添加新的地图点
        }
    }
    current_frame_->SetKeyFrame();//把当前帧设定为关键帧并分配关键帧id
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();//后端循环进行一次 更新后端，触发地图更新，启动优化

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented.未实现重置 ";
    return true;
}

}  // namespace myslam