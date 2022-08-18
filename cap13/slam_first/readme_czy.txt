1. 首先是工程的运行
 在工程主目录下，也只能在工程主目录下，执行命令：
	./bin/run_kitti_stereo

2. 如果要使用其他的数据集，只需要修改：
	./config/default.yaml里面的：
		dataset_dir 值就好
3. 为了以后更好的学习，这里我就简单说下，在运行的时候所属出的更信息的含义：
	3.1 VO is running 
		出自visual_odometry.cpp的VisualOdometry::Run()
		意思是：视觉里程计正在运行，此时一直读取数据，处理数据
	3.2 Find 197 in the last image.
		出自frontend.cpp的Frontend::TrackLastFrame()
		意思是：当前帧使用LK追踪到上一帧左图图像的特征点数目
	3.3 Outlier/Inlier in pose estimating: 21/139
		出自frontend.cpp的Frontend::EstimateCurrentPose()
		意思是：当前帧的位姿经过优化之后，外点（偏差较大的点）和内点（偏差较小的点）前端g2o是优化单个位姿，
		        边是世界坐标系下的3D坐标通过该位姿变换到当前帧下，然后变换为2D像素坐标（估计值）与该点本身
			2D坐标（真值）误差
	3.4 Current Pose = 
			0-0.322732 -0.0771316 0-0.943342 000131.939
			000.228878 000.960735 0-0.156856 000024.307
			00000.9184 0-0.266533 0-0.292406 00-108.109
			0000000000 0000000000 0000000000 0000000001
		出自frontend.cpp的Frontend::EstimateCurrentPose()
		意思是：当前帧位姿的SE3
	3.5 Outlier/Inlier in optimization: 0/2108
		出自backend.cpp的Backend::Optimize(Map::KeyframesType &keyframes,Map::LandmarksType &landmarks)
		意思是：后端优化之后，外点（偏差较大的点）和内点（偏差较小的点）
	3.6 VO cost time: 0.0215227 seconds.

		出自visual_odometry.cpp的VisualOdometry::Step()
		意思是：视觉里程计处理一帧所花费的时间
	3.7 remove keyframe 52
		出自frontend.cpp的Map::RemoveOldKeyframe()
		意思是：移除地图类中keyframe_id_为52的关键帧
	3.8 Removed 121 active landmarks
		出自map.cpp的Map::CleanMap()
		意思是：移除地图类中121个路标点
	3.9 Set frame 326 as keyframe 59
		出自frontend.cpp的Frontend::InsertKeyframe()
		意思是：设置id为326的帧为关键帧59
	3.10 Detect 133 new features
		出自frontend.cpp的Frontend::DetectFeatures()
		意思是：在当前帧中左图中检测到133个新的特征点
	3.11 Find 230 in the right image.
		出自frontend.cpp的Frontend::FindFeaturesInRight()
		意思是：在当前帧右图中使用LK光流追踪到230个特征点
	3.12 new landmarks: 122
		出自frontend.cpp的Frontend::TriangulateNewPoints()
		意思是：使用三角测量得到122个新的路标点
