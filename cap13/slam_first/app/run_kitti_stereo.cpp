//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path");

int main(int argc, char **argv) {

    // 解析gflags参数，只需要1行代码
    google::ParseCommandLineFlags(&argc, &argv, true);
    //gflags::ParseCommandLineFlags(&argc, &argv, true);
    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_config_file));
    //assert(vo->Init() == true);//注释去这句话my_czy
    vo->Init();
    vo->Run();

    return 0;
}
