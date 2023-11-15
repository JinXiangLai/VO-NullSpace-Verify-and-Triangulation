//
// Created by hyj on 18-11-11.
// Modified by JinXiangLai
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iostream>
#include <random>
#include <vector>

struct Pose {
  Pose(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t){};
  Eigen::Matrix3d Rwc;
  Eigen::Quaterniond qwc;
  Eigen::Vector3d twc;
};

Eigen::Matrix3d hat(const Eigen::Vector3d a){
    Eigen::Matrix3d res;
    res << 0, -a[2], a[1],
            a[2], 0, -a[0],
            -a[1], a[0], 0;
    return res;

}
int main() {
  int featureNums = 20;
  int poseNums = 10;
  int diem = poseNums * 6 + featureNums * 3;
  double fx = 1.;
  double fy = 1.;
  Eigen::MatrixXd H(diem, diem);
  H.setZero();

  std::vector<Pose> camera_pose;
  double radius = 8;
  for (int n = 0; n < poseNums; ++n) {
    double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
    // 绕 z轴 旋转
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d t = Eigen::Vector3d(
        radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
    camera_pose.push_back(Pose(R, t));
  }

  // 随机数生成三维特征点
  std::default_random_engine generator;
  std::vector<Eigen::Vector3d> points;
  // j个特征点
  // J = poseNums * 6 + j * 3
  for (int j = 0; j < featureNums; ++j) {
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);
    points.push_back(Pw);

    // i个相机pose， 这里同时对地图点和位姿进行了雅可比计算
    // I = 6 * i
    // 10个相机位姿，单目SLAM系统的零空间是7
    for (int i = 0; i < poseNums; ++i) {
      Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
      Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

      double x = Pc.x();
      double y = Pc.y();
      double z = Pc.z();
      double z_2 = z * z;
      Eigen::Matrix<double, 2, 3> jacobian_uv_Pc;
      jacobian_uv_Pc << fx / z, 0, -x * fx / z_2, 0, fy / z, -y * fy / z_2;
      // 关于地图点的导数
      Eigen::Matrix<double, 2, 3> jacobian_Pj = jacobian_uv_Pc * Rcw;
      // 这是关于位姿的导数？
    //   Eigen::Matrix<double, 2, 6> jacobian_Ti0;
    //   jacobian_Ti0 << -x * y * fx / z_2, (1 + x * x / z_2) * fx, -y / z * fx,
    //       fx / z, 0, -x * fx / z_2, -(1 + y * y / z_2) * fy, x * y / z_2 * fy,
    //       x / z * fy, 0, fy / z, -y * fy / z_2;
    // Eigen::Vector3d delta_t = Pw - camera_pose[i].twc * xy_rand(generator); // 线性化点不一致使得0空间改变
    Eigen::Vector3d delta_t = Pw - camera_pose[i].twc;
    Eigen::Matrix<double, 3, 6> jacobian_Pc_Ti = Eigen::Matrix<double, 3, 6>::Zero();
    jacobian_Pc_Ti.block<3, 3>(0, 0) = -Rcw * hat(delta_t);
    jacobian_Pc_Ti.block<3, 3>(0, 3) = -Rcw;
    Eigen::Matrix<double, 2, 6> jacobian_Ti = jacobian_uv_Pc * jacobian_Pc_Ti;
    // Eigen::Matrix<double, 2, 6> Zero = jacobian_Ti - jacobian_Ti0;
    // if(Zero.isApprox(Eigen::Matrix<double, 2, 6>::Zero())) {
    //     std::cout << "True\n";
    // }
    
      //   T0  T1  T2 ... P0 P1 P2 ...
      //r1
      //r2
      //r3
      // H矩阵需要计算第 I行第J列的组合值，即计算
      // H(I, I), H(I, J), H(J, I), H(J, J)
      H.block<6, 6>(i * 6, i *6) += jacobian_Ti.transpose() * jacobian_Ti;
      H.block<6, 3>(i * 6, j * 3 + 6 * poseNums) += jacobian_Ti.transpose() * jacobian_Pj;
      H.block<3, 6>(j * 3 + 6 * poseNums, i * 6) += jacobian_Pj.transpose() * jacobian_Ti;
      H.block<3, 3>(j * 3 + 6 * poseNums, j * 3 + 6 * poseNums) += jacobian_Pj.transpose() * jacobian_Pj;
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);
  std::cout << "singular num = " << svd.singularValues().rows() << std::endl;
  Eigen::VectorXd s = svd.singularValues().tail(10);
  std::cout << s.transpose() <<std::endl;
  int zero_count = 0;
  for(int i=0; i<s.rows(); ++i){
    if(s[i] < 1e-6){
        zero_count++;
    }
  }
  std::cout << "Null space should be " << 7 << " counted is " << zero_count <<std::endl;
  return 0;
}
