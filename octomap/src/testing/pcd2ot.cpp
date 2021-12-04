#include <octomap/octomap.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>

#define resolution 0.25  // unit: m

int main(int argc, char** argv) {
  std::string octomap_name = "octomap.ot";

  // Part1: Read origin point cloud
  Eigen::Vector4f centroid;
  Eigen::Matrix3f covariance_matrix;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(
      new pcl::PointCloud<pcl::PointXYZ>);

  std::cout << " --------------- "
            << " initial         "
            << " --------------- " << std::endl;

  if (argc == 2) {
    std::cout << "use input data" << std::endl;

    if (pcl::io::loadPCDFile(argv[1], *cloud_in)) {
      std::cerr << "failed to open " << argv[1] << std::endl;
      return 1;
    }

    pcl::computeMeanAndCovarianceMatrix(*cloud_in, covariance_matrix, centroid);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3) = -centroid(0);
    transform(1, 3) = -centroid(1);
    transform(2, 3) = -centroid(2);

    pcl::transformPointCloud(*cloud_in, *cloud_in, transform);
  }

  std::cout << "Input " << cloud_in->size() << " pts. " << std::endl;

  // Part2: Construct Gaussion Octomap
  std::cout << " --------------- "
            << " compute tree    "
            << " --------------- " << std::endl;

  octomap::OcTree save_tree(resolution);
  std::unordered_multimap<octomap::OcTreeKey, pcl::PointXYZ,
                          octomap::OcTreeKey::KeyHash>
      unorderedMultiMap;
  std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> set;

  for (auto p : (*cloud_in).points) {
    auto key = save_tree.coordToKey(p.x, p.y, p.z, 16);
    unorderedMultiMap.emplace(key, p);
    set.emplace(key);
  }

  for (auto iter = set.begin(); iter != set.end(); ++iter) {
    auto key = *iter;
    auto range = unorderedMultiMap.equal_range(key);
    int i = 0;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (auto it = range.first; it != range.second; ++it) {
      cloud.push_back(it->second);
      i++;
    }

    if (cloud.size() < 2) continue;

    save_tree.updateNode(key, true);
  }

  save_tree.updateInnerOccupancy();

  // Part3: Octomap
  save_tree.write(octomap_name);
}