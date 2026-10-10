#include "cuda_euclidean_clustering.hpp"

#include <pcl/io/pcd_io.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
constexpr float cluster_tolerance = 0.02f;
const std::vector<std::size_t> cluster_counts{5, 10, 50, 100, 200, 500, 1000, 2000};

std::int64_t elapsedMilliseconds(const std::chrono::steady_clock::time_point& start,
                                 const std::chrono::steady_clock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}
} // namespace

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr base_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/workspace/data/bunny.pcd", *base_cloud) == -1)
    {
        PCL_ERROR("Couldn't read input PCD file\n");
        return 1;
    }

    const std::uint64_t base_point_count = base_cloud->size();
    const std::atomic<bool> interrupted{false};
    std::cout << "total_cloud_points,cpu_clustering_ms,gpu_clustering_ms\n";

    for (const std::size_t cluster_count : cluster_counts)
    {
        std::cerr << "cluster_count: " << cluster_count << "\n";
        const std::uint64_t point_count = base_point_count * cluster_count;
        auto clustering = CudaEuclideanClustering(cluster_tolerance, 10, 10000, 1.0f, 500,
                                                   std::make_tuple(0.02f, 0.02f, 0.02f));
        clustering.setInputHostCloud(base_cloud);
        clustering.gridReplicate(cluster_count, cluster_tolerance * 1.01f);

        std::vector<pcl::PointIndices> cpu_clusters;
        const auto cpu_start = std::chrono::steady_clock::now();
        clustering.extractPclCpuImpl(cpu_clusters);
        const auto cpu_end = std::chrono::steady_clock::now();

        std::cout << point_count << ',' << elapsedMilliseconds(cpu_start, cpu_end) << ',';

        const std::uint64_t pair_count = point_count * (point_count - 1) / 2;

        std::vector<pcl::PointIndices> v3_clusters;
        const auto v3_start = std::chrono::steady_clock::now();
        clustering.extractV3(v3_clusters, interrupted);
        const auto v3_end = std::chrono::steady_clock::now();
        std::cout << elapsedMilliseconds(v3_start, v3_end) << '\n';
    }
}