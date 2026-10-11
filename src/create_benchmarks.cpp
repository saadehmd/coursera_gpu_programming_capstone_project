#include "cuda_euclidean_clustering.hpp"

#include <pcl/io/pcd_io.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
constexpr float cluster_tolerance = 0.02f;
const std::vector<std::size_t> cluster_counts{5, 10, 50, 100, 200, 500, 1000, 2000};

std::int64_t elapsedNanoseconds(const std::chrono::steady_clock::time_point& start,
                                const std::chrono::steady_clock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double nanosecondsToMilliseconds(std::int64_t nanoseconds)
{
    return static_cast<double>(nanoseconds) / 1'000'000.0;
}

bool matchesReplicatedCloudExpectation(const std::vector<pcl::PointIndices>& clusters,
                                       std::size_t expected_cluster_count,
                                       std::size_t expected_points_per_cluster,
                                       std::string_view implementation_name)
{
    if (clusters.size() != expected_cluster_count)
    {
        std::cerr << implementation_name << ": expected " << expected_cluster_count
                  << " clusters, but found " << clusters.size() << ".\n";
        return false;
    }

    for (const pcl::PointIndices& cluster : clusters)
    {
        if (cluster.indices.size() != expected_points_per_cluster)
        {
            std::cerr << implementation_name << ": expected " << expected_points_per_cluster
                      << " points per cluster, but found " << cluster.indices.size() << ".\n";
            return false;
        }
    }

    return true;
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
    bool all_results_match_expectation = true;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "total_cloud_points,cpu_clustering_ms,gpu_clustering_ms\n";

    for (const std::size_t cluster_count : cluster_counts)
    {
        std::cerr << "cluster_count: " << cluster_count << "\n";
        const std::uint64_t point_count = base_point_count * cluster_count;
        auto clustering = CudaEuclideanClustering(cluster_tolerance, 10, 10000, 1.0f, 500);
        clustering.setInputHostCloud(base_cloud);
        clustering.gridReplicate(cluster_count, cluster_tolerance * 1.01f);

        std::vector<pcl::PointIndices> cpu_clusters;
        const auto cpu_start = std::chrono::steady_clock::now();
        clustering.extractPclCpuImpl(cpu_clusters);
        const auto cpu_end = std::chrono::steady_clock::now();
        all_results_match_expectation =
            matchesReplicatedCloudExpectation(cpu_clusters, cluster_count, base_point_count, "PCL CPU") &&
            all_results_match_expectation;

        std::cout << point_count << ','
              << nanosecondsToMilliseconds(elapsedNanoseconds(cpu_start, cpu_end)) << ',';

        std::vector<pcl::PointIndices> gpu_clusters;
        const auto gpu_start = std::chrono::steady_clock::now();
        clustering.extractV2(gpu_clusters, interrupted);
        const auto gpu_end = std::chrono::steady_clock::now();
        all_results_match_expectation =
            matchesReplicatedCloudExpectation(gpu_clusters, cluster_count, base_point_count, "V2 GPU") &&
            all_results_match_expectation;
        std::cout << nanosecondsToMilliseconds(elapsedNanoseconds(gpu_start, gpu_end)) << '\n';
    }

    return all_results_match_expectation ? 0 : 1;
}