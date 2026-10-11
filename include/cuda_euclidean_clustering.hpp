#pragma once
#include <pcl/PointIndices.h>
#include <pcl/cuda/point_cloud.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_cloud.h>
#include <atomic>
#include <check_cudax.hpp>
#include <cuda/std/span>
#include <pcl/gpu/octree/octree.hpp>

class CudaEuclideanClustering
{
  public:
    /**
     * Creates a clustering instance.
     * `max_search_radius` and `max_neighbors` are retained legacy configuration values and are currently unused.
     */
    CudaEuclideanClustering(float cluster_tolerance, int min_cluster_size, int max_cluster_size, float max_search_radius,
                            int max_neighbors)
        : cluster_tolerance_(cluster_tolerance),
          min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size),
          device_cloud_(pcl::make_shared<pcl::gpu::Octree>())
    {
        cudaDeviceProp device_prop;
        // gpuDeviceInit(0);
        cudaGetDeviceProperties(&device_prop, 0);
        threads_per_block_ = device_prop.maxThreadsPerBlock;
    }

      /** Uploads the host cloud, rebuilds the GPU octree, and configures all extraction methods. */
    void setInputHostCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        host_cloud_ = cloud;
        device_cloud_arr_.create(cloud->points.size());
        device_cloud_arr_.upload(cloud->data(), cloud->size());
        device_cloud_->setCloud(device_cloud_arr_);
        device_cloud_->build();
    }
      /**
       * Sets and builds the PCL GPU octree for a device cloud.
       * This does not populate the host cloud or V2's device buffer; use setInputHostCloud for the public extractors.
       */
    void setInputDeviceCloud(const pcl::gpu::DeviceArray<pcl::PointXYZ>& cloud)
    {
        device_cloud_->setCloud(cloud);
        device_cloud_->build();
    }

    /**
     * Runs the naive host-orchestrated depth-first traversal using PCL GPU octree radius searches.
     * Does not apply the configured minimum and maximum cluster-size filters.
     */
    void extract(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig);

    /**
     * Runs all-pairs GPU union-find clustering with $O(N^2)$ distance tests.
     * Returns only components within the configured minimum and maximum cluster sizes.
     */
    void extractV2(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig);

    /** Calls PCL's hybrid host/device GPU Euclidean clustering implementation. Requires setInputHostCloud. */
    void extractPclGpuImpl(std::vector<pcl::PointIndices>& clusters);

    /** Calls PCL's CPU KD-tree Euclidean clustering implementation. Requires setInputHostCloud. */
    void extractPclCpuImpl(std::vector<pcl::PointIndices>& clusters);

    /**
     * Replaces the configured cloud with a grid of translated copies of the current host cloud.
     * @param num_clusters Total number of clusters in the result, including the original.
     * @param min_cluster_distance Minimum gap between the enclosing spheres of every pair of clusters.
     */
    void gridReplicate(size_t num_clusters, double min_cluster_distance);

  private:
    float cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr host_cloud_;
    pcl::gpu::Octree::Ptr device_cloud_;
    pcl::gpu::DeviceArray<pcl::PointXYZ> device_cloud_arr_;
    size_t threads_per_block_ = 256;
};
