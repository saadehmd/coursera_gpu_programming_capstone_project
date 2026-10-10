#pragma once
#include <pcl/PointIndices.h>
#include <pcl/cuda/point_cloud.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_cloud.h>
#include <thrust/device_vector.h>
#include <atomic>
#include <check_cudax.hpp>
#include <cuda/std/span>
#include <pcl/gpu/octree/octree.hpp>
#include <tuple>

// #include <cuda-samples/Common/helper_cuda.h>

class CudaEuclideanClustering
{
  public:
    CudaEuclideanClustering(float cluster_tolerance, int min_cluster_size, int max_cluster_size, float max_search_radius,
                            int max_neighbors, std::tuple<float, float, float> voxel_size)
        : cluster_tolerance_(cluster_tolerance),
          min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size),
          voxel_size_(voxel_size),
          device_cloud_(pcl::make_shared<pcl::gpu::Octree>())
    {
        cudaDeviceProp device_prop;
        // gpuDeviceInit(0);
        cudaGetDeviceProperties(&device_prop, 0);
        threads_per_block_ = device_prop.maxThreadsPerBlock;
    }

    void setInputHostCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        host_cloud_ = cloud;
        device_cloud_arr_.create(cloud->points.size());
        device_cloud_arr_.upload(cloud->data(), cloud->size());
        device_cloud_->setCloud(device_cloud_arr_);
        device_cloud_->build();
    }
    void setInputDeviceCloud(const pcl::gpu::DeviceArray<pcl::PointXYZ>& cloud)
    {
        device_cloud_->setCloud(cloud);
        device_cloud_->build();
    }

    /** Custom implementation of GPU-based euclidean clustering of pointclouds */
    void extract(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig);

    /** Batched-frontier GPU implementation of euclidean clustering. */
    void extractV2(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig);

    /** All-pairs GPU union-find implementation of euclidean clustering. */
    void extractV3(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig);

    /** PCL's implementation of GPU-based euclidean clustering of pointclouds  */
    void extractPclGpuImpl(std::vector<pcl::PointIndices>& clusters);

    /** PCL's implementation of CPU-based euclidean clustering of pointclouds  */
    void extractPclCpuImpl(std::vector<pcl::PointIndices>& clusters);

    /** Creates a grid of translated copies of the input point cluster.
     * @param num_clusters Total number of clusters in the result, including the original.
     * @param min_cluster_distance Minimum gap between the enclosing spheres of every pair of clusters.
    */
    void gridReplicate(size_t num_clusters, double min_cluster_distance);

  private:
    //__device__ void euclideanDistance(const pcl::PointXYZ& ptA, const pcl::PointXYZ& ptB, double& distance);
    void launchKnnKernel(const thrust::device_vector<int>& potential_neighbors_idx, const pcl::PointXYZ& query_point,
                         thrust::device_vector<double>& distances);

  private:
    float cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    std::tuple<float, float, float> voxel_size_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr host_cloud_;
    pcl::gpu::Octree::Ptr device_cloud_;
    pcl::gpu::DeviceArray<pcl::PointXYZ> device_cloud_arr_;
    size_t threads_per_block_ = 256;
};
