#include "cuda_euclidean_clustering.hpp"

#include <pcl/PointIndices.h>
#include <pcl/io/pcd_io.h>
#include <signal.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <pcl/gpu/octree/device_format.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <span>
#include <sstream>
//#include <cuda/cmath>

template <typename... Args>
void debug(const Args&... args)
{
#ifdef DEBUG
    std::ostringstream stream;
    (stream << ... << args);
    std::cout << stream.str();
#endif
}

__device__ int findRoot(const int* parent, int point_index)
{
    int root = point_index;
    while (parent[root] != root)
    {
        root = parent[root];
    }
    return root;
}

__device__ void uniteRoots(int* parent, int left_point, int right_point)
{
    while (true)
    {
        const int left_root = findRoot(parent, left_point);
        const int right_root = findRoot(parent, right_point);
        if (left_root == right_root)
        {
            return;
        }

        const int lower_root = min(left_root, right_root);
        const int higher_root = max(left_root, right_root);

        // Swap the higher and lower roots if necessary to ensure the lower root becomes the parent of the higher root.
        // What this swapping does: It takes two connected points and makes sure they both have same parent which is a lower-index parent. 
        //if (atomicCAS(parent + higher_root, higher_root, lower_root) == higher_root
        if (atomicCAS(&parent[higher_root], higher_root, lower_root) == higher_root)
        {
            return;
        }
    }
}

__global__ void unionClosePointPairs(const pcl::PointXYZ* points, int point_count, float tolerance_squared,
                                     int* parent)
{
    const int left_index = blockIdx.y * blockDim.y + threadIdx.y;
    const int right_index = blockIdx.x * blockDim.x + threadIdx.x;

    // The following looks for point-pairs only in the upper triangular matrix of an N x N matrix where N is the number of points being clustered.
    if (left_index >= point_count || right_index >= point_count || right_index <= left_index)
    {
        return;
    }

    // Imagining the similar N x N matrix as mentioned above, think of each point pair as an element Mat[left,right] of this matrix,
    //  even though points are actually indexed from a N-long array.
    const pcl::PointXYZ left = points[left_index];
    const pcl::PointXYZ right = points[right_index];
    const float delta_x = left.x - right.x;
    const float delta_y = left.y - right.y;
    const float delta_z = left.z - right.z;
    const float distance_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    if (distance_squared <= tolerance_squared)
    {
        uniteRoots(parent, left_index, right_index);
    }
}

__global__ void compressParents(int* parent, int point_count)
{
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This recursively goes through parents of the point at given index until it reaches the oldest i.e root parent. 
    // When all the thread blocks have finished running this kernel, all points have been assigned their top-level parent.
    // Each parent at this level represents a unique cluster root.
    if (point_index < point_count)
    {
        parent[point_index] = findRoot(parent, point_index);
    }
}

void CudaEuclideanClustering::extract(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig)
{
    if (!device_cloud_ || device_cloud_->cloud_ == nullptr)
    {
        throw std::runtime_error("Input cloud is not set.");
    }

    if (!device_cloud_->isBuilt())
    {
        device_cloud_->build();
    }

    debug("Allocating vector indices for clusters for ", device_cloud_->cloud_->size(), " points\n");
    thrust::device_vector<int> all_pt_indices(device_cloud_->cloud_->size());
    thrust::device_vector<unsigned char> points_visited(device_cloud_->cloud_->size(), 0);
    thrust::device_vector<int> points_to_be_queried;
    thrust::sequence(all_pt_indices.begin(), all_pt_indices.end());

    // Iterate thru points until all points have been visited
    thrust::device_vector<int> current_cluster;
    while(!interrupt_sig.load())
    {
        auto first_unvisited = thrust::find(points_visited.begin(), points_visited.end(), 0);
        if (first_unvisited == points_visited.end())
        {
            break; // All points have been visited
        }
        int first_unvisited_idx = first_unvisited - points_visited.begin();
        points_to_be_queried.push_back(first_unvisited_idx);


        while (!points_to_be_queried.empty() && !interrupt_sig.load())
        {
            int pt_idx = points_to_be_queried.back();
            points_to_be_queried.pop_back();
            points_visited[pt_idx] = 1;

            debug("points_to_be_queried size: ", points_to_be_queried.size(), "\n");
            debug("Searching neighbors for point ", pt_idx, "\n");

            // Find neighbors of this query point
            pcl::PointXYZ query_point;
            device_cloud_arr_.download(&query_point, pt_idx, 1);
            pcl::gpu::Octree::Queries queries(1);
            queries.upload(&query_point, 1);
            const int max_results = static_cast<int>(device_cloud_->cloud_->size());
            pcl::gpu::NeighborIndices neighbor_indices(1, max_results);
            device_cloud_->radiusSearch(queries, cluster_tolerance_, max_results, neighbor_indices);
            std::vector<int> host_sizes;
            neighbor_indices.sizes.download(host_sizes);
            debug("Radius search found ", host_sizes[0], " neighbors\n");

            auto visited_ptr = thrust::raw_pointer_cast(points_visited.data());
            thrust::device_vector<int> new_neighbors(host_sizes[0]);

            // Copy only the neighbors that were not visited before.
            auto new_neighbors_end = thrust::copy_if(neighbor_indices.data.ptr(), neighbor_indices.data.ptr() + host_sizes[0], new_neighbors.begin(), [visited_ptr] __device__ (int idx) { return !visited_ptr[idx]; });
            new_neighbors.resize(new_neighbors_end - new_neighbors.begin());
            points_to_be_queried.insert(points_to_be_queried.end(), new_neighbors.begin(), new_neighbors.end());

            // As soon as we add new neighbors to query queue, we mark them visited.
            thrust::for_each(new_neighbors.begin(), new_neighbors.end(),[visited_ptr] __device__ (int idx) {visited_ptr[idx] = 1;}); 
            current_cluster.push_back(pt_idx);
        }

        // close this cluster
        // Copy point indices back to host memory
        thrust::host_vector<int> host_indices(current_cluster.begin(), current_cluster.end());
        pcl::PointIndices pi;
        pi.indices.assign(host_indices.begin(), host_indices.end());
        clusters.push_back(pi);
        current_cluster.clear();
    }
}

void CudaEuclideanClustering::extractV3(std::vector<pcl::PointIndices>& clusters, const std::atomic<bool>& interrupt_sig)
{
    if (!device_cloud_ || device_cloud_->cloud_ == nullptr)
    {
        throw std::runtime_error("Input cloud is not set.");
    }

    const int point_count = static_cast<int>(device_cloud_->cloud_->size());
    if (point_count == 0 || interrupt_sig.load())
    {
        return;
    }

    thrust::device_vector<int> parent(point_count);
    thrust::sequence(parent.begin(), parent.end());

    constexpr int tile_width = 16;
    const dim3 block(tile_width, tile_width);
    const int grid_dim = (point_count + tile_width - 1) / tile_width;
    const dim3 grid(grid_dim, grid_dim);
    //const dim3 grid((point_count + tile_width - 1) / tile_width, (point_count + tile_width - 1) / tile_width);
    const float tolerance_squared = cluster_tolerance_ * cluster_tolerance_;
    unionClosePointPairs<<<grid, block>>>(device_cloud_arr_.ptr(), point_count, tolerance_squared,
                                          thrust::raw_pointer_cast(parent.data()));

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(cuda_error));
    }

    constexpr int threads_per_block = 256;
    const int blocks_per_grid = (point_count + threads_per_block - 1) / threads_per_block;
    compressParents<<<blocks_per_grid, threads_per_block>>>(thrust::raw_pointer_cast(parent.data()), point_count);
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(cuda_error));
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(cuda_error));
    }

    thrust::device_vector<int> point_indices(point_count);
    thrust::sequence(point_indices.begin(), point_indices.end());

    // Connected points are not necessarily continguous to each other in the array. We sort them by their parent points first before extracting clusters.
    thrust::sort_by_key(parent.begin(), parent.end(), point_indices.begin());

    thrust::host_vector<int> host_roots(parent.begin(), parent.end());
    thrust::host_vector<int> host_indices(point_indices.begin(), point_indices.end());

    // Each unique root is a "key" to it's respective cluster. Iterate through roots and copy their descendant points to their respective clusters.
    for (int cluster_begin = 0; cluster_begin < point_count;)
    {
        const int cluster_root = host_roots[cluster_begin];
        int cluster_end = cluster_begin;
        while (cluster_end < point_count && host_roots[cluster_end] == cluster_root)
        {
            ++cluster_end;
        }

        const int cluster_size = cluster_end - cluster_begin;
        if (cluster_size >= min_cluster_size_ && cluster_size <= max_cluster_size_)
        {
            pcl::PointIndices cluster;
            cluster.indices.assign(host_indices.begin() + cluster_begin, host_indices.begin() + cluster_end);
            clusters.push_back(std::move(cluster));
        }
        cluster_begin = cluster_end;
    }
}

void CudaEuclideanClustering::extractPclGpuImpl(std::vector<pcl::PointIndices>& clusters)
{
    if (!host_cloud_ || !device_cloud_ || device_cloud_->cloud_ == nullptr)
    {
        throw std::runtime_error("A host input cloud must be set before using PCL GPU clustering.");
    }

    // PCL's GPU implementation of euclidean clustering is hybrid so it works on both host and device copies of the same source pointcloud.
    pcl::gpu::extractEuclideanClusters<pcl::PointXYZ>(host_cloud_, device_cloud_, cluster_tolerance_, clusters,
                                                       min_cluster_size_, max_cluster_size_);
}

void CudaEuclideanClustering::extractPclCpuImpl(std::vector<pcl::PointIndices>& clusters)
{
    if (!host_cloud_)
    {
        throw std::runtime_error("A host input cloud must be set before using PCL CPU clustering.");
    }

    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(host_cloud_);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extractor;
    extractor.setClusterTolerance(cluster_tolerance_);
    extractor.setMinClusterSize(min_cluster_size_);
    extractor.setMaxClusterSize(max_cluster_size_);
    extractor.setSearchMethod(tree);
    extractor.setInputCloud(host_cloud_);
    extractor.extract(clusters);
}

void CudaEuclideanClustering::gridReplicate(size_t num_clusters, double min_cluster_distance)
{
    if (!host_cloud_ || host_cloud_->empty())
    {
        throw std::runtime_error("A non-empty host input cloud must be set before replication.");
    }
    if (num_clusters == 0)
    {
        throw std::invalid_argument("The number of clusters must be at least one.");
    }
    if (!std::isfinite(min_cluster_distance) || min_cluster_distance < 0.01)
    {
        throw std::invalid_argument("The minimum cluster distance must be finite and at least 0.01.");
    }

    struct ClusterCenter
    {
        double x;
        double y;
        double z;
    };

    ClusterCenter source_center{0.0, 0.0, 0.0};
    for (const pcl::PointXYZ& point : host_cloud_->points)
    {
        source_center.x += point.x;
        source_center.y += point.y;
        source_center.z += point.z;
    }
    const double point_count = static_cast<double>(host_cloud_->size());
    source_center.x /= point_count;
    source_center.y /= point_count;
    source_center.z /= point_count;

    double enclosing_radius = 0.0;
    for (const pcl::PointXYZ& point : host_cloud_->points)
    {
        const double delta_x = point.x - source_center.x;
        const double delta_y = point.y - source_center.y;
        const double delta_z = point.z - source_center.z;
        enclosing_radius = std::max(enclosing_radius, std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z));
    }

    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = host_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr replicated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    replicated_cloud->points.reserve(source_cloud->size() * num_clusters);
    const double center_spacing = 2.0 * enclosing_radius + min_cluster_distance;
    const size_t grid_width = static_cast<size_t>(std::ceil(std::cbrt(static_cast<double>(num_clusters))));
    for (size_t cluster_index = 0; cluster_index < num_clusters; ++cluster_index)
    {
        const size_t grid_x = cluster_index % grid_width;
        const size_t grid_y = (cluster_index / grid_width) % grid_width;
        const size_t grid_z = cluster_index / (grid_width * grid_width);
        const float translation_x = static_cast<float>(grid_x * center_spacing);
        const float translation_y = static_cast<float>(grid_y * center_spacing);
        const float translation_z = static_cast<float>(grid_z * center_spacing);
        for (const pcl::PointXYZ& point : source_cloud->points)
        {
            replicated_cloud->points.emplace_back(point.x + translation_x, point.y + translation_y, point.z + translation_z);
        }
    }
    replicated_cloud->width = static_cast<std::uint32_t>(replicated_cloud->points.size());
    replicated_cloud->height = 1;
    replicated_cloud->is_dense = source_cloud->is_dense;
    setInputHostCloud(replicated_cloud);
}

static std::atomic<bool> sig_interrupt{false};
static void sigint_handler(int /*sig*/) { sig_interrupt.store(true); }

void benchmarkClusteringImplementations(CudaEuclideanClustering& clustering, const std::atomic<bool>& interrupt_sig)
{
    std::vector<pcl::PointIndices> v3_clusters;
    const auto v3_start = std::chrono::steady_clock::now();
    clustering.extractV3(v3_clusters, interrupt_sig);
    const auto v3_end = std::chrono::steady_clock::now();

    if (interrupt_sig.load())
    {
        std::cout << "V3 clustering interrupted.\n";
        return;
    }

    /*std::vector<pcl::PointIndices> pcl_clusters;
    const auto pcl_start = std::chrono::steady_clock::now();
    clustering.extractPclGpuImpl(pcl_clusters);
    const auto pcl_end = std::chrono::steady_clock::now();*/

    std::vector<pcl::PointIndices> cpu_clusters;
    const auto cpu_start = std::chrono::steady_clock::now();
    clustering.extractPclCpuImpl(cpu_clusters);
    const auto cpu_end = std::chrono::steady_clock::now();

    std::cout << "V3 all-pairs clustering: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(v3_end - v3_start).count()
              << " milliseconds, " << v3_clusters.size() << " clusters.\n";
    /*std::cout << "PCL GPU clustering: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(pcl_end - pcl_start).count()
              << " milliseconds, " << pcl_clusters.size() << " clusters.\n";*/
    std::cout << "PCL CPU clustering: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count()
              << " milliseconds, " << cpu_clusters.size() << " clusters.\n";
}

int main(int argc, char** argv)
{
    signal(SIGINT, sigint_handler);

    const cuda::std::span<char*> args(argv, static_cast<size_t>(argc));

    if (args.size() < 2 || args.size() > 3)
    {
        std::cerr << "Usage: " << args[0] << " <cluster_tolerance> [num_of_clusters]\n";
        return 1;
    }

    const float cluster_tolerance = std::stof(args[1]);
    if (cluster_tolerance <= 0.0f)
    {
        std::cerr << "Cluster tolerance must be a positive finite number.\n";
        return 1;
    }

    size_t num_of_clusters = 1;
    if (args.size() == 3)
    {
        try
        {
            const unsigned long long parsed_num_of_clusters = std::stoull(args[2]);
            if (parsed_num_of_clusters == 0 ||
                parsed_num_of_clusters > std::numeric_limits<size_t>::max())
            {
                throw std::out_of_range("Cluster count is outside the supported range.");
            }
            num_of_clusters = static_cast<size_t>(parsed_num_of_clusters);
        }
        catch (const std::exception&)
        {
            std::cerr << "num_of_clusters must be a positive integer.\n";
            return 1;
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/workspace/data/bunny.pcd", *cloud) == -1)  //* load the file
    {
        PCL_ERROR("Couldn't read file input pcd file \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height << " data points from input pcd file.\n";
    std::cout << "Using cluster tolerance: " << cluster_tolerance << "\n";

    auto euclidean_clustering = CudaEuclideanClustering(cluster_tolerance, 10, 10000, 1.0, 500,
                                                        std::make_tuple(0.02f, 0.02f, 0.02f));
    euclidean_clustering.setInputHostCloud(cloud);
    if (num_of_clusters > 1)
    {
        euclidean_clustering.gridReplicate(num_of_clusters, cluster_tolerance);
        std::cout << "Replicated input into " << num_of_clusters << " clusters.\n";
    }
    benchmarkClusteringImplementations(euclidean_clustering, sig_interrupt);
}