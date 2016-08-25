#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/surface/mls.h>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <sstream>
#include <math.h>

namespace fs = ::boost::filesystem;

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointXYZINormal PointTypeFull;

std::string model_filename_;
std::string output_filename_ = "model.pcd";
std::string foldername;

// Algorithm parameters
float plane_thresh_(0.01f);
float cluster_tolerance_(0.02f); // cm
int min_cluster_size_(100);
int max_cluster_size_(250000);
int mean_k_(50);
float std_dev_mul_(0.3f);
bool headless(false);
bool center_cloud_(false);
float sigma_s(0.0f);
float sigma_r(1.0f);
bool folder_mode(false);
float mls_radius_(2.0f);
int max_n(5);


double resolution;

void showHelp(char *filename)
{
	std::cout << std::endl;
	std::cout << "***************************************************************************" << std::endl;
    std::cout << "*                                                                         *" << std::endl;
    std::cout << "*                      Point Cloud Model Extraction                       *" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "***************************************************************************" << std::endl << std::endl;
	std::cout << "Usage: " << filename << " (--folder foldername)|input_filename.pcd [output_filename.pcd]" << std::endl << std::endl;
	std::cout << "     --sigma_s val:            Standard Deviation of spatial Gaussian used by bilateral filter (default 0.0 - deactivated)" << std::endl;
	std::cout << "     --sigma_r val:            Standard Deviation of intensity Gaussian used by bilateral filter (default 1.0)" << std::endl;
	std::cout << "     --plane_thresh val:       RANSAC distance threshold for plane removal (default 0.01)" << std::endl;
	std::cout << "     --cluster_tolerance val:  Euclidean clustering tolerance (default 0.02)" << std::endl;
	std::cout << "     --min_cluster_size val:   Euclidean clustering minimum number of points (default 100)" << std::endl;
	std::cout << "     --max_cluster_size val:   Euclidean clustering maximum number of points (default 250000)" << std::endl;
	std::cout << "     --mls_radius val:         Moving Least Squares (MLS) sampling radius for cloud smoothing. Deactivated if radius <= 1. (default 2.0)" << std::endl;
	std::cout << "     --mean_k val:             Mean number of nearest neighbors for Statistical Outlier Removal (default 50)" << std::endl;
	std::cout << "     --std_dev_mul val:        Standard deviation multiplier threshold for Statistical Outlier Removal (default 0.3)" << std::endl;
	std::cout << "     --headless:               Do not display visualization and close program (default false)" << std::endl;
	std::cout << "     --center_cloud:           Translate output pointcloud such that its centroid is at the 3d origin (default false)" << std::endl << std::endl;
    std::cout << "     --max_n:                  Maximum number of clusters to extract (default 5)" << std::endl << std::endl;
}

void parseCommandLine(int argc, char *argv[])
{
    // Show help
	if (pcl::console::find_switch(argc, argv, "-h"))
	{
		showHelp(argv[0]);
		exit(0);
	}

	if (pcl::console::find_switch(argc, argv, "--headless"))
	{
		headless = true;
	}

	if (pcl::console::find_switch(argc, argv, "--center_cloud"))
	{
		center_cloud_ = true;
	}

	folder_mode = pcl::console::find_switch(argc, argv, "--folder");

	// Model & scene filenames
	std::vector<int> filenames;
	filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
	if (!folder_mode)
	{
		if (filenames.size() < 1)
		{
			std::cout << "Filename missing.\n";
			showHelp(argv[0]);
			exit(-1);
		}

		model_filename_ = argv[filenames[0]];
		if (filenames.size() >= 2)
		{
			output_filename_ = argv[filenames[1]];
		}	
	}
	else if (filenames.size() == 1)
	{
		output_filename_ = argv[filenames[0]];
	}

	// General parameter
	pcl::console::parse_argument(argc, argv, "--folder", foldername);
	pcl::console::parse_argument(argc, argv, "--plane_thresh", plane_thresh_);
	pcl::console::parse_argument(argc, argv, "--cluster_tolerance", cluster_tolerance_);
	pcl::console::parse_argument(argc, argv, "--min_cluster_size", min_cluster_size_);
	pcl::console::parse_argument(argc, argv, "--max_cluster_size", max_cluster_size_);
	pcl::console::parse_argument(argc, argv, "--mean_k", mean_k_);
	pcl::console::parse_argument(argc, argv, "--std_dev_mul", std_dev_mul_);
	pcl::console::parse_argument(argc, argv, "--sigma_s", sigma_s);
	pcl::console::parse_argument(argc, argv, "--sigma_r", sigma_r);
}

float mean_size(std::vector<pcl::PointIndices> &clusters)
{
	int sum = 0;
	for (int i = 0; i < clusters.size(); i++)
	{
		sum += clusters[i].indices.size();
	}
	return sum * 1.0f / clusters.size();
}

struct Color
{
	double r;
	double g;
	double b;
};

float hue2rgb(float p, float q, float t)
{
	if (t < 0.0f) t += 1.0f;
	if (t > 1.0f) t -= 1.0f;
	if (t < 1.0f/6.0f) return p + (q - p) * 6.0f * t;
	if (t < 1.0f/2.0f) return q;
	if (t < 2.0f/3.0f) return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
	return p;
}

Color get_color(float h)
{
	Color c;
	float s = 1.0f, l = 0.5f;
	float q = l + s - l * s;
	float p = 2.0f * l - q;
	c.r = hue2rgb(p, q, h + 1.0f/3.0f);
	c.g = hue2rgb(p, q, h);
	c.b = hue2rgb(p, q, h - 1.0f/3.0f);
	return c;
}

double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (! pcl_isfinite((*cloud)[i].x))
		{
			continue;
		}
		// Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

void smoothSurface(pcl::PointCloud<PointType>::Ptr &cloud, pcl::PointCloud<pcl::PointNormal> &mls_points)
{	
	// Create a KD-Tree
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);

	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<PointType, pcl::PointNormal> mls;

	mls.setComputeNormals(true);

	// Set parameters
	mls.setInputCloud(cloud);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(resolution * mls_radius_);

	// Reconstruct
	mls.process(mls_points);
}

int main(int argc, char *argv[])
{
	parseCommandLine(argc, argv);

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>());

	if (folder_mode)
	{
		//
		//  Folder Mode (Averaging PCLs)
		//
		std::cout << "Loading clouds from folder " << foldername << std::endl;
		if (!fs::exists(foldername) || !fs::is_directory(foldername))
		{
			std::cerr << "Folder does not exist!" << std::endl;
			return -1;
		}

		std::vector<std::string> filenames;

		fs::recursive_directory_iterator it(foldername);
		fs::recursive_directory_iterator endit;

		while (it != endit)
		{
			if (fs::is_regular_file(*it)
				&& it->path().extension() == ".pcd")
			{
				filenames.push_back(it->path().string());
				std::cout << "Found PCL " << filenames.back() << std::endl;			
			}
			++it;
		}

		if (filenames.empty())
		{
			std::cerr << "Could not find any .pcd files in " << foldername << std::endl;
			return -1;
		}

		pcl::io::loadPCDFile(filenames.front(), *model);
		std::cout << "Loaded cloud " << filenames.front() << std::endl;
		resolution = computeCloudResolution(model);
		std::cout << "Cloud resolution: " << resolution << std::endl;
		int *non_nans = (int*)calloc(model->points.size(), sizeof(int));
		pcl::PointCloud<PointType>::Ptr input(new pcl::PointCloud<PointType>());

		pcl::PointCloud<PointType>::iterator mit, iit;
		for (size_t i = 1; i < filenames.size(); ++i)
		{
			if (pcl::io::loadPCDFile(filenames[i], *input) >= 0)
			{
				std::cout << "Loaded cloud " << filenames[i] << std::endl;
				std::cout << "Number of points: " << input->points.size() << std::endl;
			
				int pidx = 0;
				for (mit = model->begin(), iit = input->begin();
					 iit != input->end() && mit != model->end();
					 ++iit, ++mit, ++pidx)
				{
					PointType p = *iit;
					if (std::isnan(p.x))
						continue;

					non_nans[pidx]++;
					if (std::isnan((*mit).x))
					{
						(*mit).x = p.x;
						(*mit).y = p.y;
						(*mit).z = p.z;
					}
					else
					{
						(*mit).x += p.x;
						(*mit).y += p.y;
						(*mit).z += p.z;						
					}
				}
			}
		}

		// normalize model to obtain average
		int pidx = 0;
		for (mit = model->begin(); mit != model->end(); ++mit, ++pidx)
		{
			if (non_nans[pidx] == 0)
			{
				(*mit).x = std::numeric_limits<double>::quiet_NaN();
				(*mit).y = std::numeric_limits<double>::quiet_NaN();
				(*mit).z = std::numeric_limits<double>::quiet_NaN();
			}
			else
			{
				(*mit).x /= (double)non_nans[pidx];
				(*mit).y /= (double)non_nans[pidx];
				(*mit).z /= (double)non_nans[pidx];
			}
			
		}

		free(non_nans);
	}
	else
	{
		//
		//  Load cloud
		//
		if (pcl::io::loadPCDFile(model_filename_, *model) < 0)
		{
			std::cout << "Error loading model cloud." << std::endl;
			showHelp(argv[0]);
			return -1;
		}

		resolution = computeCloudResolution(model);
		std::cout << "Cloud resolution: " << resolution << std::endl;
	}

	if (!model->isOrganized())
	{
		std::cout << "The input point cloud is not organized. "
				  << "Bilateral Filtering cannot be applied."
				  << std::endl;
	}
	else if (sigma_s > 1e-8 && sigma_r > 1e-8)
	{
		//
		//  Apply Fast Bilateral Filtering
		//
		pcl::FastBilateralFilterOMP<PointType> bilateralFilter;		
		pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
		bilateralFilter.setInputCloud(model);
		bilateralFilter.setSigmaS(sigma_s);
		bilateralFilter.setSigmaR(sigma_r);
		bilateralFilter.applyFilter(*filtered);
		*model = *filtered;
	}

	//
	//  Remove floor
	//
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<PointType> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(plane_thresh_);

	seg.setInputCloud(model);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() == 0)
	{
		PCL_ERROR("Could not estimate a planar model for the given dataset.");
		return -1;
	}

	std::cerr	<< "Model coefficients: " 
				<< coefficients->values[0] << " " 
				<< coefficients->values[1] << " "
				<< coefficients->values[2] << " " 
				<< coefficients->values[3] << std::endl;

	std::cerr << "Model inliers: " << inliers->indices.size() << std::endl;

	pcl::PointCloud<PointType>::Ptr inliers_cloud(new pcl::PointCloud<PointType>());
	pcl::ExtractIndices<PointType> inlier_filter(true); // Initializing with true will allow us to extract the removed indices
	inlier_filter.setInputCloud(model);
	inlier_filter.setIndices(inliers);
	inlier_filter.setNegative(true); // invert to select points which do not belong to the plane
	inlier_filter.filter(*inliers_cloud);

	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	tree->setInputCloud(inliers_cloud);

	std::vector<pcl::PointIndices> clusters;
	pcl::EuclideanClusterExtraction<PointType> ec;
	ec.setClusterTolerance(cluster_tolerance_);
	ec.setMinClusterSize(min_cluster_size_);
	ec.setMaxClusterSize(max_cluster_size_);
	ec.setSearchMethod(tree);
	ec.setInputCloud(inliers_cloud);
	ec.extract(clusters);

	//ece.getRemovedClusters(small_clusters, large_clusters);

	std::cout << "Number of clusters: " << clusters.size() << std::endl;

	std::vector<pcl::PointCloud<PointType>::Ptr> final_clusters;
	pcl::PointCloud<PointType> totalCloud;
	if (!clusters.empty())
	{
		std::cout << "Mean cluster size: " << mean_size(clusters) << std::endl;
        for (int i = 0; i < std::min(max_n, (int)clusters.size()); i++)
		{
			//
			//  Object-wise cloud manipulation
			//
			std::cout << "Cluster " << (i+1) << " has " << clusters[i].indices.size() << " points." << std::endl;
			pcl::PointCloud<PointType>::Ptr cluster_cloud(new pcl::PointCloud<PointType>());
			pcl::ExtractIndices<PointType> final_filter(true); // Initializing with true will allow us to extract the removed indices
			final_filter.setInputCloud(inliers_cloud);
			pcl::PointIndicesConstPtr cluster(new pcl::PointIndices(clusters[i]));
			final_filter.setIndices(cluster);
			final_filter.filter(*cluster_cloud);

			//
			//  Remove outliers
			//
			std::cout << "Removing outliers in cluster " << (i+1) << "..." << std::endl;
			pcl::PointCloud<PointType>::Ptr filtered_cloud(new pcl::PointCloud<PointType>());
			pcl::StatisticalOutlierRemoval<PointType> sor;
			sor.setInputCloud(cluster_cloud);
			sor.setMeanK(mean_k_);
			sor.setStddevMulThresh(std_dev_mul_);
			sor.filter(*filtered_cloud);

			std::cerr << "Filtered cloud size: " << filtered_cloud->points.size() << std::endl;

			//
			//  Smooth surface
			//
			if (mls_radius_ > 1.0f)
			{
				pcl::PointCloud<pcl::PointNormal> smooth;
				smoothSurface(filtered_cloud, smooth);

				pcl::copyPointCloud(smooth, *filtered_cloud);
			}

			if (center_cloud_)
			{
				// compute centroid in order to center the cloud at the point of origin
				pcl::CentroidPoint<PointType> centroid;
				for (int i = 0; i < filtered_cloud->points.size(); i++)
				{
					centroid.add(filtered_cloud->points[i]);
				}

				PointType center;
				centroid.get(center);

				// center points
				for (int i = 0; i < filtered_cloud->points.size(); i++)
				{
					filtered_cloud->points[i].x -= center.x;
					filtered_cloud->points[i].y -= center.y;
					filtered_cloud->points[i].z -= center.z;
				}
			}
			std::stringstream output_name;
			output_name << output_filename_;
			if (clusters.size() > 0)
			{
				output_name << "_" << i;
			}

			if (filtered_cloud->points.empty())
			{
				std::cerr << "Filtered cloud " << i << " is empty!" << std::endl;
			}
			else
			{
				pcl::io::savePCDFileASCII(output_name.str(), *filtered_cloud);
				std::cerr << "Saved " << filtered_cloud->points.size() << " data points to " << output_name.str() << "." << std::endl;
				final_clusters.push_back(filtered_cloud);
				totalCloud += *filtered_cloud;
			}
		}
        // save total cloud (top max_n clusters combined)
		std::string output_name = output_filename_ + "_total";
		pcl::io::savePCDFileASCII(output_name, totalCloud);
		std::cerr << "Saved " << totalCloud.points.size() << " data points to " << output_name << "." << std::endl;
	}
	else
	{
		std::cerr << "Could not extract any model." << std::endl;
	}

	if (!headless && !clusters.empty())
	{
		//
		//  Visualization
		//
        pcl::visualization::PCLVisualizer viewer("PointCloud Separation");
		float hues[] = {0, 0.2f, 0.4f, 0.6f, 0.8f};
		for (int i = 0; i < std::min(5, (int)final_clusters.size()); i++)
		{
			pcl::PointCloud<PointType>::Ptr cluster_cloud = final_clusters[i];
			Color c = get_color(hues[i]);
			pcl::visualization::PointCloudColorHandlerCustom<PointType> color_handler(cluster_cloud, 255.0*c.r, 255.0*c.g, 255.0*c.b);
			std::stringstream cluster_name;
			cluster_name << "cluster " << i;
			viewer.addPointCloud(cluster_cloud, color_handler, cluster_name.str());
		}

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
		}
	}

	return 0;
}
