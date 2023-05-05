#include <iostream>
#include <opencv2/opencv.hpp>
#include "Demons.h"

#ifdef _DEBUG
#pragma comment(lib, "D:/opencv_contrib/opencv_contrib/x64/vc17/lib/opencv_world460d.lib")
#else
#pragma comment(lib, "D:/opencv_contrib/opencv_contrib/x64/vc17/lib/opencv_world460.lib")
#endif // _DEBUG

int main(int argc, char* argv[])
{
	const std::string fixed_image_path = "D:/Code-VS/Project-Demons/Project-Demons/imgs/dataset5-templ.png";
	const std::string moved_image_path = "D:/Code-VS/Project-Demons/Project-Demons/imgs/dataset5-tested6.png";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);

	demons_params_t demons_params;
	demons_params.niter_ = 30;
	demons_params.alpha_ = 0.6;
	demons_params.sigma_fluid_ = 0.3;	// 0.3
	demons_params.sigma_diffusion_ = 0.6;	// 0.6

	Demons* demons = new Demons(demons_params);
	//demons->SingleScale(fixed_image, moved_image);
	demons->MultiScale(fixed_image, moved_image);

	cv::namedWindow("res-map", cv::WINDOW_NORMAL);
	cv::imshow("res-map", demons->res_image_);
	cv::waitKey();

	delete demons;

	return 0;
}
