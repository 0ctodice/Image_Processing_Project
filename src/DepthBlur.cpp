///////////////////////////////////////////////////////////////////////////
//
// ------------------------------------------------------------------------
//   _____      _            _ _
//  |  _  |    | |          | (_)
//  | |/' | ___| |_ ___   __| |_  ___ ___
//  |  /| |/ __| __/ _ \ / _` | |/ __/ _ \
//  \ |_/ / (__| || (_) | (_| | | (_|  __/
//   \___/ \___|\__\___/ \__,_|_|\___\___|
//
// ------------------------------------------------------------------------
//
//  Projet de traitement d'image Master 1 Informatique
//  ~ Thomas DUMONT A.K.A 0ctodice
//
// ------------------------------------------------------------------------
//
///////////////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>
#include <map>

cv::Mat gaussianKernel(double sigma)
{
  int size = 2 * static_cast<int>(ceil(3. * sigma)) + 1;
  cv::Mat kernel = cv::Mat(size, size, CV_64F, cv::Scalar(0.));

  double coef = 1. / (2. * static_cast<double>(M_PI) * pow(sigma, 2.));

  auto offset = size / 2;

  double sum = 0.;

  for (int x = -1 * offset; x <= offset; x++)
  {
    double powX = pow(static_cast<double>(x), 2.);
    for (int y = -1 * offset; y <= offset; y++)
    {
      double powY = pow(static_cast<double>(y), 2.);
      double value = coef * exp(-(powX + powY) / (2. * pow(sigma, 2.)));
      kernel.at<double>(offset + y, offset + x) = value;
      sum += value;
    }
  }

  kernel /= sum;

  return kernel;
}

cv::Vec3b depthBlurAt(const cv::Mat &image_src, const cv::Mat &depth_map, const cv::Mat &kernel, int depth_level, int ox, int oy)
{
  auto offset = kernel.rows / 2;

  double r = 0.;
  double g = 0.;
  double b = 0.;

  int kX = 0;

  for (int x = ox - offset; x <= ox + offset; x++)
  {
    int kY = 0;
    for (int y = oy - offset; y <= oy + offset; y++)
    {
      if (ox >= 0 && oy >= 0 && ox < image_src.cols && oy < image_src.rows)
      {
        if (depth_map.at<uchar>(y, x) <= depth_level)
        {
          b += static_cast<double>(image_src.at<cv::Vec3b>(y, x)[0]) * kernel.at<double>(kY, kX);
          g += static_cast<double>(image_src.at<cv::Vec3b>(y, x)[1]) * kernel.at<double>(kY, kX);
          r += static_cast<double>(image_src.at<cv::Vec3b>(y, x)[2]) * kernel.at<double>(kY, kX);
        }
        else
        {
          auto nx = x - 2 * (x - ox);
          auto ny = y - 2 * (y - oy);
          if (ny >= 0 && nx >= 0 && nx < image_src.cols && ny < image_src.rows)
          {
            b += static_cast<double>(image_src.at<cv::Vec3b>(ny, nx)[0]) * kernel.at<double>(kY, kX);
            g += static_cast<double>(image_src.at<cv::Vec3b>(ny, nx)[1]) * kernel.at<double>(kY, kX);
            r += static_cast<double>(image_src.at<cv::Vec3b>(ny, nx)[2]) * kernel.at<double>(kY, kX);
          }
        }
      }
      kY++;
    }
    kX++;
  }

  return cv::Vec3b{static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r)};
}

cv::Mat cleanDepthBlur(const cv::Mat &image_src, const cv::Mat &depth_map, int depth_level)
{
  cv::Mat image_dst;
  image_src.copyTo(image_dst);

  std::map<uchar, cv::Mat> kernels;

  double factor = depth_level <= 85 ? 1. : depth_level <= 170 ? 2.
                                                              : 3.;

  cv::Mat kernel = gaussianKernel(factor * static_cast<double>(depth_level) / 100.);

  for (int x = 0; x < image_dst.cols; x++)
  {
    for (int y = 0; y < image_dst.rows; y++)
    {
      auto current = depth_map.at<unsigned char>(y, x);
      if (current <= depth_level)
      {
        image_dst.at<cv::Vec3b>(y, x) = depthBlurAt(image_src, depth_map, kernel, depth_level, x, y);
      }
    }
  }

  return image_dst;
}

cv::Mat evolvDepthBlur(const cv::Mat &image_src, const cv::Mat &depth_map, int depth_level)
{
  cv::Mat image_dst;
  image_src.copyTo(image_dst);

  std::map<uchar, cv::Mat> kernels;

  double factor = depth_level <= 85 ? 1. : depth_level <= 170 ? 2.
                                                              : 3.;

  for (int x = 0; x < image_dst.cols; x++)
  {
    for (int y = 0; y < image_dst.rows; y++)
    {
      auto current = depth_map.at<unsigned char>(y, x) != 0 ? depth_map.at<unsigned char>(y, x) : depth_level / factor;
      if (current <= depth_level)
      {
        cv::Mat kernel;
        auto it = kernels.find(current);
        if (it != kernels.end())
        {
          kernel = it->second;
        }
        else
        {
          kernel = gaussianKernel(factor * static_cast<double>(depth_level) / (static_cast<double>(current)));
          kernels.emplace(current, kernel);
        }

        image_dst.at<cv::Vec3b>(y, x) = depthBlurAt(image_src, depth_map, kernel, depth_level, x, y);
      }
    }
  }

  return image_dst;
}

void usage(char **argv)
{
  std::cout << "usage: " << argv[0] << " image depth_image depth_level blur_type" << std::endl;
  exit(-1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  // check arguments

  if (argc != 5)
  {
    usage(argv);
  }

  std::string blur_type = argv[4];

  if (blur_type.compare("evolve") != 0 && blur_type.compare("clean") != 0)
  {
    usage(argv);
  }

  // load the input image
  std::cout << "load image ..." << std::endl;
  cv::Mat image = cv::imread(argv[1]);
  cv::Mat depth = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

  if (image.empty())
  {
    std::cout << "error loading " << argv[1] << std::endl;
    return -1;
  }

  if (depth.empty())
  {
    std::cout << "error loading " << argv[2] << std::endl;
    return -1;
  }

  std::cout << "image size : " << image.cols << " x " << image.rows << std::endl;

  cv::Mat blured = blur_type.compare("evolve") == 0 ? evolvDepthBlur(image, depth, std::stoi(argv[3])) : cleanDepthBlur(image, depth, std::stoi(argv[3]));

  // setup a window
  cv::imshow("image source", image);
  cv::imshow("depth map", depth);
  cv::imshow("depth blur", blured);
  cv::waitKey(0);
  cv::imwrite("../../output/depth_blur.jpg", blured);

  return 1;
}
