#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>
#include <map>
#include <array>

cv::Mat gaussianKernel(double sigma)
{
  int size = 2 * (int)ceil(3. * sigma) + 1;
  cv::Mat kernel = cv::Mat{size, size, CV_64F, cv::Scalar(0.)};

  int offset = size / 2;
  double coef = 1. / (2. * (double)M_PI * (sigma * sigma));

  double tot = 0.;

  for (int x = (-1) * offset; x <= offset; x++)
  {
    for (int y = (-1) * offset; y <= offset; y++)
    {
      double sqrX = pow(x, 2);
      double sqrY = pow(y, 2);

      kernel.at<double>((kernel.rows / 2) + y, (kernel.cols / 2) + x) = coef * exp(-(sqrX + sqrY) / (2. * sigma * sigma));
      tot += kernel.at<double>((kernel.rows / 2) + y, (kernel.cols / 2) + x);
    }
  }

  for (int x = 0; x < kernel.cols; x++)
  {
    for (int y = 0; y < kernel.rows; y++)
    {
      kernel.at<double>(y, x) /= tot;
    }
  }

  return kernel;
}

cv::Vec3b gaussianAt(const cv::Mat &src, const cv::Mat &kernel, int x, int y)
{
  auto offset = kernel.rows / 2;

  cv::Vec3b sum = 0;

  int kX = 0;
  int kY = 0;

  for (int nx = x - offset; nx <= x + offset; nx++)
  {
    kY = 0;
    for (int ny = y - offset; ny <= y + offset; ny++)
    {
      if (nx >= 0 && ny >= 0 && nx < src.cols && ny < src.rows)
      {
        sum += src.at<cv::Vec3b>(ny, nx) * kernel.at<double>(kY, kX);
      }
      kY++;
    }
    kX++;
  }

  return sum;
}

cv::Mat blur(const cv::Mat &src, double sigma)
{
  cv::Mat dst;
  src.copyTo(dst);

  auto kernel = gaussianKernel(sigma);

  for (int x = 0; x < src.cols; x++)
  {
    for (int y = 0; y < src.rows; y++)
    {
      dst.at<cv::Vec3b>(y, x) = gaussianAt(src, kernel, x, y);
    }
  }

  return dst;
}

cv::Vec3b depthGaussianAt(const cv::Mat &src, const cv::Mat &depth_src, int depth, const cv::Mat &kernel, int x, int y)
{
  auto offset = kernel.rows / 2;

  cv::Vec3b sum = 0;

  int kX = 0;
  int kY = 0;

  for (int nx = x - offset; nx <= x + offset; nx++)
  {
    kY = 0;
    for (int ny = y - offset; ny <= y + offset; ny++)
    {
      if (nx >= 0 && ny >= 0 && nx < src.cols && ny < src.rows)
      {
        if (depth_src.at<unsigned char>(ny, nx) <= depth)
        {
          sum += src.at<cv::Vec3b>(ny, nx) * kernel.at<double>(kY, kX);
        }
        else
        {
          sum += src.at<cv::Vec3b>(ny - 2 * (ny - y), nx - 2 * (nx - x)) * kernel.at<double>(kY - 2 * (kY - offset), kX - 2 * (kX - offset));
        }
      }
      kY++;
    }
    kX++;
  }

  return sum;
}

cv::Mat depthBlur(const cv::Mat &src, const cv::Mat &depth_src, double sigma, int depth)
{
  cv::Mat dst;
  src.copyTo(dst);

  std::array<cv::Mat, 256> kernels;

  for (int x = 0; x < src.cols; x++)
  {
    for (int y = 0; y < src.rows; y++)
    {
      int current_depth = depth_src.at<unsigned char>(y, x);
      if (current_depth <= depth && current_depth > 0)
      {
        cv::Mat kernel = kernels.at(current_depth);

        if (kernel.empty())
        {
          kernel = gaussianKernel(static_cast<double>(depth) / static_cast<double>(current_depth));
          kernels.at(current_depth) = kernel;
        }
        dst.at<cv::Vec3b>(y, x) = depthGaussianAt(src, depth_src, depth, kernel, x, y);
      }
    }
  }

  return dst;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  // check arguments

  if (argc != 5)
  {
    std::cout << "usage: " << argv[0] << "image depth_image blur_level depth_level" << std::endl;
    return -1;
  }

  // if (argv[1] == "no-stereo" && argc != 6)
  // {
  //   std::cout << "usage: " << argv[0] << " stereo image depth_image blur_level depth_level" << std::endl;
  //   return -1;
  // }
  // else if (argv[1] == "stereo" && argc != 5)
  // {
  //   std::cout << "usage: " << argv[0] << "stereo image blur_level depth_level" << std::endl;
  //   return -1;
  // }

  // load the input image
  std::cout << "load image ..." << std::endl;
  cv::Mat image = cv::imread(argv[1]);
  cv::Mat depth = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  // cv::Mat depth = argv[1] == "no-stereo" ? cv::imread(argv[2], cv::IMREAD_GRAYSCALE) : imageToDepth(image);

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

  cv::Mat blur = depthBlur(image, depth, std::stod(argv[3]), std::stoi(argv[4]));

  // setup a window
  cv::namedWindow("image", 1);
  cv::imshow("image", image);
  cv::imshow("depth", depth);
  cv::imshow("depth blur", blur);
  cv::waitKey(0);
  cv::imwrite("../../output/depth_blur.jpg", blur);

  return 1;
}
