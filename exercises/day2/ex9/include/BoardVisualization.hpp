#ifndef BOARD_VISUALIZATION_HH_
#define BOARD_VISUALIZATION_HH_

#include <opencv/cv.h>
#include <opencv/highgui.h>

class BoardVisualization {
public:
  BoardVisualization (const size_t width, const size_t height, const size_t radius) 
    : w(width), h(height), r(radius) {
    img = cv::Mat::zeros(2 * r * h, 2 * r * w, CV_8UC1);
    cv::namedWindow("Game of Life", CV_WINDOW_AUTOSIZE);
  }

  void displayBoard(const cv::Mat& board) {
    boardToImage(board, img);
    cv::imshow("Game of Life", img);
  }

private:
  size_t w;
  size_t h;
  size_t r;
  cv::Mat img;

  void boardToImage(const cv::Mat& board, cv::Mat& image) {
    for (int i = 0, x = r; i < h; i++, x += 2 * r) {
      for (int j = 0, y = r; j < w; j++, y += 2 * r) {
        cv::Scalar color(board.at<unsigned char>(j, i) ? 255 : 0);
        cv::circle(image, cv::Point2i(y, x), r, color, CV_FILLED, 8, 0);
      }
    }
  }
};

#endif // BOARD_VISUALIZATION_HH_
