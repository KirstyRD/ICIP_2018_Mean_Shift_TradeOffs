#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>

using namespace cv;

// first form
typedef std::vector<uint16_t> v1d;
typedef std::vector<v1d> v2d;

std::string IntToString (int a)
{
  std::ostringstream temp;
  temp<<a;
  return temp.str();
}

//v1d: x,y;r,g,b
bool nearRGB(v1d a, v1d b, int windowRg) {
  return (a[2]-b[2])*(a[2]-b[2]) + (a[3]-b[3])*(a[3]-b[3]) + (a[4]-b[4])*(a[4]-b[4]) < windowRg*windowRg;
}

bool nearXY(v1d a, v1d b, int windowSp) {
  return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) < windowSp*windowSp;
}

int findIn(int pos, v2d positions) {
  for (int i=0;i<positions.size();i++) {
    if (pos == positions[i][0]) {
      return i;
    }
  }
  return -1;
}

bool isIn(cv::Vec3b pos, std::vector<cv::Vec3b> positions) {
  for (int i=0;i<positions.size();i++) {
    if (pos[0] == positions[i][0] && pos[1] == positions[i][1] && pos[2] == positions[i][2]) {
      return true;
    }
  }
  return false;
}




//input: 1) image file 2) range window size 3) spatial window size 4) threads 5) output marker
int main(int argc, char** argv) {
	Mat inputImage		= imread(argv[1]);
	int rangeWindow	= atoi(argv[2]);
	int spatialWindow	= atoi(argv[3]);
	int threads			= atoi(argv[4]);
	std::string str 	= "../outputs/";
	str += argv[5];
	str += "_wr" + IntToString(rangeWindow) + "_ws" + IntToString(spatialWindow) + "_1phase.png";
	//image dimensions
	int imageWidth		= inputImage.cols;
	int imageHeight	= inputImage.rows;
	//joint space peaks of each point
	Mat peaksRGB(imageHeight, imageWidth, CV_8UC3);
	int peaksXY[imageWidth][imageHeight][2];
	//regions for each pixel
	int regions[imageWidth][imageHeight];
	int region = 0;
	//contains the clusters greater than threshold size
	v2d bigClusters;
	v2d smallClusters;
	bool finished = false;
	const int dxdy[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
	
	//save images as uncompressed png
	vector<int> compression_params;
   compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
   compression_params.push_back(0);


	// phase 1, for each co-ord in joint space, find the peak
	#pragma omp parallel for shared(peaksRGB, peaksXY, inputImage) num_threads(threads)
	for (int i=0; i<imageWidth; i++) {
	  for (int j=0; j<imageHeight; j++) {
	    //initialise
	    int peakFound = false;
	    int count     = 0;
		//initialise peak = input position
	    peaksRGB.at<cv::Vec3b>(j,i)[0] = inputImage.at<cv::Vec3b>(j,i)[0];
	    peaksRGB.at<cv::Vec3b>(j,i)[1] = inputImage.at<cv::Vec3b>(j,i)[1];
	    peaksRGB.at<cv::Vec3b>(j,i)[2] = inputImage.at<cv::Vec3b>(j,i)[2];
	    peaksXY[i][j][0] = i;
	    peaksXY[i][j][1] = j;
	    
	    std::vector<cv::Vec3b> previousPeaks;
	    cv::Vec3b posn = peaksRGB.at<cv::Vec3b>(j,i);
	    previousPeaks.push_back(posn);

	    //loop until peak is found
	    while (peakFound == false) {
	      count++;
	      int rVal = 0;
	      int gVal = 0;
	      int bVal = 0;
	      int xVal = 0;
	      int yVal = 0;
	      int norm = 0;
	      
	      // for each point within chosen window, find center of mass
	      for (int k=(-1)*spatialWindow; k<spatialWindow+1; k++) {
				//std::cout << i << std::endl;
				for (int l=(-1)*spatialWindow; l<spatialWindow+1; l++) {
					// if the point in within a circle of the center
					if (l*l+k*k <= spatialWindow*spatialWindow){
						// if the point is within the image
						if (k+peaksXY[i][j][0] < imageWidth and k+peaksXY[i][j][0] >= 0
							and l+peaksXY[i][j][1] < imageHeight and l+peaksXY[i][j][1] >= 0) {
							// if point is within RGB window
							if ( (peaksRGB.at<cv::Vec3b>(j,i)[0] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [0]) * (peaksRGB.at<cv::Vec3b>(j,i)[0] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [0])
								+ (peaksRGB.at<cv::Vec3b>(j,i)[1] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [1]) * (peaksRGB.at<cv::Vec3b>(j,i)[1] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [1])
								+ (peaksRGB.at<cv::Vec3b>(j,i)[2] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [2]) * (peaksRGB.at<cv::Vec3b>(j,i)[2] - inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [2])
								<= rangeWindow*rangeWindow ) {
	
			
								// update values of 5-vector
								rVal += inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [0] - peaksRGB.at<cv::Vec3b>(j,i)[0];
								gVal += inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [1] - peaksRGB.at<cv::Vec3b>(j,i)[1];
								bVal += inputImage.at<cv::Vec3b>(peaksXY[i][j][1]+l, peaksXY[i][j][0]+k) [2] - peaksRGB.at<cv::Vec3b>(j,i)[2];
								xVal += k;
								yVal += l;
								norm++;

							}
						}
					}
				}
	      }
      
	      // update value of each peak in
	      if (norm !=0) {
				peaksRGB.at<cv::Vec3b>(j,i)[0] += round(rVal/norm);
				peaksRGB.at<cv::Vec3b>(j,i)[1] += round(gVal/norm);
				peaksRGB.at<cv::Vec3b>(j,i)[2] += round(bVal/norm);
				peaksXY[i][j][0] += round(xVal/norm);
				peaksXY[i][j][1] += round(yVal/norm);
	      }
	      // check if current point is the peak
	      if (isIn(peaksRGB.at<cv::Vec3b>(j,i), previousPeaks) || (rVal==0 && gVal==0 && bVal==0) ) {
				peakFound = true;
	      }
	    	cv::Vec3b posn = peaksRGB.at<cv::Vec3b>(j,i);
	    	previousPeaks.push_back(posn);
	    }
	  }
	}
	
	//write to file
	const char* filename = str.c_str();
	imwrite(filename, peaksRGB );

	return 0;
}
