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




//input: 1) image file 2) range window size 3) spatial window size  5) minimum region size 6) threads 7) output marker
int main(int argc, char** argv) {
	Mat inputImage		= imread(argv[1]);
	int rangeWindow	= atoi(argv[2]);
	int spatialWindow	= atoi(argv[3]);
	int minCluster 	= atoi(argv[4]);
	int threads			= atoi(argv[5]);
	std::string str 	= "";
	str += argv[6];
	str += "_wr" + IntToString(rangeWindow) + "_ws" + IntToString(spatialWindow) + "_g" + IntToString(minCluster) + ".png";
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

	//phase 2, merge all clusters with peaks within window size of each other
	//for all pixels in image
	for (int i=0; i<imageWidth; i++) {
		for (int j=0; j<imageHeight; j++) {
			
			//if label has not been found yet
			if (regions[i][j]==0) {
		      //label region
	   	   region++;
		      regions[i][j] = region;
		   	   
				//is label big or small
				bool bigCluster = false;
	      
	      	//create label for this point
		      v1d point(5,0);
		      point[0] = i;
	   	   point[1] = j;
	      	point[2] = peaksRGB.at<cv::Vec3b>(j,i)[0];
		      point[3] = peaksRGB.at<cv::Vec3b>(j,i)[1];
		      point[4] = peaksRGB.at<cv::Vec3b>(j,i)[2];
	      
	   	   //random colour value
	      	int r = peaksRGB.at<cv::Vec3b>(j,i)[0];//rand() % 254 + 1;
		      int g = peaksRGB.at<cv::Vec3b>(j,i)[1];//rand() % 254 + 1;
		      int b = peaksRGB.at<cv::Vec3b>(j,i)[2];//rand() % 254 + 1;
	      
	   	   peaksRGB.at<cv::Vec3b>(j,i)[0] = r;
	      	peaksRGB.at<cv::Vec3b>(j,i)[1] = g;
		      peaksRGB.at<cv::Vec3b>(j,i)[2] = b;
				
		      //add to list of neighbours
	   	   v2d neighbours;
	      	neighbours.push_back(point);

		      //grow region by nearest neighbours
		      while(!neighbours.empty()){
					if (neighbours.size() >= minCluster) {
						bigCluster = true;
					}
					//get point from end of list
					point = neighbours.back();
					neighbours.pop_back();
		
					//get neighbouring points
					for(int k = 0; k < 8; k++){                                                                                                        
						int hx = point[0] + dxdy[k][0];
						int hy = point[1] + dxdy[k][1];
		  
						//if point is in scope and has no region
						if((hx > -1) && (hy > -1) && (hx < imageWidth) && (hy < imageHeight) && (regions[hx][hy] == 0)){
							v1d p(5,0);   
							p[0] = hx;
							p[1] = hy;
							p[2] = peaksRGB.at<cv::Vec3b>(hy,hx)[0];
							p[3] = peaksRGB.at<cv::Vec3b>(hy,hx)[1];
							p[4] = peaksRGB.at<cv::Vec3b>(hy,hx)[2];

							//if colour is within window, same region 
							if(nearRGB(p,point,rangeWindow)) {
								regions[hx][hy] = region;
								neighbours.push_back(p);
                                                                                                                                
								peaksRGB.at<cv::Vec3b>(hy,hx)[0] = r;
								peaksRGB.at<cv::Vec3b>(hy,hx)[1] = g;
								peaksRGB.at<cv::Vec3b>(hy,hx)[2] = b;
							}
						}
					}
				}
				//if this is a big cluster, add to list
				if (bigCluster) {
					v1d cluster(4);
					cluster[0] = region;
					cluster[1] = r;
					cluster[2] = g;
					cluster[3] = b;
					bigClusters.push_back(cluster);
				}
				//if this is a small cluster, add to list
				else {
					v1d position(2);
					position[0] = i;
					position[1] = j;
					smallClusters.push_back(position);
				}
			}
		}
	}
	

	
	//phase 3, erase regions with less than threshold number of pixels
	//for each small region
	for (int i=0; i<smallClusters.size(); i++) {
		//get region and add to list of neighbours
		int myRegion = regions[smallClusters[i][0]] [smallClusters[i][1]];
  
		v2d neighbours;
		v2d neighbours_copy;
		neighbours.push_back(smallClusters[i]);
		neighbours_copy.push_back(smallClusters[i]);
		//to find surrounding large region
		v2d adjacentRegions;
		
		//find all other points in this region
		while(!neighbours.empty()){
			
			//get a point from list
			v1d point = neighbours.back();
			neighbours.pop_back();
			//for all neighbouring points
			for(int k = 0; k < 8; k++){
				int hx = point[0] + dxdy[k][0];
				int hy = point[1] + dxdy[k][1];
					
				//if point is in scope
				if((hx > -1) && (hy > -1) && (hx < imageWidth) && (hy < imageHeight)) {
					//and has same region
					if (myRegion == regions[hx][hy]){
						//add to collection of points in this region
						v1d p(2,0);   
						p[0] = hx;
						p[1] = hy;
						neighbours.push_back(p);
						neighbours_copy.push_back(p);
						//remove region
						regions[hx][hy] = 0;
					}

					//if region is a large region
			        	if (findIn(regions[hx][hy],bigClusters) > -1) {
						//if it is already in adjacent regions list, increment number of adjacent points
						int adjRegNum = findIn(regions[hx][hy],adjacentRegions);
						if (adjRegNum > -1) {
							adjacentRegions[adjRegNum][3]++;
						}
						//otherwise add to adjacent regions list
						else{
							v1d adjacent(2,0);
							adjacent[0] = regions[hx][hy];
							adjacent[1] = 1;
							adjacentRegions.push_back(adjacent);
						}
					}
				}
			}
		}
			
		//find adjacent region with most points
		int biggest[2];
		biggest[1] = 0;
		for (int k=0; k<adjacentRegions.size(); k++) {
			if (adjacentRegions[k][3] > biggest[1]) {
				biggest[0] = adjacentRegions[k][0];
				biggest[1] = adjacentRegions[k][1];
			}
		}
		
		//find this in the list of big regions to get rgb values
		int regionNumber = findIn(biggest[0],bigClusters);
		if (regionNumber > -1) {
 			for (int k=0; k<neighbours_copy.size(); k++) {
				//change colour and region of points
				regions[neighbours_copy[k][0]][neighbours_copy[k][1]] = biggest[0];
				peaksRGB.at<cv::Vec3b>(neighbours_copy[k][1],neighbours_copy[k][0])[0] = (uchar)bigClusters[regionNumber][1];
				peaksRGB.at<cv::Vec3b>(neighbours_copy[k][1],neighbours_copy[k][0])[1] = (uchar)bigClusters[regionNumber][2];
				peaksRGB.at<cv::Vec3b>(neighbours_copy[k][1],neighbours_copy[k][0])[2] = (uchar)bigClusters[regionNumber][3];
		  
			}
		}
	}
	
	std::cout << bigClusters.size() << ",";
	
	//write to file
	const char* filename = str.c_str();
	imwrite(filename, peaksRGB, compression_params);

	return 0;
}
