# ICIP_2018_Mean_Shift_TradeOffs

## [PARALLEL MEAN SHIFT ACCURACY AND PERFORMANCE TRADE-OFFS](https://www.macs.hw.ac.uk/~rs46/papers/icip2018/icip-2018.pdf)

The following 3 files contain source code for meanshift segmentation, clustering
and merging only and clustering only respectively:

• meanshift_segment.cpp

• meanshift_segment_2phases.cpp

• meanshift_segment_1phase.cpp

In order to compile the C++ files, save them in the same directory as
CMakeLists.txt and type the following 2 commands into the terminal:
cmake .
make

Input arguments for the executables are as follows:

• Meanshift_segment:  input image,  range window radius,  spatial window radius,  threshold,  number of threads,  output file name

• Meanshift_segment_2phases:  input image,  range window radius,  spatial window radius,  number of threads,  output file name

• Meanshift_segment_1phase:   input image,  range window radius,  spatial window radius,  number of threads,  output file name

The Berkley segmentation dataset is available from:
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
