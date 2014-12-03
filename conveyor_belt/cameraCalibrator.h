#ifndef _CAMERA_CALIBRATOR_H_
#define _CAMERA_CALIBRATOR_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

class CameraCalibrator
{
	// input points
	// the points in world coordinates
	vector<vector<Point3f>> _objectPoints;

	// the points positions in pixels
	vector<vector<Point2f>> _imagePoints;

	// output matrices
	Mat _cameraMatrix;
	Mat _distCoeffs;

	// flag to specify how calibration is done
	int _flag;

	// used in image distortion
	Mat _map1, _map2;
	bool _mustInitUndistort;

public:
	CameraCalibrator(): _flag(CV_CALIB_FIX_PRINCIPAL_POINT | CV_CALIB_ZERO_TANGENT_DIST), _mustInitUndistort(true) {};
	
	bool calibrate();
	Mat undistortImage(const Mat& img);
};

#endif