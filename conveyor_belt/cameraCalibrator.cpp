#include "cameraCalibrator.h"
#include "util.h"
#include "parameters.h"

extern Parameters param;

bool
CameraCalibrator::calibrate()
{
	char filename [256];
	Mat chessImg8UC3, chessImg8U;

	for(int i = 1; i <= 4; ++i){
		if(i > 1) continue;
		sprintf_s(filename, "chessboard%03d.png", i);
		chessImg8UC3 = imread(filename);
		if(!chessImg8UC3.data){
			cout << "Error: calibration image(s) not provided." << endl << endl;
			return false;
		}
		cvtColor(chessImg8UC3, chessImg8U, CV_RGB2GRAY);
		Size patternSize (9, 9);
		vector<Point2f> imageCorners;

		//showImage("input", chessImg8U, 1, 1);

		equalizeHist(chessImg8U, chessImg8U);
		//showImage("equalizeHist", chessImg8U, 1, 1);

		adaptiveThreshold(chessImg8U, chessImg8U, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 17, 0);
		//threshold(chessImg8U, chessImg8U, 120, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//showImage("threshold", chessImg8U, 1, 0);
		
		bool patternFound = findChessboardCorners(chessImg8U, patternSize, imageCorners,
			CALIB_CB_FAST_CHECK);

		TermCriteria criteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
		if(patternFound)
			cornerSubPix(chessImg8U, imageCorners, Size(11, 11), Size(-1, -1), criteria);
	
		Mat drawImg = chessImg8UC3.clone();
		drawChessboardCorners(drawImg, patternSize, Mat(imageCorners), patternFound);
		//showImage("chessboard", drawImg, 1, 0);
		sprintf_s(filename, "%s\\chessboardCorners.jpg", param.getOutputPath());
		imwrite(filename, drawImg);
		drawImg.release();
		
		if(patternFound){
			vector<Point3f> objectCorners;
			for(int row = 0; row < patternSize.height; ++row)
				for(int col = 0; col < patternSize.width; ++col)
					objectCorners.push_back(Point3f(col*25, row*25, 0));

			_objectPoints.push_back(objectCorners);
			_imagePoints.push_back(imageCorners);
		}
	}

	vector<Mat> rvecs, tvecs;
	double err = calibrateCamera(_objectPoints, _imagePoints, chessImg8U.size(), 
						 		 _cameraMatrix, _distCoeffs, rvecs, tvecs, 
								 _flag);

	Mat undistortedImg;
	undistort(chessImg8UC3, undistortedImg, _cameraMatrix, _distCoeffs);
	//showImage("undistorted", undistortedImg);
	//sprintf_s(filename, "%s\\undistorted.jpg", param.getOutputPath());
	//imwrite(filename, undistortedImg);

	return true;
}

Mat
CameraCalibrator::undistortImage(const Mat& img)
{
	Mat undistorted;

	/*initUndistortRectifyMap(_cameraMatrix, _distCoeffs, Mat(), Mat(), chessImg8U.size(), CV_32FC1, _map1, _map2);
	remap(temp, undistorted, _map1, _map2, INTER_LINEAR);*/
	
	undistort(img, undistorted, _cameraMatrix, _distCoeffs);
	
	//showImage("undistorted", undistorted);

	return undistorted;
}