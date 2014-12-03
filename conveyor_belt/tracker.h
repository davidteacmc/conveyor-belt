#ifndef _TRACKER_H_
#define _TRACKER_H_

#include <iostream>
#include <list>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "fgObject.h"

using namespace std;
using namespace cv;

//******************************* Track ****************************************
class Track
{
public:
    Track(const FgObject& obj, int frame);
	~Track() {}
    
    int getStartFrame() const { return _startFrame; }
    int getEndFrame() const { return _endFrame; }
	Point2f	getPosition() const { return _position; }
    Point2f	getVelocity() const { return _velocity; }
	FgObject getLastObject() const { return _objSeq.back(); }
	Mat getAlphaMap() const { return _alphaMap8U; }

    void setStartFrame(int n) { _startFrame = n; }
    void setEndFrame(int n) { _endFrame = n; }
	void setPosition(const Point2f& x) { _position = x; }
    void setVelocity(const Point2f& v) { _velocity = v; }
	
    void addObject(const FgObject& obj);
	void incrementLossCount();
	bool isActive() const;
	Rect boundingBox() const { return _objSeq.back().boundingBox(); }

	void aggregateTarget(OutputArray dst);
	int aggregateTarget(OutputArray dst, double weight, Point2f* pts);
    int aggregateTargetAlpha(OutputArray dst, double weight, Point2f* pts);

private:
	int             _startFrame;
    int             _endFrame;
	int				_lossCount;

	Point2f	        _position;
	Point2f         _velocity;
	
	Mat				_alphaMap8U;
    vector<FgObject>    _objSeq;
};

//***************************** TrackUpdate ************************************
class TrackUpdate
{
public:
    TrackUpdate();
	~TrackUpdate();
    
    void refineTargetBoundary(const Track& tr);

};

//**************************** MyKalmanFilter **********************************
class MyKalmanFilter
{
public:
    MyKalmanFilter(int dynamParams, int measureParams, int ctrlParams = 0);
    void init(int dynamParams, int measureParams, int ctrlParams = 0);
    const Mat& predict(const Mat& control = Mat());
    const Mat& correct(const Mat& measurement);
    
    Mat     _A;     // transition matrix
    Mat     _B;     // control matrix
    Mat     _H;     // measurement matrix
    Mat     _xPre;  // predicted state [x'(k) = A*x(k-1) + B*u(k)]
    Mat     _xPost; // corrected state [x(k) = x'(k) + K(k)*(z(k)-H*x'(k))]
    Mat     _Q;     // process noise covariance matrix
    Mat     _R;     // measurement noise covariance matrix
    Mat     _PPre;  // a priori error estimate covariance matrix [P'(k) = A*P(k-1)*At + Q)]
    Mat     _PPost; // a posteriori error estimate covariance matrix [P(k) = (I - K(k)*H)*P'(k)]
    Mat     _K;     // Kalman gain matrix [K(k) = P'(k)*Ht*inv(H*P'(k)*Ht + R)]
};

//****************************** Tracker ***************************************
class Tracker
{
public:
    Tracker();
    ~Tracker();

    void drawTargets(Mat img);
	bool makeTargetMaskOutputs(InputArray src, OutputArray dst);
	void trackTargets(const vector<FgObject>& fgObjs);
	
private:
	void createTarget(const FgObject& obj);
	
	vector<Track>   _trackList;

	TrackUpdate		_trackUpdate;

	// stored frames for target output
	list<Mat>		_frameList;
	int				_frameNumFront;

	int				_objCount;
};

#endif // _TRACKER_H_