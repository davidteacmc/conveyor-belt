#include <iomanip>
#include <fstream>
using namespace std;

#include "tracker.h"
#include "util.h"
#include "parameters.h"

extern Parameters param;
extern ofstream csvOut;

//******************************* Track ****************************************

Track::Track(const FgObject& obj, int frame)
	: _startFrame(frame), 
	  _endFrame(-1),
	  _lossCount(0),
	  _position(obj.center())
{
	if(param.getVideoSourceType() == 0)
		_velocity = Point2f(10, 0);
	else
		_velocity = Point2f(0, 10);
	
	_objSeq.reserve(20);
	addObject(obj);
}

void
Track::addObject(const FgObject& obj)
{
	_objSeq.push_back(obj);
	Scalar color = _objSeq[0].getRectColor();
	_objSeq.back().setRectColor(color);
}

void
Track::incrementLossCount()
{
	++_lossCount;
	if(_lossCount > 3) _endFrame = param.getFrameNum();
}

bool
Track::isActive() const
{
	return (_endFrame == -1);
}

void
Track::aggregateTarget(OutputArray dst)
{
	dst.create(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
	Mat fg = dst.getMat();
	fg = 0;

	//char filename[256];
	vector<vector<Point> > shiftedContours;
	shiftedContours.reserve(_objSeq.size());

	Point frameCen(param.getFrameWidth() / 2, param.getFrameHeight() / 2);
	Point objCen;
	for(size_t i = 0; i < _objSeq.size(); ++i){
		FgObject obj = _objSeq[i];
		objCen = obj.center();
		vector<Point> contour = obj.getContour();
		vector<Point> c;
		c.reserve(contour.size());
		shiftedContours.push_back(c);
		
		for(size_t j = 0; j < contour.size(); ++j){
			Point p = contour[j] - objCen + frameCen;
			shiftedContours[i].push_back(p);
		}
	}

	for(size_t i = 0; i < _objSeq.size(); ++i){
		Mat tempFg = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
		drawContours(tempFg, shiftedContours, i, Scalar::all(255), -1);
		
		/*if(_objSeq.size() > 3){
			sprintf_s(filename, "%s\\ensemble\\%05d.jpg", param.getOutputPath(), getStartFrame() + i);
			imwrite(filename, tempFg);
		}*/

		for(int y = 0; y < tempFg.rows; ++y){
			for(int x = 0; x < tempFg.cols; ++x){
				if(tempFg.at<uchar>(y,x) > 0)
					fg.at<uchar>(y,x) = fg.at<uchar>(y,x) + 1;
			}
		}
	}

	
	double thresh = _objSeq.size() * 0.5;
	threshold(fg, fg, thresh, 255, CV_THRESH_BINARY);
	
	Point lastObjCen = _objSeq.back().center();
	vector<vector<Point> > cc = extractContours(fg);
	vector<Point> c = cc[0];
	Point fgCen = minAreaRect(c).center;
	vector<vector<Point> > finalContours;
	finalContours.push_back(vector<Point>());
	for(size_t i = 0; i < c.size(); ++i){
		Point p = c[i] - fgCen + lastObjCen;
		finalContours[0].push_back(p);
	}

	fg = 0;
	drawContours(fg, finalContours, 0, Scalar::all(255), -1);
	
	/*if(_objSeq.size() > 3){
		sprintf_s(filename, "%s\\ensemble\\final.jpg", param.getOutputPath(), param.getFrameNum());
		imwrite(filename, fg);
	}*/
}

int
Track::aggregateTarget(OutputArray dst, double weight, Point2f* pts)
{
	dst.create(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
	Mat fg = dst.getMat();
	fg = 0;
	weight = weight < 0 ? 0.0 : (weight > 1.0 ? 1.0 : weight);

	vector<vector<Point> > shiftedContours;
	shiftedContours.reserve(_objSeq.size());

	Point frameCen(param.getFrameWidth() / 2, param.getFrameHeight() / 2);
	Point objCen;
	for(size_t i = 0; i < _objSeq.size(); ++i){
		FgObject obj = _objSeq[i];
		objCen = obj.center();
		vector<Point> contour = obj.getContour();
		vector<Point> c;
		c.reserve(contour.size());
		shiftedContours.push_back(c);
		
		for(size_t j = 0; j < contour.size(); ++j){
			Point p = contour[j] - objCen + frameCen;
			shiftedContours[i].push_back(p);
		}
	}

	for(size_t i = 0; i < _objSeq.size(); ++i){
		Mat tempFg = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
		drawContours(tempFg, shiftedContours, i, Scalar::all(255), -1);
		//showImage("tempFg", tempFg, 1);

		for(int y = 0; y < tempFg.rows; ++y){
			for(int x = 0; x < tempFg.cols; ++x){
				if(tempFg.at<uchar>(y,x) > 0)
					fg.at<uchar>(y,x) = fg.at<uchar>(y,x) + 1;
			}
		}
	}

	
	double thresh = _objSeq.size() * 0.5;
	threshold(fg, fg, thresh, 255, CV_THRESH_BINARY);
	//showImage("fg", fg, 1);

	vector<vector<Point> > cc = extractContours(fg);
	vector<Point> c;
	for(size_t j = 0; j < cc.size(); ++j){
		if(contourArea(cc[j]) * 4 >= param.getMinArea()){
			c = cc[j];
			break;
		}
	}
	if(c.size() == 0) c = cc[0];
	
	// choose the object image having the most similar silhouette to the 
	//    generated one and return it as the output of this target
	int halfSeqLen = _objSeq.size() / 2;
	int idx = -1;
	double minCost = 1e6;
	for(size_t i = 0; i < _objSeq.size(); ++i){
		double overlapRate = overlappingArea(FgObject(shiftedContours[i]), FgObject(c)) / contourArea(c);
		double temporalDist = abs(double(i) - halfSeqLen) / double(halfSeqLen);

		if(overlapRate < 0.7) continue;
		double cost = weight*(1-overlapRate) + (1-weight)*temporalDist;

		if(cost < minCost){
			idx = i;
			minCost = cost;
		}
	}
	if(idx == -1) idx = halfSeqLen;

	vector<Point> bestContour = _objSeq[idx].getContour();
	Point bestObjCen = orientedBoundingBox(bestContour).center;
	Point fgCen = orientedBoundingBox(c).center;
	vector<vector<Point> > finalContours;
	finalContours.push_back(vector<Point>());
	for(size_t i = 0; i < c.size(); ++i){
		Point p = c[i] - fgCen + bestObjCen;
		finalContours[0].push_back(p);
	}

	fg = 0;
	drawContours(fg, finalContours, 0, Scalar::all(255), -1);
	//showImage("fg", fg, 1);

	// extract and return by pointer the corner points of oriented bounding box
	RotatedRect orientedBox = orientedBoundingBox(_objSeq[idx].getContour());
	orientedBox.points(pts);

	// return the frame number of the best match
	return (getStartFrame() + idx);
}

int
Track::aggregateTargetAlpha(OutputArray dst, double weight, Point2f* pts)
{
	dst.create(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
	Mat fg = dst.getMat();
	fg = 0;
	weight = weight < 0 ? 0.0 : (weight > 1.0 ? 1.0 : weight);

	vector<vector<Point> > shiftedContours;
	shiftedContours.reserve(_objSeq.size());
	vector<Mat> shiftedAlphaMaps;
	shiftedAlphaMaps.reserve(_objSeq.size());


	Mat alphaMap32F = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_32F);
	
	Point frameCen(param.getFrameWidth() / 2, param.getFrameHeight() / 2);
	Point objCen;
	for(size_t i = 0; i < _objSeq.size(); ++i){
		const FgObject& obj = _objSeq[i];
		objCen = obj.center();

        // shift the whole alpha map via affine transformation (only translation is used)
		Mat affine = (Mat_<float>(2, 3) << 1, 0, frameCen.x - objCen.x,
										   0, 1, frameCen.y - objCen.y);

		Mat alpha = obj.getAlphaMap();
		Mat newAlpha(param.getFrameHeight(), param.getFrameWidth(), CV_32F);
		warpAffine(alpha, newAlpha, affine, newAlpha.size());
		shiftedAlphaMaps.push_back(newAlpha);
		//showImage("new alpha", newAlpha, 1);
		alphaMap32F += newAlpha;

		char filename [256];
		Mat outputAlpha;
		newAlpha.convertTo(outputAlpha, CV_8U, 255.0);
		//sprintf_s(filename, "%s\\alpha\\%05d-%02d.jpg", param.getOutputPath(), _startFrame, i);
		//imwrite(filename, outputAlpha);
		
        // shift every point on the contour
		vector<Point> contour = obj.getContour();
		vector<Point> c;
		c.reserve(contour.size());
		shiftedContours.push_back(c);
		for(size_t j = 0; j < contour.size(); ++j){
			Point p = contour[j] - objCen + frameCen;
			shiftedContours[i].push_back(p);
		}
	}

	alphaMap32F /= float(_objSeq.size());

	// apply kernel density estimation to generate a final alpha map
	float kernelWidth = 0.15;
	float stopEpsilon = 0.001;

	for(size_t y = 0; y < alphaMap32F.rows; ++y){
		for(size_t x = 0; x < alphaMap32F.cols; ++x){
			
			float avgAlpha = alphaMap32F.at<float>(y, x);
			if(avgAlpha < stopEpsilon)
				continue;

			vector<float> sampleAlpha;
			sampleAlpha.reserve(_objSeq.size());

			for(size_t i = 0; i < _objSeq.size(); ++i){
				//showImage("alpha", shiftedAlphaMaps[i], 1);
				float a = shiftedAlphaMaps[i].at<float>(y, x);
				sampleAlpha.push_back(a);
			}

			// approaching maximum by mean-shift
			float numer = 0, denom = 1e-6;
			float lastAvgAlpha = 0;
			do{
				for(size_t i = 0; i < _objSeq.size(); ++i){
					numer += sampleAlpha[i] * exp(-pow(avgAlpha - sampleAlpha[i], 2)/(2.0*pow(kernelWidth, 2)));
					denom += exp(-pow(avgAlpha - sampleAlpha[i], 2)/(2.0*pow(kernelWidth, 2)));
				}

				lastAvgAlpha = avgAlpha;
				avgAlpha = numer / denom;
				
			}while(abs(avgAlpha - lastAvgAlpha) > stopEpsilon);

			alphaMap32F.at<float>(y, x) = avgAlpha;
		}
	}

	//showImage("alphaMeanShift", alphaMap32F, 1);

	_alphaMap8U = Mat::zeros(alphaMap32F.size(), CV_8U);
	alphaMap32F.convertTo(_alphaMap8U, CV_8U, 255);
	//showImage("final alpha", alphaMap32F, 1);

	double thresh = 0.2;
	threshold(alphaMap32F, alphaMap32F, thresh, 1.0, CV_THRESH_BINARY);
	alphaMap32F.convertTo(fg, CV_8U, 255);
	
	vector<vector<Point> > cc = extractContours(fg);
	vector<Point> c;
	for(size_t j = 0; j < cc.size(); ++j){
		if(contourArea(cc[j]) * 4 >= param.getMinArea()){
			c = cc[j];
			break;
		}
	}
	if(c.size() == 0) c = cc[0];

	// choose the object image having the most similar silhouette to the 
	//    generated one and return it as the output of this target
	int halfSeqLen = _objSeq.size() / 2;
	int idx = -1;
	double minCost = 1e6;
	for(size_t i = 0; i < _objSeq.size(); ++i){
		double overlapRate = overlappingArea(FgObject(shiftedContours[i]), FgObject(c)) / contourArea(c);
		double temporalDist = abs(double(i) - halfSeqLen) / double(halfSeqLen);

		if(overlapRate < 0.7) continue;
		double cost = weight * (1 - overlapRate) + (1 - weight) * temporalDist;

		if(cost < minCost){
			idx = i;
			minCost = cost;
		}
	}
	if(idx == -1) idx = halfSeqLen;

	vector<Point> bestContour = _objSeq[idx].getContour();
	Point bestObjCen = orientedBoundingBox(bestContour).center;
	Point fgCen = orientedBoundingBox(c).center;
	vector<vector<Point> > finalContours;
	finalContours.push_back(vector<Point>());
	for(size_t i = 0; i < c.size(); ++i){
		Point p = c[i] - fgCen + bestObjCen;
		finalContours[0].push_back(p);
	}

	fg = 0;
	drawContours(fg, finalContours, 0, Scalar::all(255), -1);
	//showImage("fg", fg, 1);

	// extract and return by pointer the corner points of oriented bounding box
	RotatedRect orientedBox = orientedBoundingBox(_objSeq[idx].getContour());
	orientedBox.points(pts);

	// return the frame number of the best match
	//return (getStartFrame() + idx);
	return _objSeq[idx].getFrameNum();
}

//**************************** TrackUpdate *************************************

TrackUpdate::TrackUpdate()
{

}

TrackUpdate::~TrackUpdate()
{

}

void
TrackUpdate::refineTargetBoundary(const Track& tr)
{

}

//****************************** Tracker ***************************************

Tracker::Tracker() : _frameNumFront(0), _objCount(0)
{

}

Tracker::~Tracker()
{

}

void Tracker::createTarget(const FgObject& obj)
{
	Track newTrack(obj, param.getFrameNum());
	_trackList.push_back(newTrack);
}

void Tracker::drawTargets(Mat img)
{
	for(size_t i = 0; i < _trackList.size(); ++i){
		Track t = _trackList[i];
		if(t.isActive()){
			FgObject obj = t.getLastObject();
			Point vec = Point(t.getPosition() - obj.center());
			obj.moveContour(vec);
			obj.draw(img);
		}
	}
}

bool
Tracker::makeTargetMaskOutputs(InputArray src, OutputArray dst)
{
	bool hasOutput = false;
	if(!src.obj) return false;
	Mat inImg = src.getMat();
	dst.create(Size(param.getFrameWidth(), param.getFrameHeight()), inImg.type());
	Mat targetImg = dst.getMat();
	char filename [256];

	// store the input frame to frame list
	if(_frameList.empty()) _frameNumFront = param.getFrameNum();
	Mat img = inImg.clone();
	_frameList.push_back(img);

	for(size_t i = 0; i < _trackList.size(); ++i){
		Track tr = _trackList[i];
		if(param.getFrameNum() != tr.getEndFrame())
			continue;
		
		Mat mask = Mat::zeros(inImg.size(), inImg.type());
		
		//tr.generateTarget(mask); int outFrameNum = param.getFrameNum();
		Point2f pts [4];
		int outFrameNum = tr.aggregateTargetAlpha(mask, param.getWeightOverlapping(), pts);

		char filename [256];
		//sprintf_s(filename, "%s\\alphaMaps\\frame%05d_obj%02d.jpg", param.getOutputPath(), param.getFrameNum(), i);
		//imwrite(filename, tr.getAlphaMap());

		list<Mat>::const_iterator it = _frameList.begin();
		for(int k = 0; k < outFrameNum - _frameNumFront; ++k)
			++it;
		Mat bestImg = *it;

		//showImage("mask", mask, 0, 1);
		//showImage("bestImg", bestImg);
		//bitwise_and(inFrame, mask, targetImg);

		float sumR = 0, sumG = 0, sumB = 0;
		for(int y = 0; y < targetImg.rows; ++y){
			for(int x = 0; x < targetImg.cols; ++x){
				if(mask.at<uchar>(y,x) == 0)
					targetImg.at<Vec3b>(y,x) = Vec3b::all(128);
				else{
					targetImg.at<Vec3b>(y,x) = bestImg.at<Vec3b>(y,x);
					sumR += bestImg.at<Vec3b>(y,x)[0];
					sumG += bestImg.at<Vec3b>(y,x)[1];
					sumB += bestImg.at<Vec3b>(y,x)[2];
				}
			}
		}
		double r = sumR/(sumR+sumG+sumB), g = sumG/(sumR+sumG+sumB);
		
		// put target number
		vector<vector<Point>> contours = extractContours(mask);
		vector<Point> cc;
		int maxSize = 0;
		for(int j = 0; j < contours.size(); ++j){
			if(contours[j].size() > maxSize){
				maxSize = contours[j].size();
				cc = contours[j];
			}
		}

		double area = contourArea(cc);
		Rect box = boundingRect(cc);
		RotatedRect orientedBox = orientedBoundingBox(cc);
		double areaEllipse = PI * orientedBox.size.width * orientedBox.size.height / 4.0;
		double occupRate = area / areaEllipse;
		double cssCost0 = tr.getLastObject().getCSSCost0();
		double cssCost1 = tr.getLastObject().getCSSCost1();
		double cssCost2 = tr.getLastObject().getCSSCost2();

		// use a pre-trained decision tree (CART) to select fish targets
		
		// the decision tree is trained based on the occupancy rate, CSS cost with 3 template
		// fish objects and the average object color in normalized-RG space.
		if(cssCost0 < 39.8 && cssCost2 >= 47.125 && cssCost0 >= 34.2)
			continue;
		if(cssCost0 >= 39.8 && r < 0.275019)
			continue;
		if(area < 2500) continue;
		
		++_objCount;
		
		// put target parameters (length, width, etc.)
		float objLength = max(orientedBox.size.width, orientedBox.size.height);
		float objWidth = min(orientedBox.size.width, orientedBox.size.height);
		
		Point textPoint;
		if(box.tl().x + box.br().x >= param.getFrameWidth())
			textPoint = Point(20, 30);
		else
			textPoint = Point(param.getFrameWidth() / 2 - 20, 30);

		ostringstream sout;
		sout << "frame     " << outFrameNum;
		putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
		sout.str("");

		textPoint.y += 30; 
		sout << "object #  " << _objCount;
		putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
		sout.str("");
		
		if(param.getShowMetaData()){
			textPoint.y += 30; 
			sout << "length     " << setprecision(4) << objLength * 2;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "width      " << objWidth * 2;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "aspRatio   " << objLength / objWidth;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "area       " << setprecision(6) << area * 4;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "occup      " << setprecision(4) << occupRate * 100 << '%';
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "cssCost    " << cssCost0;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");
		
			textPoint.y += 30; 
			sout << "avgColor   R " << sumR/area;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "            G " << sumG/area;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");

			textPoint.y += 30; 
			sout << "            B " << sumB/area;
			putText(targetImg, sout.str(), textPoint, FONT_HERSHEY_PLAIN, 2, Scalar::all(0), 2);
			sout.str("");
		}
		// write to ARFF file for classifier training by WEKA
		//sprintf_s(filename, "%s\\fish.arff", param.getOutputPath());
		//ofstream fout(filename, ios::app);
		//fout << "@relation fish" << endl
		//	 << "@attribute 0 numeric" << endl
		//	 << "@attribute 1 numeric" << endl
		//	 << "@attribute 2 numeric" << endl
		//	 << "@attribute 3 numeric" << endl
		//	 << "@attribute class {0,1}" << endl
		//	 << "@data" << endl;
		//fout << param.getFrameNum() << ' ' << area*4 << ' ' << occupRate << ' ' 
		//	 << cssCost0 << ' ' << cssCost1 << ' ' << cssCost2 
		//	 << ' ' << r << ' ' << g << endl;

		//showImage("targetImg", targetImg);
		sprintf_s(filename, "%s\\frame%05d_obj%03d.jpg", param.getOutputPath(), outFrameNum, _objCount);
		imwrite(filename, targetImg);

		// write the corner coordinates to a CSV file
		// Ensure that the points are ordered clockwise starting in the lower left corner.
		//    It appears that minAreaRect assigns the first point to the point with the greatest y value
		//    (note that the image origin is in the UPPER LEFT CORNER). The points are always returned
		//    in clockwise order so we just need to determine the starting point.
		int k = (pts[0].x > pts[2].x) ? 1 : 0;

		csvOut << outFrameNum << ',' << _objCount << ','
			   << pts[k].x*2 << ',' << pts[k].y*2 << ',' << pts[k+1].x*2 << ',' << pts[k+1].y*2 << ',' 
			   << pts[k+2].x*2 << ',' << pts[k+2].y*2 << ',' << pts[(k+3)%4].x*2 << ',' << pts[(k+3)%4].y*2 << ','
			   << objLength * 2 << ',' << objWidth * 2 << ',' << objLength / objWidth << ','
			   << area * 4 << ',' << occupRate << ',' << cssCost0 << ','
			   << sumR/area << ',' << sumG/area << ',' << sumB/area
			   << endl;

		hasOutput = true;
	}

	// pop old frames from frame list
	int minStartFrame = param.getFrameNum();
	for(size_t i = 0; i < _trackList.size(); ++i){
		Track track = _trackList[i];
		int startFrame = track.getStartFrame();
		if(track.isActive() && startFrame < minStartFrame)
			minStartFrame = startFrame;
	}

	while(!_frameList.empty() && minStartFrame > _frameNumFront){
		_frameList.front().release();
		_frameList.pop_front();
		++_frameNumFront;
	}

	return hasOutput;
}

// track by Kalman filter
// state vector: [x, dx/dt, y, dy/dt]'
void Tracker::trackTargets(const vector<FgObject>& fgObjs)
{
    KalmanFilter KF(4, 4, 0);
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 1, 0, 0, 
			                                    0, 1, 0, 0, 
												0, 0, 1, 1,
												0, 0, 0, 1);
	setIdentity(KF.measurementMatrix, Scalar::all(1));
	setIdentity(KF.processNoiseCov, Scalar::all(1e-2));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

	vector<bool> matched;
	for(size_t i = 0; i < fgObjs.size(); ++i)
		matched.push_back(false);

	// data association by nearest-neighbor method
	for(size_t i = 0; i < _trackList.size(); ++i){
		if(!_trackList[i].isActive())
			continue;

		Point2f p = _trackList[i].getPosition();
		Point2f v = _trackList[i].getVelocity();
		KF.statePost = (Mat_<float>(4, 1) << p.x, v.x, p.y, v.y);

		Mat predictState = KF.predict();
		float predictPx = predictState.at<float>(0);
		float predictVx = predictState.at<float>(1);
		float predictPy = predictState.at<float>(2);
		float predictVy = predictState.at<float>(3);
		Point2f predictPosition (predictPx, predictPy);
		Point2f predictVelocity (predictVx, predictVy);

		double minDist = 1000;
		FgObject minObj;
		int idx = -1;
		for(size_t j = 0; j < fgObjs.size(); ++j){
			if(matched[j]) continue;
            Point2f center = fgObjs[j].center();
			double dist = norm(center - predictPosition);
			if(dist < 100 && dist < minDist){
				minDist = dist;
				minObj = fgObjs[j];
				idx = j;
			}
        }

		// if no observation is associated, target is either leaving FOV or lost temporarily
		if(idx == -1){
			Rect bbox = _trackList[i].boundingBox();
			bbox.x = predictPx + predictVx - bbox.width/2;
			bbox.y = predictPy + predictVy - bbox.height/2;
			bool inFOV = isInFOV(bbox, param.getVideoSourceType());

			if(!inFOV){    // target leaves FOV
				_trackList[i].setEndFrame(param.getFrameNum());
			}
			else{          // target is lost temporarily
				_trackList[i].setPosition(predictPosition + predictVelocity);
				_trackList[i].incrementLossCount();
			}
		}
		else{              // target is found
			_trackList[i].addObject(minObj);
			matched[idx] = true;

			Point2f pm = minObj.center();
			float vmx = pm.x - p.x;
			float vmy = pm.y - p.y;
			Point2f vm (vmx, vmy);
			Mat measurement = (Mat_<float>(4, 1) << pm.x, vm.x, pm.y, vm.y);
			
			Mat correctState = KF.correct(measurement);
			float correctPx = correctState.at<float>(0);
			float correctVx = correctState.at<float>(1);
			float correctPy = correctState.at<float>(2);
			float correctVy = correctState.at<float>(3);
			Point2f correctPosition (correctPx, correctPy);
			Point2f correctVelocity (correctVx, correctVy);
			_trackList[i].setPosition(correctPosition);
			_trackList[i].setVelocity(correctVelocity);
		}
	}

	for(size_t i = 0; i < matched.size(); ++i){
		if(!matched[i]){   // target enters FOV
			this->createTarget(fgObjs[i]);
		}
	}

}

