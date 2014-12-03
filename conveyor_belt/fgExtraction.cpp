#include "fgExtraction.h"
#include "util.h"
#include "parameters.h"

extern Parameters param;

FgExtraction::FgExtraction()
{
	_segmentation = new WatershedSegmentation;
	//_segmentation = new ThresholdSegmentation;
	//_segmentation = new CannyEdgeSegmentation;
	
    _fgUpdate = new HistBackprojectUpdate;
	//_fgUpdate = new MattingUpdate;
}

FgExtraction::~FgExtraction()
{
	delete _segmentation;
	delete _fgUpdate;
}

//***************************** FgExtraction ***********************************
int
FgExtraction::extractForeground(InputArray src, OutputArray dst, vector<FgObject>& fgObjs)
{
	if(!src.obj) return -1;
	Mat inImg = src.getMat();
	dst.create(inImg.size(), CV_8UC3);
	Mat outImg = dst.getMat();
	
	Mat fgMask;
	Mat newFgMask;
	Mat alphaMap;

	//**********************************************************************
	// perform image segmentation
	_segmentation->segmentObjects(inImg, fgMask);

	// crop the side regions of the belt...
    if(param.getVideoSourceType() == 1){
	    rectangle(fgMask, Point(0, 0), Point(45, fgMask.rows-1), Scalar::all(0), -1);
	    rectangle(fgMask, Point(675, 0), Point(fgMask.cols-1, fgMask.rows-1), Scalar::all(0), -1);
    }
	
	Mat grayFgMask;
	cvtColor(fgMask, grayFgMask, CV_RGB2GRAY);
	vector<vector<Point>> contours = extractContours(grayFgMask);

	if(contours.empty())
		return 0;
	
	//**********************************************************************
	// Look into the appearance of each object
	//showImage("fgMask", fgMask, 1);

	this->appearanceFiltering(fgMask, fgMask, contours, inImg);
	//showImage("newfgMask", fgMask, 1, 0);

		
	//**********************************************************************
	// refine object boundary by histogram backprojection	
	_fgUpdate->_inImg = inImg;
	_fgUpdate->_alphaMaps.clear();
	_fgUpdate->updateObject(fgMask, newFgMask);
	
	//showImage("fgMask", fgMask);
	//showImage("newFgMask", newFgMask);
	
	//**********************************************************************
	// postprocessing: open and close
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(param.getSESizeSeg(), param.getSESizeSeg()));
	morphologyEx(fgMask, fgMask, CV_MOP_OPEN, se);
	morphologyEx(fgMask, fgMask, CV_MOP_CLOSE, se);
    morphologyEx(fgMask, fgMask, CV_MOP_OPEN, se);
#ifndef VIDEO_OUTPUT
    //showImage("fgMaskFinal", fgMask, 1);
#endif
	
	fgMask.copyTo(outImg);

	contours.clear();
	cvtColor(fgMask, grayFgMask, CV_RGB2GRAY);
    contours = extractContours(grayFgMask);
	fgObjs.clear();
	for(size_t i = 0; i < contours.size(); ++i){
		Mat cssImg8U;
		cssImage(contours[i], cssImg8U);
		Mat se = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
		dilate(cssImg8U, cssImg8U, se);

		vector<vector<Point>> cssContours = extractContours(cssImg8U);
		vector<pair<double, int>> cssMax;
		for(int j = cssContours.size()-1; j >= 0; --j){
			Point pt = cssContours[j][0];
			double s = (199.0 - pt.y)*0.15 + 1.0;
			if(j == cssContours.size()-1){
				cssMax.push_back(make_pair(s, pt.x));
				continue;
			}

			if(s <= cssMax.front().first * param.getCSSMaxCoeff()) break;
			
			pair<double, int> back = cssMax.back();
			// calculate the midpoint of two branch peaks
			if(abs(back.first - s) < 0.15 && abs(back.second - pt.x) < 7){
				int midX = (pt.x + back.second)/2;
				cssMax.pop_back();
				cssMax.push_back(make_pair(s, midX));
			}
			else
				cssMax.push_back(make_pair(s, pt.x));
		}
		
		//assert(contours.size() == _fgUpdate->_alphaMaps.size());
		FgObject obj(contours[i], _fgUpdate->_alphaMaps[i]);

		double mc;
		mc = cssMatchingCost(cssMax, 0);
		obj.setCSSCost0(mc);
		mc = cssMatchingCost(cssMax, 1);
		obj.setCSSCost1(mc);
		mc = cssMatchingCost(cssMax, 2);
		obj.setCSSCost2(mc);

		fgObjs.push_back(obj);
	}

	return fgObjs.size();
}

void 
FgExtraction::appearanceFiltering(InputArray src, OutputArray dst, vector<vector<Point> > contours, Mat inFrame)
{
	if(!src.obj) return;
	Mat inImg = src.getMat();
	dst.create(inImg.size(), inImg.type());
	Mat outImg = dst.getMat();
	
	int count = 0;
	for(size_t i = 0; i < contours.size(); ++i)
	{			
		// drop the object if its area is too small
		double area = contourArea(contours[i]);
		// because the frame is down-sampled by 2, the area is actually 1/4 of the original one
		if(area * 4 < param.getMinArea() || area * 4 > param.getMaxArea()){
			drawContours(outImg, contours, i, Scalar::all(0), -1);
			continue;
		}

		// drop the object if it is not fully within FOV
		Rect boundBox = boundingRect(contours[i]);
		bool inFOV = isInFOV(boundBox, param.getVideoSourceType());
		if(!inFOV){
			drawContours(outImg, contours, i, Scalar::all(0), -1);
			continue;
		}

		// drop the object if its aspect ratio is not in the reasonable range
		RotatedRect orientBox = orientedBoundingBox(contours[i]);
		double aspRatio = orientBox.size.width / orientBox.size.height;
		aspRatio = max(aspRatio, 1.0/aspRatio);
		if (aspRatio < param.getMinAspRatio() || aspRatio > param.getMaxAspRatio()){
			drawContours(outImg, contours, i, Scalar::all(0), -1);
			continue;
		}

		
	}
}

void FgExtraction::localBinaryPatterns(InputArray src, OutputArray dst, int radius, Mat mask)
{
	if(!src.obj) return;
    Mat img = src.getMat();
	dst.create(img.size(), CV_8UC3);
	Mat outImg = dst.getMat();
    
	Mat grayImg;
	cvtColor(img, grayImg, CV_RGB2GRAY);
	
	for(int y = radius; y < grayImg.rows - radius; ++y){
		for(int x = radius; x < grayImg.cols - radius; ++x){
			Point cen(x, y);
			uchar cenVal = grayImg.at<uchar>(cen);
			double lbp[8];

			if(mask.at<uchar>(cen) == 0){
				outImg.at<Vec3b>(cen) = Vec3b::all(0);
				continue;
			}

			for(int t = 0; t < 8; ++t){
				float xx = cen.x + radius*cos(t*PI/4.0);
				float yy = cen.y + radius*sin(t*PI/4.0);

				float a = xx - floor(xx);
				float b = yy - floor(yy);

				uchar ptVal = uchar((1-a) * (1-b) * img.at<uchar>(int(floor(yy)), int(floor(xx)))
								  +     a * (1-b) * img.at<uchar>(int(floor(yy)), int(ceil(xx)))
							      + (1-a) *     b * img.at<uchar>(int(ceil(yy)), int(floor(xx)))
							      +     a *     b * img.at<uchar>(int(ceil(yy)), int(ceil(xx))) );

				if (ptVal > cenVal - 3)
				    //lbp += 1 << (7-t);
				    lbp[t] = 1.0;
				else
					lbp[t] = 0.0;
			}
			
			//outImg.at<Vec3b>(cen) = Vec3b::all(lbp);

			// measure proximity by histogram intersection
			double histIntersect = 0.0;
			for(int t = 0; t < 8; ++t){
				histIntersect += min(lbp[t], 0.9);
			}

			
			if(histIntersect > 3.6)
				outImg.at<Vec3b>(cen) = Vec3b::all(255);
			else
				outImg.at<Vec3b>(cen) = Vec3b::all(0);
		}
	}
}

void
FgExtraction::cssImage(vector<Point> contour, OutputArray cssImg)
{
	int nP = 200;
	int nS = 200;
	double step = 0.15;
	int kw = 4;

	cssImg.create(nS, nP, CV_8U);
	Mat outImg8U = cssImg.getMat();
	outImg8U.setTo(Scalar(0));

	vector<int> x, y;
	x.reserve(nP);
	y.reserve(nP);
	size_t size = contour.size();
	for(int u = 0; u < nP; ++u){
		x.push_back(contour[u*size/nP].x);
		y.push_back(contour[u*size/nP].y);
	}

	int r = nS-1;
	for(double s = 1.0; s < 1.14+(nS-1)*step; s+=step){
		// 1D gaussian kernel
		vector<double> g, gu, guu;
		for(int v = 0; v <= 2*kw*s; ++v){
			double G = exp(-0.5 * pow((v-kw*s)/s, 2)) / sqrt(2*PI) / s;
			g.push_back(G);
			gu.push_back(-(v-kw*s)/pow(s, 2) * G);
			guu.push_back((-pow(s, 2) + pow(v-kw*s, 2)) / pow(s, 4) * G);
		}

		// convolution and calculate curvature
		vector<double> X, Xu, Xuu, Y, Yu, Yuu, k;
		vector<bool> dyLarge;
		X.reserve(nP);
		Xu.reserve(nP);
		Xuu.reserve(nP);
		Y.reserve(nP);
		Yu.reserve(nP);
		Yuu.reserve(nP);
		k.reserve(nP);
		dyLarge.reserve(nP);
		for(int i = 0; i < nP; ++i){
			X.push_back(0);
			Xu.push_back(0);
			Xuu.push_back(0);
			Y.push_back(0);
			Yu.push_back(0);
			Yuu.push_back(0);
			dyLarge.push_back(true);
			for(int j = 0; j <= 2*kw*s; ++j){
				int idx = i-j+kw*s;
				idx = idx < 0 ? idx + nP : (idx >= nP ? idx - nP : idx);
				X[i]   += x[idx] * g[j];
				Xu[i]  += x[idx] * gu[j];
				Xuu[i] += x[idx] * guu[j];
				Y[i]   += y[idx] * g[j];
				Yu[i]  += y[idx] * gu[j];
				Yuu[i] += y[idx] * guu[j];
			}
			double ki = (Xu[i]*Yuu[i] - Xuu[i]*Yu[i]) / pow((Xu[i]*Xu[i] + Yu[i]*Yu[i]), 1.5);
			k.push_back(ki);
		}

		/*for(int u = 1; u < nP-1; ++u){
			if(abs(Y[u] - Y[u-1]) < 0.5 && abs(Y[u] - Y[u+1]) < 0.5){
				for(int v = -1; v <= 1; ++v){
					int idx = u+v < 0 ? 0 : (u+v >= nP ? nP-1 : u+v);
					dyLarge[idx] = false;
				}
			}
		}*/

		for(int u = 0; u < nP; ++u){
			if(k[u]*k[(u+1)%nP] < 0)
				outImg8U.at<uchar>(r, u) = 255;
			else
				outImg8U.at<uchar>(r, u) = 0;
		}
		r--;

		/*if(r % 10 == 9){
			vector<Point> cc;
			cc.reserve(200);
			for(int i = 0; i < nP; ++i)
				cc.push_back(Point(X[i], Y[i]));

			vector<vector<Point>> ccs;
			ccs.push_back(cc);
			Mat smoothContour = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
			drawContours(smoothContour, ccs, 0, Scalar(255), -1);
			showImage("smoothContour", smoothContour, 1, 1);
			showImage("cssImg", outImg8U, 1);
			int a = 1;
		}*/
		
	}

	/*Mat se = getStructuringElement(MORPH_RECT, Size(1, 3));
	morphologyEx(outImg8U, outImg8U, CV_MOP_CLOSE, se);
	morphologyEx(outImg8U, outImg8U, CV_MOP_OPEN, se);*/

	/*vector<vector<Point> > cssContours = extractContours(outImg8U);
	for(size_t i = 0; i < cssContours.size(); ++i){
		Rect bRect = boundingRect(cssContours[i]);
		if(bRect.tl().y < 2 && bRect.height/(double)bRect.width > 4.0)
			drawContours(outImg8U, cssContours, i, Scalar(0), -1);
	}*/
	//showImage("CSS", outImg8U, 1, 0);
}

double
FgExtraction::cssMatchingCost(const vector<pair<double, int>>& cssMax, int modelNum)
{
	vector<pair<double, int> > cssMaxRef;
	switch(modelNum){
		case 0:
			// object 0, frame 481
			cssMaxRef.push_back(make_pair(17.5, 39));
			cssMaxRef.push_back(make_pair(15.1, 183));
			cssMaxRef.push_back(make_pair(7.45, 80));
			cssMaxRef.push_back(make_pair(6.55, 130));
			cssMaxRef.push_back(make_pair(5.65, 10));
			break;

		case 1:
			// object 2, frame 462
			cssMaxRef.push_back(make_pair(16.6, 133));
			cssMaxRef.push_back(make_pair(15.55, 65));
			cssMaxRef.push_back(make_pair(7.75, 96));
			cssMaxRef.push_back(make_pair(6.85, 26));
			cssMaxRef.push_back(make_pair(5.35, 62));
			cssMaxRef.push_back(make_pair(3.55, 41));
			cssMaxRef.push_back(make_pair(3.55, 137));
			break;

		case 2:
			// object 2, frame 505
			cssMaxRef.push_back(make_pair(17.35, 174));
			cssMaxRef.push_back(make_pair(15.25, 34));
			cssMaxRef.push_back(make_pair(8.2, 71));
			cssMaxRef.push_back(make_pair(5.05, 78));
			cssMaxRef.push_back(make_pair(4.75, 168));
			cssMaxRef.push_back(make_pair(4.0, 131));
			cssMaxRef.push_back(make_pair(3.85, 6));
			cssMaxRef.push_back(make_pair(3.85, 112));
			break;

		default:
			return -1;
			break;
	}

	double cost = abs(cssMax[0].first - cssMaxRef[0].first);
	int shift = cssMax[0].second - cssMaxRef[0].second;
	for(size_t j = 1; j < cssMax.size(); ++j){
		if(j >= cssMaxRef.size() || abs(cssMax[j].second - cssMaxRef[j].second - shift) > 0.2*100)
			cost += cssMax[j].first;
		else
			cost += abs(cssMax[j].first - cssMaxRef[j].first);
	}
	if(cssMax.size() < cssMaxRef.size()){
		for(size_t k = 0; k < cssMaxRef.size() - cssMax.size(); ++k)
			cost += cssMaxRef[k + cssMax.size()].first;
	}

	return cost;
}


//*********************** WatershedSegmentation ********************************
void
WatershedSegmentation::segmentObjects(InputArray src, OutputArray dst)
{
	if(!src.obj) return;
    Mat img = src.getMat();
    dst.create(img.size(), CV_8UC3);
	Mat outImg = dst.getMat();
	Mat fgMask, markers;

	//char filename [256];

	// segmentation by threshold
	_threshSegment->segmentObjects(img, fgMask);
	//showImage("threshold", fgMask, 1, 1);
	
	// generate markers by using distance transform
	int markersCount = generateMarkers(fgMask, markers);
	
	Mat markersMask;
	markers.convertTo(markersMask, CV_8UC3, 256);
	//showImage("markers", markersMask, 1, 1);
	
	watershed(img, markers);
	for(int y = 0; y < markers.rows; ++y){
		for(int x = 0; x < markers.cols; ++x){
			int idx = markers.at<int>(y, x);
			if(idx == -1)
				outImg.at<Vec3b>(y,x) = Vec3b::all(255);
			else if(idx <= 0 || idx > markersCount)
				outImg.at<Vec3b>(y,x) = Vec3b::all(0);
			else
				outImg.at<Vec3b>(y,x) = Vec3b::all(0);
		}
	}
	
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(outImg, outImg, se);
	//showImage("watershed", outImg, 1, 1);
	//sprintf_s(filename, "%s\\e.jpg", param.getOutputPath());
	//imwrite(filename, outImg);

	// put markers on input image
	/*img.setTo(Scalar(0,255,0), markersMask);
	//showImage("img with markers", imgMarked);
	markersMask.release();*/

	outImg = fgMask - outImg;
	morphologyEx(outImg, outImg, CV_MOP_OPEN, se);

	//showImage("after", outImg, 1, 0);
	//sprintf_s(filename, "%s\\f.jpg", param.getOutputPath());
	//imwrite(filename, outImg);

}

int
WatershedSegmentation::generateMarkers(InputArray src, OutputArray dst)
{
    if(!src.obj) return 0;
	Mat img = src.getMat();
	dst.create(img.size(), CV_32SC1);
    Mat outImg = dst.getMat();
	Mat binImg, distMap, dtImg;
    
    // convert to grayscale
    cvtColor(img, binImg, CV_RGB2GRAY);
	//showImage("binImg", binImg, 1, 1);
	
	// perform distance transform
	distanceTransform(binImg, distMap, CV_DIST_L2, 5);
	//distanceTransform(binImg, distMap, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	normalize(distMap, dtImg, 0, 1, NORM_MINMAX);
	//showImage("DT", dtImg, 1);
	//char filename [256];
	//sprintf_s(filename, "%s\\c.jpg", param.getOutputPath());
	//imwrite(filename, dtImg);

	vector<vector<Point> > contours = extractContours(binImg);
	Mat oneObjMask = Mat::zeros(distMap.size(), CV_32F);
	Mat oneObjDistMap = Mat::zeros(distMap.size(), distMap.type());
	Mat oneObjDtImg = Mat::zeros(dtImg.size(), dtImg.type());
	Mat threshDtImg = Mat::zeros(dtImg.size(), dtImg.type());
	
	int count = 0;

	for(size_t i = 0; i < contours.size(); ++i){
		int delta = 10;

		drawContours(oneObjMask, contours, i, Scalar::all(1), -1);
		
		oneObjDistMap = distMap.mul(oneObjMask);
		normalize(oneObjDistMap, oneObjDtImg, 0, 1, NORM_MINMAX);
		oneObjDtImg.convertTo(oneObjDtImg, CV_8U, 255);
		
		// thresholding
		double maxVal;
		minMaxLoc(oneObjDtImg, NULL, &maxVal, NULL, NULL);
		int th = int(maxVal * 0.72);
		threshold(oneObjDtImg, threshDtImg, th, 255, THRESH_BINARY);
		//showImage("threshDt", threshDtImg, 1);

		// h-maxima transform
		/*		
		oneObjDistMap = distMap.mul(oneObjMask);
		normalize(oneObjDistMap, dtImg, 0, 1, NORM_MINMAX);
		dtImg.convertTo(dtImg, CV_8U, 255);


		int h = 1;
		vector<Point> localMax = hMax(dtImg, h, oneObjMask);*/
		
		/*
		threshold(dtImg, threshDtImg, h, 255, CV_THRESH_BINARY);
		vector<vector<Point> > cc = extractContours(threshDtImg);
		
		int ncc = cc.size();
		if(cc.size() > 1){
			while(cc.size() >= ncc && h >= 0){
				ncc = cc.size();
				h -= delta;
				threshold(dtImg, threshDtImg, h, 255, CV_THRESH_BINARY);
				cc = extractContours(threshDtImg);

				cout << "h = " << h << endl;
				showImage("threshDt", threshDtImg, 1);
			}
			h = min(h + delta, 255); 
		}
		threshold(dtImg, threshDtImg, h, 255, CV_THRESH_BINARY);
		*/
		

		vector<vector<Point> > cc = extractContours(threshDtImg);
		for(size_t j = 0; j < cc.size(); ++j){
			drawContours(outImg, cc, j, Scalar::all(++count), -1);
		}
		
		//drawContours(threshDtImg, contours, i, Scalar::all(255), 1);
		/*for(int j = 0; j < localMax.size(); ++j)
			circle(threshDtImg, localMax[j], 1, Scalar::all(255), -1 );*/

		//showImage("DT", threshDtImg, 1);
		
		drawContours(threshDtImg, contours, i, Scalar::all(0), -1);
		drawContours(oneObjMask, contours, i, Scalar::all(0), -1);
	}

	// markers manually put for the sidebars
	circle(outImg, Point(20, outImg.rows/2), 5, Scalar::all(++count), -1);
	circle(outImg, Point(outImg.cols-20, outImg.rows/2), 5, Scalar::all(++count), -1);

	// marker manually put for bg
	bool bg = false;
	int x = 0, y = 0;
	while(!bg){
		x = int(rng.uniform(100, outImg.cols-100) + 0.5);
		y = int(rng.uniform(100, outImg.rows-100) + 0.5);
		bg = binImg.at<uchar>(y, x) == 0;
	}
	circle(outImg, Point(x, y), 5, Scalar::all(++count), -1);
	
	/*Mat tempImg;
	outImg.convertTo(tempImg, CV_8U, 10);
	showImage("outimg", tempImg);*/

	//cvtColor(outImg, outImg, CV_GRAY2RGB);
    
	return count;
}

int
WatershedSegmentation::generateMarkers2(InputArray src, OutputArray dst)
{
    if(!src.obj) return 0;
	Mat img = src.getMat();
    dst.create(img.size(), CV_32SC1);
	Mat outImg = dst.getMat();
    
	// convert to grayscale
    Mat binImg;
	cvtColor(img, binImg, CV_RGB2GRAY);
	//showImage("binImg", binImg, 1, 1);
	
	// perform distance transform
	Mat distMap;
	Mat dtImg;
	distanceTransform(binImg, distMap, CV_DIST_L2, 3);
	normalize(distMap, dtImg, 0, 1, NORM_MINMAX);
	//showImage("DT", dtImg, 1);
	
	// find local maxima by morphological dilation
	Mat dilateDistMap;
	Mat dilateDtImg;
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(distMap, dilateDistMap, se);
	normalize(dilateDistMap, dilateDtImg, 0, 1, NORM_MINMAX);
	//showImage("dilateDT", temp, 1);

	Mat subDistMap = dilateDistMap - distMap;
	Mat subImg;
	normalize(subDistMap, subImg, 0, 1, NORM_MINMAX);
	//showImage("subImg", subImg, 1, 1);

	//Mat maskImg = src.getMat();
	//maskImg.convertTo(maskImg, CV_8U, 255);
	Mat maskImg;
	erode(binImg, maskImg, se);
	//showImage("maskImg", maskImg, 1);
	Mat temp;
	subtract(Mat::ones(subImg.size(), subImg.type()), subImg, temp, maskImg);
	//showImage("subImg", temp, 1, 1);

	Mat subImgThresh;
	threshold(temp, subImgThresh, 0.5, 1, THRESH_BINARY);
	morphologyEx(subImgThresh, subImgThresh, MORPH_CLOSE, se);
	//showImage("subImgThresh", subImgThresh, 1, 1);

	subImgThresh.convertTo(temp, CV_8U);
	vector<vector<Point> > cc = extractContours(temp);
	temp = Mat::zeros(temp.size(), CV_8U);
	int count = 0;
	for(size_t j = 0; j < cc.size(); ++j){
		double area = contourArea(cc[j]);
		if(area > 50){
			drawContours(outImg, cc, j, Scalar::all(++count), -1);
			drawContours(temp, cc, j, Scalar::all(255), -1);
		}
	}

	// markers manually put for the sidebars
	circle(outImg, Point(20, outImg.rows/2), 5, Scalar::all(++count), -1);
	circle(temp, Point(20, outImg.rows/2), 5, Scalar::all(255), -1);
	circle(outImg, Point(outImg.cols-20, outImg.rows/2), 5, Scalar::all(++count), -1);
	circle(temp, Point(outImg.cols-20, outImg.rows/2), 5, Scalar::all(255), -1);

	// marker manually put for bg
	bool bg = false;
	int x = 0, y = 0;
	while(!bg){
		x = int(rng.uniform(100, outImg.cols-100) + 0.5);
		y = int(rng.uniform(100, outImg.rows-100) + 0.5);
		bg = binImg.at<uchar>(y, x) == 0;
	}
	circle(outImg, Point(x, y), 5, Scalar::all(++count), -1);
	circle(temp, Point(x, y), 5, Scalar::all(255), -1);

	//showImage("temp", temp, 1);

	vector<vector<Point> > contours = extractContours(binImg);
	
	return count;
}

vector<Point>
WatershedSegmentation::hMax(InputArray src, int h, InputArray mask)
{
	vector<Point> localMax;
	if(!src.obj) return localMax;

	Mat inImg = src.getMat();
    Mat dilateImg, subImg;
	//showImage("subImg", inImg, 1);

	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(inImg, dilateImg, se, Point(1,1));
	//showImage("subImg", dilateImg, 1);

	subImg = dilateImg - inImg;
	//showImage("subImg", subImg, 1);

	Mat maskImg = mask.getMat();
	maskImg.convertTo(maskImg, CV_8U, 255);
	bitwise_not(subImg, subImg, maskImg);
	//showImage("subImg", subImg, 1);

	threshold(subImg, subImg, 254, 255, THRESH_BINARY);
	//showImage("subImg", subImg, 1);
	
	vector<vector<Point> > cc = extractContours(subImg);
	
	for(size_t i = 0; i < cc.size(); ++i){
		Point cen = cc[i][0];
		uchar maxVal = 0;
		for(int y = cen.y-1; y <= cen.y+1; ++y){
			for(int x = cen.x-1; x <= cen.x+1; ++x){
				if(x == cen.x && y == cen.y) 
					continue;
				if(inImg.at<uchar>(y,x) > maxVal)
					maxVal = inImg.at<uchar>(y,x);
			}
		}

		uchar cenVal = inImg.at<uchar>(cen);
		if(cenVal - maxVal >= h)
			localMax.push_back(cen);
	}
	return localMax;
}


//*********************** ThresholdSegmentation ********************************
void
ThresholdSegmentation::segmentObjects(InputArray src, OutputArray dst)
{
    if(!src.obj) return;
	Mat img = src.getMat();
	dst.create(img.size(), img.type());
    Mat outImg = dst.getMat();
    Mat grayImg, blurredImg, fgMask;
    
    // convert to grayscale and apply Gaussian filter
    cvtColor(img, grayImg, CV_RGB2GRAY);
	GaussianBlur(grayImg, blurredImg, Size(0, 0), 5);
	//blurredImg = grayImg;
    //showImage("blurred", blurredImg, 1, 1);

    // thresholding
    //threshold(blurredImg, fgMask, 50, 255, THRESH_BINARY_INV | THRESH_OTSU);
	int u = 0;
	int thresh = getOtsuThreshold(blurredImg, 1, 255, &u);
	thresh += int(param.getThreshAdjustCoeff() * (u - thresh));
	thresh = thresh < param.getMinThreshold() ? param.getMinThreshold() : (thresh > param.getMaxThreshold() ? param.getMaxThreshold() : thresh);
    
	// performs thresholding by using a look-up table (much faster)
	Mat lut(1, 256, CV_8U);
	for(int i = 0; i < 256; ++i)
		lut.data[i] = i >= thresh ? 0 : 255;
	LUT(blurredImg, lut, fgMask);
	//showImage("thresh", fgMask, 1, 1);

	if(param.getVideoSourceType() == 1){
		// crop the side regions of the belt...
		rectangle(fgMask, Point(0, 0), Point(45, fgMask.rows-1), Scalar::all(0), -1);
		rectangle(fgMask, Point(675, 0), Point(fgMask.cols-1, fgMask.rows-1), Scalar::all(0), -1);
	}
	//showImage("before hole filling", fgMask, 1, 1);

	// fill in holes inside connected components
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat tempImg = fgMask.clone();
	findContours(tempImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point());
	tempImg.release();
	for(size_t i = 0; i < contours.size(); ++i){
		if(hierarchy[i][3] >= 0)
			drawContours(fgMask, contours, i, Scalar(255), -1);
	}

	//showImage("after hole filling", fgMask, 1);
        
	Mat se = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(blurredImg, blurredImg, CV_MOP_OPEN, se);
	
	cvtColor(fgMask, outImg, CV_GRAY2RGB);
}

/*******************************************************************************
* Function:      getOtsuThreshold  
* Description:   computes the threhsold using Otsu's method
* Arguments:
	lowerVal      -   lower bound of pixel value
	upperVal      -   upper bound of pixel value
	u2Ptr         -   pointer to receive the mean of class 2
	roiMask       -   ROI binary mask
	
* Returns:       int - Otsu threshold
*******************************************************************************/
inline int 
ThresholdSegmentation::getOtsuThreshold(InputArray inImg, int lowerVal, int upperVal, int* u2Ptr, InputArray roiMask)
{
	Mat _inImg = inImg.getMat();
	Mat _roiMask = Mat::ones(_inImg.size(), _inImg.type());
	// Mat _roiMask = roiMask.getMat();

	int channels[] = {0};
	int nbins = 256;
    const int histSize[] = {nbins};
    float range[] = {0, 255};
    const float* ranges[] = {range};
	Mat hist;
    cv::calcHist(&_inImg, 1, channels, roiMask, hist, 1, histSize, ranges);
	
	Mat_<float> hist_(hist);
	float size = float(sum(hist)[0]);

	float w1, w2, u1, u2;
  	float max = -1;
	int index = 1;
	float u2max = -1;
	float histMax = 0;
	int mode = 0;
	float count = 0;

	for (int i = lowerVal+1; i < upperVal; ++i){	
		if(hist_(i,0) > histMax) {
			histMax = hist_(i,0);
			mode = i;
		}
		w1 = 0;
		
		for (int j = lowerVal+1; j <= i; ++j){
			w1 = w1 + hist_(j-1,0);
		}
		w1 = w1 / size;
		w2 = 1 - w1;

		u1 = 0;
		count = 0;
		for (int j = lowerVal; j <= i-1; ++j){
			u1 = u1 + j*hist_(j,0);
			count += hist_(j,0);
		}
		u1 /= count;

		u2 = 0;
		count = 0;
		for (int j = i; j <= upperVal; ++j){
			u2 = u2 + j*hist_(j, 0);
			count += hist_(j, 0);
		}
		u2 /= count;

		if (w1 * w2 * (u1-u2) * (u1-u2) > max){
			max = w1 * w2 * (u1-u2) * (u1-u2);
			index = i;
			u2max = u2;
		}
		else{
			max = max;
			index = index;
		}
	}
	
	//cout << "mode = " << mode << endl;
	//cout << "u1 = " << u1max << "; index = " << index << "; ";
	
	*u2Ptr = (int)(u2max + 0.5);
	return index;
}

int
ThresholdSegmentation::getThreshold(InputArray src)
{
	Mat img = src.getMat();

	// get threshold value by Otsu's method
	int u = 0;
	int thresh = getOtsuThreshold(img, 1, 255, &u);

	return thresh;
}


//*********************** CannyEdgeSegmentation ********************************
void
CannyEdgeSegmentation::segmentObjects(InputArray src, OutputArray dst, bool threeChannels)
{
    if(!src.obj) return;
    Mat img = src.getMat();
	dst.create(img.size(), img.type());
    Mat outImg = dst.getMat();
    Mat blurredImg [3];
    Mat edgeImg [3];
    int seSize = 5;

    // split RGB channels
    Mat mv [3];
    split(img, mv);

    for(int i = 0; i < 3; ++i){
        // apply Gaussian filter
        GaussianBlur(mv[i], blurredImg[i], Size(0, 0), 5);
        //showImage("blurred", blurredImg, 1);

        // Canny edge detection
        Canny(blurredImg[i], edgeImg[i], 15, 10);
        
        // close and dilate
        Mat se = getStructuringElement(MORPH_ELLIPSE, Size(seSize, seSize));
	    morphologyEx(edgeImg[i], edgeImg[i], CV_MOP_CLOSE, se);

	    Mat dilateSE = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	    dilate(edgeImg[i], edgeImg[i], dilateSE);
        //showImage("edge", edgeImg, 1);
    }

    showImage("edgeR", edgeImg[0], 1, 1);
    showImage("edgeG", edgeImg[0], 1, 1);
    showImage("edgeB", edgeImg[0], 1);

    cvtColor(edgeImg[0], outImg, CV_GRAY2RGB);
}

void
CannyEdgeSegmentation::segmentObjects(InputArray src, OutputArray dst)
{
	// convert to grayscale
    if(!src.obj) return;
    Mat img = src.getMat();
	dst.create(img.size(), img.type());
    Mat outImg = dst.getMat();
    Mat blurredImg;
    Mat edgeImg;
    int seSize = 5;

    // convert to grayscale and apply Gaussian filter
    cvtColor(img, blurredImg, CV_RGB2GRAY);
    GaussianBlur(blurredImg, blurredImg, Size(0, 0), 5);
    //showImage("blurred", blurredImg, 1);

    // Canny edge detection
    Canny(blurredImg, edgeImg, 15, 5);
        
    // close and dilate
    Mat se = getStructuringElement(MORPH_ELLIPSE, Size(seSize, seSize));
	morphologyEx(edgeImg, edgeImg, CV_MOP_CLOSE, se);

	Mat dilateSE = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	dilate(edgeImg, edgeImg, dilateSE);
    //showImage("edge", edgeImg, 1);

    cvtColor(edgeImg, outImg, CV_GRAY2RGB);
}

//************************ GradientSegmentation ********************************
void
GradientSegmentation::segmentObjects(InputArray src, OutputArray dst)
{
    if(!src.obj) return;
	Mat img = src.getMat();
	dst.create(img.size(), img.type());
    Mat outImg = dst.getMat();
    Mat blurredImg [3];
    Mat edgeImg [3];
    int seSize = 5;

    // split RGB channels
    Mat mv [3];
    split(img, mv);

    for(int i = 0; i < 3; ++i){
        // apply Gaussian filter
        GaussianBlur(mv[i], blurredImg[i], Size(0, 0), 5);
        //showImage("blurred", blurredImg, 1);

        // get gradient by Sobel operator
        Sobel(blurredImg[i], edgeImg[i], -1, 1, 1, 5, 2.0);
        
        // close and dilate
        Mat se = getStructuringElement(MORPH_ELLIPSE, Size(seSize, seSize));
	    morphologyEx(edgeImg[i], edgeImg[i], CV_MOP_CLOSE, se);

	    Mat dilateSE = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	    dilate(edgeImg[i], edgeImg[i], dilateSE, Point(5/2, 5/2));
        //showImage("edge", edgeImg, 1);
    }

    showImage("edgeR", edgeImg[0], 1, 1);
    showImage("edgeG", edgeImg[0], 1, 1);
    showImage("edgeB", edgeImg[0], 1);

    cvtColor(edgeImg[0], outImg, CV_GRAY2RGB);
}
