#include "fgExtraction.h"
#include "util.h"
#include "parameters.h"
#include "fgUpdate.h"

extern Parameters param;


//****************************** FgUpdate *************************************
FgUpdate::FgUpdate()
{

}

FgUpdate::FgUpdate(Mat inImg) :
	_inImg(inImg)
{
	_alphaMaps.reserve(16);
}

//************************ HistBackprojectUpdate ******************************
HistBackprojectUpdate::HistBackprojectUpdate()
{
	_ratioHistBP = Mat::zeros(_inImg.size(), CV_32F);
}

HistBackprojectUpdate::~HistBackprojectUpdate()
{

}

void
HistBackprojectUpdate::updateObject(InputArray src, OutputArray dst)
{
	if(!src.obj) return;
	Mat fgMask = src.getMat();
	dst.create(fgMask.size(), fgMask.type());
	Mat newFgMask = dst.getMat();
	
	int seSize = 25;
	double theta = param.getThetaHistBP();
	
	Mat grayFgMask;
	cvtColor(src, grayFgMask, CV_RGB2GRAY);
	vector<vector<Point> > contours;
	contours = extractContours(grayFgMask);
		
	Mat roiMask = Mat::zeros(_inImg.size(), CV_8U);
	Mat highMask = Mat::zeros(_inImg.size(), CV_8U);
	Mat lowMask = Mat::zeros(_inImg.size(), CV_8U);
	Mat grayNewFgMask = Mat::zeros(_inImg.size(), CV_8U);

	for(size_t i = 0; i < contours.size(); ++i){
		// generate a binary mask indicating the ROI
		RotatedRect orientedBox = minAreaRect(contours[i]);
		orientedBox.size.width *= 1.2f;
        orientedBox.size.height *= 1.2f;
		ellipse(roiMask, orientedBox, Scalar(255), -1);
		
		// a dilated version of foreground object
		bitwise_and(grayFgMask, roiMask, highMask);
		Mat se = getStructuringElement(CV_SHAPE_ELLIPSE, Size(seSize, seSize));
		dilate(highMask, lowMask, se);

		Mat highImg, lowImg;
		_inImg.copyTo(highImg, highMask);
		_inImg.copyTo(lowImg, lowMask);
		//showImage("highImg", highImg, 0);
		//showImage("lowImg", lowImg, 0);

		// generate histograms of two foregrounds
		int channels[] = {0, 1, 2};
		int nbins = 16;
		const int histSize[] = {nbins, nbins, nbins};
		float range[] = {0, 255};
		const float* ranges[] = {range, range, range};
		Mat highHist;
		Mat lowHist;
		calcHist(&_inImg, 1, channels, highMask, highHist, 3, histSize, ranges);
		calcHist(&_inImg, 1, channels, lowMask, lowHist, 3, histSize, ranges);

		//cout << highHist << endl;
		//cout << lowHist << endl;

		// get the ratio histogram
		Mat ratioHist = highHist / lowHist;
		
		// backproject the ratio histogram to image plane
		_ratioHistBP = histBackProject(ratioHist, 256/nbins, roiMask);
		Mat alpha(_ratioHistBP.size(), _ratioHistBP.type());
		bitwise_and(_ratioHistBP, _ratioHistBP, alpha, lowMask);
		_alphaMaps.push_back(alpha);
		//showImage("alpha", alpha, 1);

		// thresholding on the backprojection
		threshold(_ratioHistBP, _ratioHistBP, theta, 255, THRESH_BINARY);
		Mat histBP_8U;
		_ratioHistBP.convertTo(histBP_8U, CV_8U);
		
		bitwise_or(histBP_8U, grayNewFgMask, grayNewFgMask, roiMask);
		ellipse(roiMask, orientedBox, Scalar(0), -1);
		ellipse(highMask, orientedBox, Scalar(0), -1);
		ellipse(lowMask, orientedBox, Scalar(0), -1);
	}

	cvtColor(grayNewFgMask, newFgMask, CV_GRAY2RGB);

	//showImage("Fg", fgMask, 1, 1);
	//showImage("newFg", newFgMask, 1);
}

Mat
HistBackprojectUpdate::histBackProject(InputArray hist, int binWidth, Mat roiMask)
{
	if(!hist.obj) return Mat();
	Mat histMat = hist.getMat();
	
	Mat backProj(_inImg.size(), CV_32F);

	for(int y = 0; y < _inImg.rows; ++y){
		for(int x = 0; x < _inImg.cols; ++x){
			if(roiMask.at<uchar>(y, x) > 0){
				Vec3b pixel = _inImg.at<Vec3b>(y, x);
				float h = histMat.at<float>(pixel[0]/binWidth, pixel[1]/binWidth, pixel[2]/binWidth);
				backProj.at<float>(y, x) = h < 1.0f ? h : 1.0f;
			}
		}
	}
	
	return backProj;
}

//**************************** MattingUpdate **********************************
MattingUpdate::MattingUpdate()
{
	
}

MattingUpdate::~MattingUpdate()
{

}

void
MattingUpdate::updateObject(InputArray src, OutputArray dst)
{
	if(!src.obj) return;
	Mat fgMask = src.getMat();
	dst.create(fgMask.size(), fgMask.type());
	Mat newFgMask = dst.getMat();
	
	int seSize = 5;
	double theta = 0.3;
	
	Mat grayFgMask;
	cvtColor(src, grayFgMask, CV_RGB2GRAY);
	vector<vector<Point>> contours;
	contours = extractContours(grayFgMask);
		
	Mat objFgMask = Mat::zeros(_inImg.size(), CV_8U);
	Mat grayNewFgMask = Mat::zeros(_inImg.size(), CV_8U);

	for(size_t i = 0; i < contours.size(); ++i){
		// generate a binary mask indicating the ROI
		Rect box = boundingRect(contours[i]);
		box.x = int(max(box.x - box.width*0.1, 0.0));
		box.y = int(max(box.y - box.height*0.1, 0.0));
		box.width *= 1.2;
		box.height *= 1.2;
        Mat roiImg = _inImg(box);
		Mat roiMask = fgMask(box);
		
		// calculate alpha value within the ROI
        Mat alphaMap;
        findOptimalAlphaMap(roiImg, roiMask, alphaMap);
		
		//bitwise_or(histBP_8U, grayNewFgMask, grayNewFgMask, roiMask);
		//ellipse(roiMask, orientedBox, Scalar(0), -1);
		//ellipse(objFgMask, orientedBox, Scalar(0), -1);
		
	}

	cvtColor(grayNewFgMask, newFgMask, CV_GRAY2RGB);

	//showImage("Fg", fgMask, 1, 1);
	//showImage("newFg", newFgMask, 1);
}

void MattingUpdate::findOptimalAlphaMap( InputArray src, InputArray srcFg, OutputArray dst )
{
	if(!src.obj || !srcFg.obj) return;
	Mat img = src.getMat();
	Mat fg = srcFg.getMat();
	dst.create(fg.size(), CV_32F);
	Mat alphaMap = dst.getMat();

	int w = 3;
	int N = img.rows * img.cols;
	int lambda = 100;

	Mat L = Mat::zeros(N, N, CV_64F);
	Mat alpha = Mat::zeros(N, 1, CV_64F);
	Mat Ds = Mat::zeros(N, N, CV_64F);
	Mat bs = Mat::zeros(N, 1, CV_64F);

	// calculate entries of matrix L
	for(int i = 0; i < L.rows; ++i){
		for(int j = 0; j < L.cols; ++j){

			int xi = i % img.cols;
			int yi = i / img.cols;
			int xj = j % img.cols;
			int yj = j / img.cols;

			Mat pixels(N, 3, CV_8U);

			for(int y = -w/2; y <= w/2; ++y){
				for(int x = -w/2; x <= w/2; ++x){
			        
                    Vec3b Ii = _inImg.at<Vec3b>(yi + y, xi + x);
					Vec3b Ij = _inImg.at<Vec3b>(yj + y, xj + x);

					for(int wy = -w/2; wy <= w/2; ++wy){
						for(int wx = -w/2; wx <= w/2; ++wx){
							//pixels.at<uchar>(n, 0) = I[0];
							//pixels.at<uchar>(n, 1) = I[1];
							//pixels.at<uchar>(n, 2) = I[2];
						}
					}
                }
            }

            Mat covar, mean;
            calcCovarMatrix(pixels, covar, mean, CV_COVAR_ROWS);

                


			
		}
	}


}