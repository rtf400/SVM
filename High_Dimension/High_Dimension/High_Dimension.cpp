#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int, char**)
{
	int i, j;
	float *feature;									// Use 59*68-D vector
	feature = (float*)calloc(sizeof(float), 59*68);
	int *labels;									// 200 Data
	labels = (int*)calloc(sizeof(int),200);
	

	// Set up training data	
	Mat trainingDataMat(200, 59 * 68, CV_32FC1, Scalar(0));

	feature[0] = 501;								// 1st Data : { 501, 10, 0, ... , 0 }
	feature[1] = 10;
	for (i = 0; i < 59 * 68; i++) {
		trainingDataMat.at<float>(0, i) = feature[i];
	}

	feature[0] = 255;								// 2nd Data : { 255, 10, 0, ... , 0 }
	feature[1] = 10;
	for (i = 0; i < 59 * 68; i++) {
		trainingDataMat.at<float>(1, i) = feature[i];
	}

	feature[0] = 501;								// 3rd Data : { 501,255, 0, ... , 0 }
	feature[1] = 255;
	for (i = 0; i < 59 * 68; i++) {
		trainingDataMat.at<float>(2, i) = feature[i];
	}

	feature[0] = 10;								// 4th Data : { 10, 501, 0, ... , 0 }
	feature[1] = 501;
	for (i = 0; i < 59 * 68; i++) {
		trainingDataMat.at<float>(3, i) = feature[i];
	}

	feature[0] = 0;									// Other Data : { 0, ... , 0 }
	feature[1] = 0;
	for (j = 4; j < 200; j++) {
		for (i = 0; i < 59 * 68; i++) {
			trainingDataMat.at<float>(j, i) = feature[i];
		}
	}
	

	labels[0] = 1;									// label : 1, 0, 0, ... , 0
	labels[1] = 0;
	labels[2] = 0;
	labels[3] = 0;
	Mat labelsMat(200, 1, CV_32SC1, labels);

	
	// Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
	

	Mat sampleMat(1, 59 * 68, CV_32FC1, Scalar(0));	// test the sample data
	sampleMat.at<float>(0, 0) = 501;
	sampleMat.at<float>(0, 1) = 10;
	float response = svm->predict(sampleMat);
	
	printf("\n%f",response);						// result : print the label

	free(feature);
	
	return 0;
}