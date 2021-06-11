///////////////////////////////////////////////////////////////////////////////////////////////////////
/// DemoTracker.cpp
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <time.h>
#include <Windows.h>
#include <intraface/FaceAlignment.h>
#include <intraface/XXDescriptor.h>
#include <fstream>



using namespace std;
using namespace cv;

bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }



char detectionModel[] = "IntraFaceResources/models/DetectionModel-v1.5.bin";
char trackingModel[] = "IntraFaceResources/models/DetectionModel-v1.5.bin";
string faceDetectionModel("IntraFaceResources/models/haarcascade_frontalface_alt2.xml");
float score, notFace = 0.3;


INTRAFACE::XXDescriptor* xxd;
INTRAFACE::FaceAlignment* fa;
cv::CascadeClassifier* face_cascade;


cv::Mat* frame;
cv::Mat* X0;
cv::Mat* X;
cv::Mat* angle;
cv::Mat* rot;
bool isDetect;


extern "C" __declspec(dllexport) int init(int rows, int cols, unsigned char* fdata, unsigned char* Xdata, unsigned char* X0data, unsigned char*  angleData, unsigned char* rotData) {


	// initialize a XXDescriptor object
	xxd = new INTRAFACE::XXDescriptor(4);
	// initialize a FaceAlignment object
	fa = new INTRAFACE::FaceAlignment(detectionModel, detectionModel, xxd);


	if (!fa->Initialized()) {
		cout << "FaceAlignment cannot be initialized." << endl;
		throw exception();
	}

	// loading OpenCV face detector model
	face_cascade = new cv::CascadeClassifier();
	if (!face_cascade->load(faceDetectionModel))
	{
		cout << "Error loading face detection model." << endl;
		throw exception();
	}

	frame = new cv::Mat(rows, cols, CV_8UC3, fdata);
	X = new cv::Mat(2, 49, CV_32F, Xdata);
	X0 = new cv::Mat(2, 49, CV_32F, X0data);
	angle = new cv::Mat(1, 3, CV_32F, angleData);
	rot = new cv::Mat(3, 3, CV_32F, rotData);
	isDetect = true;

}


extern "C" __declspec(dllexport) int evalDetect(unsigned char* fdata) {

	// Storing the frame
	frame->data = fdata;
	
	int returnvalue = 1;
	cv::Mat X1;


	// face detection
	vector<cv::Rect> faces;
	face_cascade->detectMultiScale(*frame, faces, 1.2, 2, 0, cv::Size(50, 50));
	if (faces.empty()) {
		std::cout << "NO Face Detected" << endl;
		// NO FACE FOUND WITH CV
		//Using pervious frame
		returnvalue = -1;

		//If the previous frame is 0 fail
		if (X0->at<float>(0, 0) == 0) {
			std::cout << "No prvious answer " << endl;
			return -3;
		}

	} else {
		// If face is found
		X0->copyTo(X1);
		// Intraface detecting, more expencse than tracking so is normally only called on first one, only updates X0
		if (fa->Detect(*frame, *max_element(faces.begin(), faces.end(), compareRect), *X0, score) != INTRAFACE::IF_OK) {
			std::cout << "IntraFace not calling " << endl;
			return -2;

		}

		if (score < notFace) { // Proberbly was not a face so will use the prevous frame instead of face
			std::cout << "Bad Face Found" << endl;
			returnvalue = -1;

			X1.copyTo(*X0);

			//If the previous frame is 0 fail
			if (X0->at<float>(0, 0) == 0) {
				std::cout << "No prvious answer " << endl;
				return -3;
			}
		}
	}
	
	// Intraface face tracking using prevous frame or new frame
	if (fa->Track(*frame, *X0, *X, score) != INTRAFACE::IF_OK) {
		std::cout << "IntraFace not calling " << endl;
		return -2;
	}
	// Storing X to X0
	X->copyTo(*X0);

	if (score < notFace) { // Proberbly was not a face so will use backup of X0
		std::cout << "Bad final score " << endl;
		return -4;
	}

	// Getting headpose 
	INTRAFACE::HeadPose hp;
	hp.rot = *rot;
	fa->EstimateHeadPose(*X0, hp);

	for (int i = 0; i < 3; i++)
		angle->at<float>(0, i) = hp.angles[i];


	return returnvalue;
}



extern "C" __declspec(dllexport) int detect(unsigned char* fdata) {
	

	frame->data = fdata;

	if (isDetect)
	{
		// face detection
		vector<cv::Rect> faces;
		face_cascade->detectMultiScale(*frame, faces, 1.2, 2, 0, cv::Size(50, 50));
		// if no face found, do nothing
		if (faces.empty()) {
			std::cout << "No Face Detected" << endl;
			return -1;
		}
		// facial feature detection on largest face found
		if (fa->Detect(*frame, *max_element(faces.begin(), faces.end(), compareRect), *X0, score) != INTRAFACE::IF_OK) {
			std::cout << "IntraFace not calling " << endl;
			return -2;
		}
		isDetect = false;
	}
	else
	{
		// facial feature tracking
		if (fa->Track(*frame, *X0, *X, score) != INTRAFACE::IF_OK) {
			return -2;
		}
		X->copyTo(*X0);
	}
	if (score < notFace) {// Not good enought to use 
		isDetect = true;
		std::cout << "Not good enought face detect again not track " << endl;
		return -1;
	}

	// Getting headpose 
	INTRAFACE::HeadPose hp;
	hp.rot = *rot;
	fa->EstimateHeadPose(*X0, hp);

	for (int i = 0; i < 3; i++) 
		angle->at<float>(0, i) = hp.angles[i];

	//std::cout << score << "\n";

	return 1;
}



extern "C" __declspec(dllexport) int destruct() {

	delete xxd;
	delete fa;
	delete face_cascade;

	delete frame;
	delete X0;
	delete X;
	delete angle;
	delete rot;
	return 1;
}





