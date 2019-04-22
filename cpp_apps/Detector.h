#define _CRT_SECURE_NO_WARNINGS
#pragma once
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>

using namespace dlib;

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = dlib::relu<dlib::affine<con5d<32, dlib::relu<affine<con5d<32, dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = dlib::relu<dlib::affine<con5<45, SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

using std::string;
using std::cout;
using std::cin;
using std::endl;

class Detector {
public:
	// Attributes
	string face_detector_src;
	string landmark_detector_src;
	net_type faceNet;
	std::shared_ptr<torch::jit::script::Module> landmarkNet;

	// Methods
	Detector();
	Detector(string face_detector_source, string landmark_detector_source);
	void detectLandmark(string image_file_path, string landmark_out_path, bool visualize);
	void detectLandmark(cv::Mat image, string landmark_out_path, bool visualize);

private:
	// Attributes
	int resolution;
	double scale;
	at::Tensor center;
	bool face_src_exist;
	bool landmark_src_exist;

	// Methods
	bool checkFile(string file_path);
	dlib::rectangle detectFace(cv::Mat image);
	cv::Mat preProcessInput(cv::Mat image);
	at::Tensor imageToTensor(cv::Mat image);
	at::Tensor transform(at::Tensor& input, bool invert);
	at::Tensor postProcessOutput(at::Tensor& net_output);
	void writeToFile(at::Tensor& landmark, string landmark_file_path);
	void drawLandmark(at::Tensor& landmark, cv::Mat image);
	string getCurrentDir();


};
