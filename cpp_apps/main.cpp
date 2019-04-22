#define _CRT_SECURE_NO_WARNINGS
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

bool checkFile(string filename);
cv::Mat preProcessInput(cv::Mat image, at::Tensor center, double scale, double resolution); // TODO
at::Tensor imageToTensor(cv::Mat image);
std::vector<cv::Rect> detectFace(cv::Mat image, string detector_source);
at::Tensor transform(at::Tensor point, at::Tensor center, float scale, float resolution, bool invert);
at::Tensor postProcessOutput(at::Tensor net_output, torch::Tensor center , double scale);
void writeToFile(at::Tensor& landmark, string filename);
void drawLandmark(at::Tensor& landmark, cv::Mat img);

string frontal_face_source = "D:/Dropbox (IVCL)/library/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
string profile_face_source = "D:/Dropbox (IVCL)/library/opencv/sources/data/haarcascades/haarcascade_profileface.xml";
cv::CascadeClassifier frontal_cascade, profile_cascade;

using std::cout;
using std::cin;
using std::endl;

using namespace dlib;

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET> using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET> using downsampler = dlib::relu<dlib::affine<con5d<32, dlib::relu<affine<con5d<32, dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5 = dlib::relu<dlib::affine<con5<45, SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

int main() {

	string profile_path = "D:/current_project/RBF/RBF_Modeling/texturemap/IVCL_FaceData_renamed/images_profile/001.jpg";
	string frontal_path = "D:/current_project/RBF/RBF_Modeling/texturemap/IVCL_FaceData_renamed/image/001.jpg";

	cv::Mat img_frontal = cv::imread(frontal_path);
	auto frontal_faces_bbox = detectFace(img_frontal, frontal_face_source);
	std::cout << frontal_faces_bbox.size() << std::endl;

	cv::Mat img_profile = cv::imread(profile_path);
	auto profile_faces_bbox = detectFace(img_profile, profile_face_source);
	std::cout << profile_faces_bbox.size() << std::endl;

	// DLIB Detector -----------------------------------------------------
	dlib::matrix<dlib::rgb_pixel> dlib_img;
	dlib::assign_image(dlib_img, dlib::cv_image<dlib::bgr_pixel>(img_frontal));
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	//std::vector<dlib::rectangle> bbox = detector(dlib_img);

	net_type net;
	//const char* detector_src = "mmod_human_face_detector.dat.bz2";
	const char* detector_src = "mmod_human_face_detector.dat";
	dlib::deserialize("D:/others/test_pytorch_vs15_no_cmake/Project1/mmod_human_face_detector.dat") >> net;
	auto net_out = net(dlib_img);
	auto bbox = net_out[0].rect;

	int dleft = bbox.left();
	int dtop = bbox.top();
	int dright = bbox.right();
	int dbot = bbox.bottom();
	cout << "dlib bbox " << bbox.left() << " " << bbox.top() << " " << bbox.right() << " " << bbox.bottom() << endl;

	cv::Mat img_draw = img_frontal.clone();
	cv::rectangle(img_draw, cv::Rect(cv::Point(dleft,dtop), cv::Point(dright,dbot)), cv::Scalar(0,0,255));
	cv::imshow("dlib detector", img_draw);
	cv::waitKey(0);




	float left = (float)dleft;
	float top = (float)dtop;
	float right = (float)dright;
	float bottom = (float)dbot;

	//left = 66;
	//top = 166;
	//right = 311;
	//bottom = 410;

	at::Tensor ctr = torch::ones(2);
	ctr[0] = right - ((right - left)) / 2.0;
	ctr[1] = bottom - ((bottom - top)) / 2.0;
	ctr[1] = ctr[1] - ((bottom - top)) * 0.12;
	double scl = (right - left + bottom - top) / 195.0;// / 195.0;

	cout << "center " << ctr << endl;
	cout <<  "scale " << scl << endl;


	auto img_rsz = preProcessInput(img_frontal, ctr, scl, 256.0);
	cv::imshow("img resz", img_rsz);
	cv::waitKey(0);



	// ----

	torch::Tensor matrix = torch::ones({ 5,5 });
	torch::Tensor matrix2 = at::clone(matrix);
	matrix[0] = 10;
	std::tuple<at::Tensor, at::Tensor> index_max  = torch::max(matrix, 1); //func
	auto val  = std::get<0>(index_max);
	auto idx  = std::get<1>(index_max);
	std::cout << "Max val = " << val;
	std::cout << " in Index  " << idx;
	std::cout << matrix.view({1,25}) << std::endl; //func
	std::cout << matrix2 << std::endl; //func
	torch::Tensor a = torch::ones({ 1,2 });
	std::cout << a.size(0) << a.size(1) << std::endl;

	auto n = torch::rand({ 1, 2 });
	auto m = torch::rand({ 1, 68, 2 });
	//m.add_(1).floor_();
	std::cout << m[0][0] << std::endl;
	n = m[0][0];
	std::cout << "N" << std::endl;
	std::cout << n << std::endl;
	//int k = m[0][0].item().to<int64_t>();
	//std::cout << "check -> "<< k << std::endl;


	string filename = "D:/current_project/RBF/RBF_Modeling/x64/Release/face-alignment/jit_test.pt";
	auto file_exist = checkFile(filename);
	if (!file_exist) {
		std::cin.get();
		return -1;
	}
	std::cout << "Loading pytorch model\n";
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(filename);


	auto im_tensor = imageToTensor(img_rsz);
	cout << im_tensor.size(0) << " - " << im_tensor.size(1) << " - " << im_tensor.size(2) << endl;

	//auto im_tensor = torch::ones({ 1, 3, 256, 256 }) * 55;


	std::vector<torch::jit::IValue>	inputs;
	inputs.push_back(im_tensor);

	std::cout << "Running inference\n";
	at::Tensor output = module->forward(inputs).toTensor();
	std::cout << "Finished running inference\n";
	std::cout << output.size(0) << " " << output.size(1) << " " << output.size(2) <<  " " << output.size(3) << std::endl;
	at::Tensor output_ori = postProcessOutput(output, ctr, scl);
	std::cout << "Proc output " << output_ori.size(0) << " x " << output_ori.size(1) << " x " << output_ori.size(2) << std::endl;
	std::cout << output_ori << std::endl;

	writeToFile(output_ori, "landmark.txt");
	drawLandmark(output_ori, img_frontal);

	assert(module != nullptr);
	std::cout << "ok" << std::endl;
	std::cin.get();
}
// ================================================== End of main ========================================================== //

// ---

// ================================================== Function Definition ========================================================== //
cv::Mat preProcessInput(cv::Mat image, at::Tensor center, double scale, double resolution) {
	at::Tensor inp = torch::ones(2);
	auto ul = transform(inp, center, scale, resolution, true);
	auto br = transform(inp.mul_(resolution), center, scale, resolution, true);

	cout << "ul ->  " << ul << endl;
	cout << "br ->  " << br << endl;

	// Cast output to int
	int br0 = br[0].item().to<int64_t>();
	int br1 = br[1].item().to<int64_t>();
	int ul0 = ul[0].item().to<int64_t>();
	int ul1 = ul[1].item().to<int64_t>();

	// Create 0 matrix
	cv::Mat new_img = cv::Mat(br1 - ul1, br0 - ul0, CV_8UC3);
	cout << new_img.channels() << " " <<  new_img.rows << " x " << new_img.cols << endl;
	int ht = image.rows;
	int wd = image.cols;
	int newx0 = std::max(1, -ul0 + 1);
	int newx1 = std::min(br0, wd) - ul0;
	int newy0 = std::max(1, -ul1 + 1);
	int newy1 = std::min(br1, ht) - ul1;
	int oldx0 = std::max(1, ul0 + 1);
	int oldx1 = std::min(br0, wd);
	int oldy0 = std::max(1, ul1 + 1);
	int oldy1 = std::min(br1, ht);

	cout << "newx0 " << newx0 << " - " << newx1 << endl;
	cout << "newy0 " << newy0 << " - " << newy1 << endl;
	cout << "oldx0 " << oldx0 << " - " << oldx1 << endl;
	cout << "oldy0 " << oldy0 << " - " << oldy1 << endl;

	// Re-arrange image
	for (int y = newy0 - 1, i = oldy0 - 1; y < newy1; y++, i++) {
		for (int x = newx0 - 1, j = oldx0 - 1; x < newx1; x++, j++) {
			new_img.at<cv::Vec3b>(y, x)[0] = image.at<cv::Vec3b>(i, j)[0];
			new_img.at<cv::Vec3b>(y, x)[1] = image.at<cv::Vec3b>(i, j)[1];
			new_img.at<cv::Vec3b>(y, x)[2] = image.at<cv::Vec3b>(i, j)[2];
		}
	}

	cv::Mat img_rsz;
	cv::resize(new_img, img_rsz, cv::Size(256, 256));

	cv::imshow("resized img", img_rsz);
	cv::waitKey(0);
	return img_rsz;
}

at::Tensor imageToTensor(cv::Mat image) {
	/*cv::Mat image_mod, image_float;
	cv::cvtColor(image, image_mod, CV_BGR2RGB);
	image_mod.convertTo(image_float, CV_32F, 1.0 / 255.0);
	at::Tensor im_tensor = torch::from_blob(image_float.data, {1, 3, image.rows, image.cols}, at::kFloat);
	im_tensor = im_tensor.to(at::kFloat);*/
	at::Tensor im_tensor = torch::ones({ 1,3,256,256 });
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			im_tensor[0][0][i][j] = image.at<cv::Vec3b>(i, j)[2];
			im_tensor[0][1][i][j] = image.at<cv::Vec3b>(i, j)[1];
			im_tensor[0][2][i][j] = image.at<cv::Vec3b>(i, j)[0];
		}
	}
	im_tensor = im_tensor.to(at::kFloat);
	im_tensor.div_(255.0);
	return im_tensor;
}

at::Tensor postProcessOutput(at::Tensor input, at::Tensor center, double scale) {
	/*
		Re-implemented from original Pytorch code and ported to Pytorch C++ API.
		- Post process the output from the network to match the original image scale.
	*/
	at::Tensor input_rsz = input.view({ input.size(0), input.size(1), input.size(2) * input.size(3) }); // Reshape tensor
	std::tuple<at::Tensor, at::Tensor> max_idx_tup = torch::max(input_rsz, 2); //max_value and max_idx stored in tuple
	at::Tensor max_val = std::get<0>(max_idx_tup); // get the max_value from tuple
	at::Tensor max_idx = std::get<1>(max_idx_tup); // get the max_idx from tuple
	max_idx += 1;
	at::Tensor preds0 = max_idx.view({ max_idx.size(0), max_idx.size(1), 1 }).toType(at::kFloat); // reshape and cast to float
	at::Tensor preds1 = at::clone(preds0);

	cout << max_val << endl;
	cout << max_idx << endl;

	// Do some prepocessing
	preds0 = (preds0 - 1) % input.size(3) + 1;
	preds1.add_(-1).div_(input.size(2)).floor_().add_(1);// equivalent term
	auto preds = torch::cat({ preds0, preds1 }, 2); // concat and make -> (1, 68, 2)

	for (int i = 0; i < preds.size(0); i++) {
		for (int j = 0; j < preds.size(1); j++) {
			int px = preds[i][j][0].item().to<int64_t>() - 1;
			int py = preds[i][j][1].item().to<int64_t>() - 1;
			if (px > 0 && px < 63 && py > 0 && py < 63) {
				auto p1 = input[i][j][py][px + 1] - input[i][j][py][px - 1];
				auto p2 = input[i][j][py + 1][px] - input[i][j][py - 1][px];
				preds[i][j][0].add_(p1.sign_().mul_(0.25));
				preds[i][j][1].add_(p2.sign_().mul_(0.25));
			}
		}
	}
	preds.add_(-0.5);

	at::Tensor preds_orig = torch::zeros_like(preds);
	for (int i = 0; i < input.size(0); i++) {
		for (int j = 0; j < input.size(1); j++) {
			auto _pt = torch::ones({3,1});
			_pt[0] = preds[i][j][0];
			_pt[1] = preds[i][j][1];

			double h = 200.0 * scale;
			auto t = torch::eye({ 3 });
			t[0][0] = input.size(2) / h;
			t[1][1] = input.size(2) / h;
			t[0][2] = input.size(2) * (-center[0] / h + 0.5);
			t[1][2] = input.size(2) * (-center[1] / h + 0.5);

			t = torch::inverse(t);
			auto new_point = torch::matmul(t, _pt).toType(torch::kLong);
			preds_orig[i][j][0] = new_point[0][0];
			preds_orig[i][j][1] = new_point[1][0];
		}
	}
	return preds_orig;
}

at::Tensor transform(at::Tensor point, at::Tensor center, float scale, float resolution, bool invert) {
	auto _pt = torch::ones({ 3 });
	_pt[0] = point[0];
	_pt[1] = point[1];

	double h = 200.0 * scale;
	auto t = torch::eye(3);
	t[0][0] = resolution / h;
	t[1][1] = resolution / h;
	t[0][2] = resolution * (-center[0] / h + 0.5);
	t[1][2] = resolution * (-center[1] / h + 0.5);

	if (invert) {
		t = torch::inverse(t);
	}
	auto new_point = torch::matmul(t, _pt).toType(torch::kLong);
	return new_point;
}

std::vector<cv::Rect> detectFace(cv::Mat image, string detector_source) {
	// Convert image to grayscale
	cv::Mat image_gray;
	cv::cvtColor(image, image_gray, CV_BGR2GRAY);
	cv::equalizeHist(image_gray, image_gray);

	// Detection output
	std::vector<cv::Rect> faces_bbox;

	// Load face detector
	cv::CascadeClassifier face_detector;
	if (!face_detector.load(detector_source)) {
		std::cout << "Error loading face detector\n";
	}
	else {
		// Detect faces if loading successful
		face_detector.detectMultiScale(image_gray, faces_bbox, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(20, 20));
	}
	cv::Mat img_sample = image.clone();
	auto top_left = cv::Point(faces_bbox[0].x, faces_bbox[0].y);
	auto bot_right = cv::Point(faces_bbox[0].x + faces_bbox[0].width, faces_bbox[0].y + faces_bbox[0].height);
	//cv::rectangle(img_sample, cv::Rect(top_left, bot_right), cv::Scalar(0, 0, 255));
	//cv::imshow("Detected face", img_sample);
	//cv::waitKey(0);

	return faces_bbox;
}

void writeToFile(at::Tensor& landmark, string filename) {
	std::cout << "Writing tensor to " + filename << std::endl;
	std::ofstream fs(filename, std::ios::out);
	for (int i = 0; i < landmark.size(1); i++) {
		fs << landmark[0][i][0].item().to<int64_t>() << " " << landmark[0][i][1].item().to<int64_t>() << " " << "0";
		if (i <= landmark.size(1) - 1) {
			fs << "\n";
		}
	}
	fs.close();
	std::cout << "Finish writing file" << std::endl;
}

void drawLandmark(at::Tensor& landmark, cv::Mat img) {
	for (int i = 0; i < landmark.size(1); i++) {
		int x = landmark[0][i][0].item().to<int64_t>();
		int y = landmark[0][i][1].item().to<int64_t>();
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 1);
	}

	cv::imshow("landmark", img);
	cv::waitKey(0);
}

bool checkFile(string filename) {
	std::ifstream fs(filename, std::ios::in);
	if (!fs.is_open()) {
		std::cout << "File " << filename << " is not found\n";
		return false;
	}
	fs.close();
	return true;
}
// ================================================== End of Function Definition ========================================================== //
