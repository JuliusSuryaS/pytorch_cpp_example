#include "Detector.h"

/* Default empty constructor */
Detector::Detector() {
	string current_dir = getCurrentDir();

	this->face_src_exist = false;
	this->landmark_src_exist = false;
	this->resolution = 256;
	this->scale = 1.0;
	this->center = torch::ones(2);
	this->face_detector_src = current_dir + "/data/network_model/face_net_model.dat";
	this->landmark_detector_src = current_dir + "/data/network_model/landmark_net_model.pt";

	// Check source file, load if exist
	if (checkFile(face_detector_src)) {
		this->face_src_exist = true; // Flag to control error
		cout << "Loading face detector ... ";
		deserialize(face_detector_src) >> this->faceNet;
		cout << "loaded successfully" << endl;
	}
	else {
		cout << "\n::ERROR::\nCannot open face detector source\n" << endl;
	}

	// Check source file, load if exist
	if (checkFile(landmark_detector_src)) {
		this->landmark_src_exist = true; // Flag to control error
		cout << "Loading landmark detector ... ";
		this->landmarkNet = torch::jit::load(landmark_detector_src);
		cout << "loaded successfully" << endl;
	}
	else {
		cout << "\n::ERROR::\nCannot open landmark detector source\n" << endl;
	}
}

/*
Constructor
--------------------------------------------
Class constructor takes 'faceNet' weights and
'landmarkNet' weights.
+ Init class attributes
+ Serialize 'faceNet' and 'landmarkNet' weights

*/
Detector::Detector(string face_detector_src, string landmark_detector_src) {
	this->face_src_exist = false;
	this->landmark_src_exist = false;
	this->resolution = 256;
	this->scale = 1.0;
	this->center = torch::ones(2);
	this->face_detector_src = face_detector_src;
	this->landmark_detector_src = landmark_detector_src;

	// Check source file, load if exist
	if (checkFile(face_detector_src)) {
		this->face_src_exist = true; // Flag to control error
		cout << "Loading face detector ... ";
		deserialize(face_detector_src) >> this->faceNet;
		cout << "loaded successfully" << endl;
	}
	else {
		cout << "\n::ERROR::\nCannot open face detector source\n" << endl;
	}

	// Check source file, load if exist
	if (checkFile(landmark_detector_src)) {
		this->landmark_src_exist = true; // Flag to control error
		cout << "Loading landmark detector ... ";
		this->landmarkNet = torch::jit::load(landmark_detector_src);
		cout << "loaded successfully" << endl;
	}
	else {
		cout << "\n::ERROR::\nCannot open landmark detector source\n" << endl;
	}
}

/*
Detect Landmark
--------------------------------------------
Main method to dectect landmark. if visualize is
set to 'true', show detected landmark (default
is false)
+ Read image to opencv mat
+ Detect faces using dlib cnn detector
+ Calculate center and rescale the image
+ Detect landmark
+ Rescale landmark to original image

*/
void Detector::detectLandmark(string img_file, string output_landmark, bool visualize) {
	cv::Mat img = cv::imread(img_file);

	// If not empty process image
	if (!img.empty() && this->face_src_exist && this->landmark_src_exist) {
		auto bbox = detectFace(img); //detect face location

									 // calculate center and scale for preporcessing
		float left = (float)bbox.left();
		float top = (float)bbox.top();
		float right = (float)bbox.right();
		float bottom = (float)bbox.bottom();
		// --
		this->center[0] = right - ((right - left)) / 2.0;
		this->center[1] = bottom - ((bottom - top)) / 2.0;
		this->center[1] = this->center[1] - ((bottom - top)) * 0.12;
		this->scale = (right - left + bottom - top) / 195.0;// / 195.0;

															// Preprocess input
		auto img_rsz = preProcessInput(img); // rescale image
		auto im_tensor = imageToTensor(img_rsz);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(im_tensor);

		cout << "Detecting landmark ..... ";
		auto net_output = this->landmarkNet->forward(inputs).toTensor();
		auto landmark = postProcessOutput(net_output); // get landmark to orignal scale
		cout << "detected" << endl;

		writeToFile(landmark, output_landmark); // save landmark to file

		if (visualize) {
			cv::Mat img_copy = img.clone();
			drawLandmark(landmark, img_copy);
		}
	}
	else {
		cout << "\n::ERROR::\nImage is not found\n" << endl;
		cin.get();
	}
}

/*
Detect Landmark
--------------------------------------------
Overide method that takes opencv mat instead
of image file.
+ Detect faces using dlib cnn detector
+ Calculate center and rescale the image
+ Detect landmark
+ Rescale landmark to original image

*/
void Detector::detectLandmark(cv::Mat img, string output_landmark, bool visualize) {
	auto bbox = detectFace(img); //detect face location

								 // calculate center and scale for preporcessing
	float left = (float)bbox.left();
	float top = (float)bbox.top();
	float right = (float)bbox.right();
	float bottom = (float)bbox.bottom();
	// --
	this->center[0] = right - ((right - left)) / 2.0;
	this->center[1] = bottom - ((bottom - top)) / 2.0;
	this->center[1] = this->center[1] - ((bottom - top)) * 0.12;
	this->scale = (right - left + bottom - top) / 195.0;

	// Preprocess input
	auto img_rsz = preProcessInput(img);
	auto im_tensor = imageToTensor(img_rsz);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(im_tensor);

	cout << "Detecting landmark" << endl;
	auto net_output = this->landmarkNet->forward(inputs).toTensor();
	auto landmark = postProcessOutput(net_output); // get landmark to orignal scale

	writeToFile(landmark, output_landmark);

	if (visualize) {
		cv::Mat img_copy = img.clone();
		drawLandmark(landmark, img_copy);
	}
}

/*
Detect Faces
--------------------------------------------
Using dlib cnn face detector to detect faces.
Takes opencv mat as parameter. Must be
serialized first from constructor.
+ Detect faces using dlib cnn detector
+ Output bounding box of rectangle
*/
dlib::rectangle Detector::detectFace(cv::Mat img) {
	// DLIB Detector -----------------------------------------------------
	dlib::matrix<dlib::rgb_pixel> dlib_img;
	dlib::assign_image(dlib_img, dlib::cv_image<dlib::bgr_pixel>(img));
	auto net_output = this->faceNet(dlib_img);
	auto bbox = net_output[0].rect;
	return bbox;
}

/*
Preprocess Input
--------------------------------------------
Re-implementation from original python code.
Find the center of detected bounding box.
Rescale the image to 256 x 256 based on the
center (maintain aspect ratio).
+ Output rescaled image
*/

cv::Mat Detector::preProcessInput(cv::Mat image) {
	auto center = this->center;
	double scale = this->scale;
	double resolution = (float)this->resolution;

	at::Tensor inp = torch::ones(2);
	auto ul = transform(inp, true);
	auto br = transform(inp.mul_(resolution), true);

	// Cast output to int
	int br0 = br[0].item().to<int64_t>();
	int br1 = br[1].item().to<int64_t>();
	int ul0 = ul[0].item().to<int64_t>();
	int ul1 = ul[1].item().to<int64_t>();

	// Create 0 matrix
	cv::Mat new_img = cv::Mat(br1 - ul1, br0 - ul0, CV_8UC3);
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

	return img_rsz;
}

/*
Convert Mat to Tensor
--------------------------------------------
Convert opencv mat to torch tensor. Manually
loop over the tensor and copy the pixel value
+ Output torch Tensor
*/
at::Tensor Detector::imageToTensor(cv::Mat image) {
	/*
	This is recommended way to load image to tensor
	form official doc, but somehow it didn't work.
	Instead use manual way.
	------
	cv::Mat image_mod, image_float;
	cv::cvtColor(image, image_mod, CV_BGR2RGB);
	image_mod.convertTo(image_float, CV_32F, 1.0 / 255.0);
	at::Tensor im_tensor = torch::from_blob(image_float.data, {1, 3, image.rows, image.cols}, at::kFloat);
	im_tensor = im_tensor.to(at::kFloat);
	------
	*/
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

/*
Transform Point
--------------------------------------------
Re-implementation from original python code.
Utility method to transform scale from network
output to original image scale.
*/
at::Tensor Detector::transform(at::Tensor& point, bool invert) {
	auto center = this->center;
	double scale = this->scale;
	double resolution = (float)this->resolution;

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

/*
Post Process Output
--------------------------------------------
Re-implementation from original python code.
Rescale the output landmark from network to match
original image scale.
+ Output torch tensor
*/
at::Tensor Detector::postProcessOutput(at::Tensor& input) {

	auto center = this->center;
	double scale = this->scale;

	at::Tensor input_rsz = input.view({ input.size(0), input.size(1), input.size(2) * input.size(3) }); // Reshape tensor
	std::tuple<at::Tensor, at::Tensor> max_idx_tup = torch::max(input_rsz, 2); //max_value and max_idx stored in tuple
	at::Tensor max_val = std::get<0>(max_idx_tup); // get the max_value from tuple
	at::Tensor max_idx = std::get<1>(max_idx_tup); // get the max_idx from tuple
	max_idx += 1;
	at::Tensor preds0 = max_idx.view({ max_idx.size(0), max_idx.size(1), 1 }).toType(at::kFloat); // reshape and cast to float
	at::Tensor preds1 = at::clone(preds0);

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
			auto _pt = torch::ones({ 3,1 });
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

/*
Write Landmark to File
--------------------------------------------
Write post-processed landmark tensor to file.
*/
void Detector::writeToFile(at::Tensor& landmark, string filename) {
	std::ofstream fs(filename, std::ios::out);
	for (int i = 0; i < landmark.size(1); i++) {
		fs << landmark[0][i][0].item().to<int64_t>() << " "
			<< landmark[0][i][1].item().to<int64_t>() << " " << "0";
		if (i <= landmark.size(1) - 1) {
			fs << "\n";
		}
	}
	fs.close();
	std::cout << "Landmark written to '" + filename << "'" << std::endl;
}

/*
Draw Landmark
--------------------------------------------
Draw landmark on copy of image for visualization
purpose. Method is called in 'detectLandmark', if
'visualize' is set to true
*/
void Detector::drawLandmark(at::Tensor& landmark, cv::Mat img) {
	for (int i = 0; i < landmark.size(1); i++) {
		int x = landmark[0][i][0].item().to<int64_t>();
		int y = landmark[0][i][1].item().to<int64_t>();
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2);
	}

	cv::imshow("Detected Landmark", img);
	cv::waitKey(0);
}

/*
Check File
--------------------------------------------
Return true if source file is exist.
Return false if source file is not exist.
*/
bool Detector::checkFile(string file_path) {
	std::ifstream fs(file_path, std::ios::in);
	if (!fs.is_open()) {
		cout << "File " << file_path << " is not found" << endl;
		return false;
	}
	fs.close();
	return true;
}

string Detector::getCurrentDir() {
	const unsigned long max_dir = 500;
	char current_dir[max_dir];
	GetCurrentDirectory(max_dir, current_dir);
	return current_dir;
}