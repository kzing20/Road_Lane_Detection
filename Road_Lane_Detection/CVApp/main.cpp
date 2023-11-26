#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ������ ����
Mat frame, gray, edges, result;
int width, height;


// HoughLinesP �Լ� ����
double rho = 2;
double theta = CV_PI / 180;
int hough_threshold = 100;
double minLineLength = 100;
double maxLineGap = 250;

//���� �Ǵ� ����
double prev_center=2000;
int change_lane = -1;  // ���� ���� Ƚ��


// ���� threshold
double slope_min_threshold = 0.6;
double slope_max_threshold = 1.2;


// ���� line ���� ���� ����
vector<Vec4i> r_save_lines;
int right_sum[4] = { 0 };
int right_avg[4];
int r_save_cnt = 0;
vector<Vec4i> l_save_lines;
int left_sum[4] = { 0 };
int left_avg[4];
int l_save_cnt = 0;

//warning ���� �˻�
vector<bool> warning_vec = { false };

// ȭ�鿡�� line�� ������ ����(ROI) ����
Mat setROI(Mat edges, Point* points)
{
	const Point* pts[1] = { points };
	int npts[1] = { 4 };

	// ROI�� ������� ĥ�Ѵ�.
	Mat mask = Mat::zeros(edges.rows, edges.cols, CV_8UC1);
	fillPoly(mask, pts, npts, 1, Scalar(255));

	// ������� ĥ�� �κп� �ش��ϴ� edge�� �����Ѵ�.
	Mat ROI_edges;
	bitwise_and(edges, mask, ROI_edges);

	return ROI_edges;
}


// ���� ����� 1�� ���͸�
void filterByColor(Mat image, Mat& filtered)
{
	// ��� ����(RGB)
	Scalar lower_white = Scalar(140, 140, 140);
	Scalar upper_white = Scalar(255, 255, 255);
	// �Ķ��� ����(HSV)
	Scalar lower_blue = Scalar(40, 0, 40);
	Scalar upper_blue = Scalar(130, 255, 255);

	Mat image_bgr = image.clone(), image_hsv;
	Mat white_mask, white_image;
	Mat blue_mask, blue_image;

	// ��� ������ ���͸��ؼ� white_mask�� �����Ѵ�.
	inRange(image_bgr, lower_white, upper_white, white_mask);
	bitwise_and(image_bgr, image_bgr, white_image, white_mask);

	// ��� ������ ä���� ���δ�.
	cvtColor(white_image, white_image, COLOR_BGR2HSV);
	vector<Mat> channel_white;
	split(white_image, channel_white);
	channel_white[1] *= 2;
	merge(channel_white, white_image);
	cvtColor(white_image, white_image, COLOR_HSV2BGR);

	// RGB ������ HSV �������� ��ȯ�Ѵ�.
	cvtColor(image_bgr, image_hsv, COLOR_BGR2HSV);

	// �Ķ��� ������ ���͸��ؼ� blue_mask�� �����Ѵ�.
	inRange(image_hsv, lower_blue, upper_blue, blue_mask);
	bitwise_and(image_hsv, image_hsv, blue_image, blue_mask);

	// �Ķ��� ������ ä���� ���δ�.
	vector<Mat> channel_blue;
	split(blue_image, channel_blue);
	channel_blue[1] *= 3;
	channel_blue[2] += 100;
	merge(channel_blue, blue_image);
	cvtColor(blue_image, blue_image, COLOR_HSV2BGR);

	// ��� ������ �Ķ��� ������ ��ģ��.
	addWeighted(white_image, 1.0, blue_image, 1.0, 0.0, filtered);
	namedWindow("filtered color region", 0);
	imshow("filtered color region", filtered);
}
//���� ��ȯ �Լ�
string changeDir(string output_s, float ratio, float left_thres, float right_thres) {
	if (ratio > left_thres) {
		output_s = "Left Turn";
	}
	else if (ratio < right_thres) {
		output_s = "Right Turn";
	}
	else {
		output_s = "Straight";
	}
	return output_s;

}

// ���� ����
void detectLane(Mat& mark, Mat& detected, vector<Vec4i> lines)
{
	
	bool warning = false;
	vector<double> slopes;
	vector<Vec4i> filtered_lines;
	vector<Vec4i> warning_lines;

	// �� ������ ���⸦ ���Ѵ�
	int lines_size = (int)lines.size();
	for (int i = 0; i < lines_size; i++) {
		Vec4i line = lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope;

		// ���⸦ ����Ѵ�.
		// zero division�� ���� ����
		if (x2 - x1 == 0)
			slope = 999.0;
		else
			slope = ((double)y2 - y1) / ((double)x2 - x1);

		// ���Ⱑ min_threshold �̻� max_threshold ������ line�� ���� �ĺ��� �����Ѵ�.
		if (abs(slope) > slope_min_threshold && abs(slope) < slope_max_threshold) {
			slopes.push_back(slope);
			filtered_lines.push_back(line);
		}
		// ���Ⱑ 1.6���� ũ�� ������ x ��ǥ�� ������ �߾ӿ� ��ġ�ϸ� warning �����̴�.
		//if (abs(slope) > 2 && (x1 > width * 0.4 && x1 < width * 0.5) && (x2 > width * 0.4 && x2 < width * 0.55)) {
		//	warning = true;
		//	warning_lines.push_back(line);
		//}
	}
	Mat tmp1 = frame.clone();
	// lines�� ����� ���� �׸���
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(tmp1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
	}

	// ����� ǥ��
	namedWindow("Hough Lines", 0);
	imshow("Hough Lines", tmp1);
	//// ����� ���͸��� �� � ������ ������ Ȯ���ϴ� �ڵ�
	//Mat tmp2 = frame.clone();
	//for (Vec4i l : warning_lines) {
	//	line(tmp2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
	//}
	//namedWindow("warning_lines", 0);
	//imshow("warning_lines", tmp2);


	// ���� line�� ������ line�� ����.
	// ���� 1: ������ ������ ����� ���, ���� ������ ����� ����
	// ���� 2: ������ ������ x ��ǥ ��հ� > �߾� 
	vector<Vec4i> right_lines;
	vector<Vec4i> left_lines;
	int center_x = (int)(width * 0.5);  // �߾� x ��ǥ

	int filtered_lines_size = (int)filtered_lines.size();
	for (int i = 0; i < filtered_lines_size; i++) {
		Vec4i line = filtered_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope = slopes[i];
		int avg_x = (x1 + x2) >> 1;  // ��Ʈ ������ ����ؼ� �ӵ� ��

		// ���� > 0, x ��ǥ ��հ� > �߾� x ��ǥ�̸� ������ ������ ���� avg_x > center_x
		if (slope > 0.6) {
			right_lines.push_back(line);
		}
		// ���� < 0, x ��ǥ ��հ� < �߾� x ��ǥ�̸� ���� ������ ����
		else if (slope < -0.6) {
			left_lines.push_back(line);
		}
	}
	//warning ����(left,right line�� ��� ������ �ȵǰ� ���� 5�� �����ӿ����� warning ���¿��ٸ�)
	size_t vectorSize = warning_vec.size();

	if (left_lines.size() == 0 && right_lines.size() == 0) {
		if (vectorSize >= 5 &&
			warning_vec[vectorSize - 1] &&
			warning_vec[vectorSize - 2] &&
			warning_vec[vectorSize - 3] &&
			warning_vec[vectorSize - 4] &&
			warning_vec[vectorSize - 5]) {
			warning = true;
		}
		else {
			printf("warning ����");
			warning_vec.push_back(true);
		}
	} else {
		warning_vec.push_back(false);
	}
	//warning ����(������)
	/*if (left_lines.size() == 0 && right_lines.size() == 0) {
		warning = true;
	}*/
	Mat tmp3 = frame.clone();
	// right_lines�� ����� ���� �׸���
	for (size_t i = 0; i < right_lines.size(); i++) {
		Vec4i l = right_lines[i];
		
		line(tmp3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}

	// left_lines�� ����� ���� �׸���	
	for (size_t i = 0; i < left_lines.size(); i++) {
		Vec4i l = left_lines[i];
		
		line(tmp3, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
	}
	//int right_lines_size = (int)right_lines.size();

	//// ������ filtered line�� �ϳ��� ������ ��� ���ϱ�
	//if (right_lines_size != 0) {
	//	
	//	// ���� ������ ������ right line�� �����ϰ� ��� line�� �߰�
	//	r_save_lines.erase(r_save_lines.begin(), r_save_lines.begin() + 1);
	//	r_save_lines.push_back(Vec4i(max_line[0], max_line[1], max_line[2], max_line[3]));

	//	r_save_cnt++;
	//}
	//int left_lines_size = (int)left_lines.size();

	//// ���� filtered line�� �ϳ��� ������ ��ǥ�� ��� ���ϱ�
	//if (left_lines_size != 0) {

	//	// ���� ������ ������ left line�� �����ϰ� ��� line�� �߰�
	//	l_save_lines.erase(l_save_lines.begin(), l_save_lines.begin() + 1);
	//	l_save_lines.push_back(Vec4i(min_line[0], min_line[1], min_line[2], min_line[3]));

	//	l_save_cnt++;
	//}
	// ����� ǥ��
	namedWindow("filtered Lines", 0);
	imshow("filtered Lines", tmp3);

	////���� ȸ�� ����
	//Point p1, p2, p3, p4;
	//vector<Point> left_points, right_points;
	//Vec4d left_line, right_line;
	//Point right_b, left_b;
	//double right_m = 1.5;
	//double left_m = -1.5;
	//for (auto i : right_lines) {
	//	p1 = Point(i[0], i[1]);
	//	p2 = Point(i[2], i[3]);

	//	right_points.push_back(p1);
	//	right_points.push_back(p2);
	//}
	//if (right_points.size() > 0) {
	//	//�־��� contour�� ����ȭ�� ���� ����
	//	fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

	//	right_m = right_line[1] / right_line[0];  //����
	//	right_b = Point(right_line[2], right_line[3]);
	//}

	//for (auto j : left_lines) {
	//	p3 = Point(j[0], j[1]);
	//	p4 = Point(j[2], j[3]);

	//	left_points.push_back(p3);
	//	left_points.push_back(p4);
	//}

	//if (left_points.size() > 0) {
	//	//�־��� contour�� ����ȭ�� ���� ����
	//	fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

	//	left_m = left_line[1] / left_line[0];  //����
	//	left_b = Point(left_line[2], left_line[3]);
	//}

	////�¿� �� ������ �� ���� ����Ѵ�.
	////y = m*x + b  --> x = (y-b) / m
	//int y1 = height;
	//int y2 = (int)(height * 0.6);

	//double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	//double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

	//double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
	//double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;


	//// ---------- 1) filtered line ��� ���ϱ�

	// ������ filtered line ��� ���ϱ�
	right_sum[0] = 0;
	right_sum[1] = 0;
	right_sum[2] = 0;
	right_sum[3] = 0;

	int right_lines_size = (int)right_lines.size();
	int max_right_x = numeric_limits<int>::min(); // max_right_x�� ���� ���� ������ �ʱ�ȭ
	int max_i = 0;
	for (int i = 0; i < right_lines_size; i++) {
		Vec4i line = right_lines[i];
		int current_max_x = max(line[0], line[2]);
		if (current_max_x > max_right_x) {
			max_right_x = current_max_x;
			max_i = i;
		}
		
	}

	for (int i = 0; i < right_lines_size; i++) {
		Vec4i line = right_lines[i];

		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		if (i == max_i) {
			right_sum[0] += (5 * x1);
			right_sum[1] += (5 * y1);
			right_sum[2] += (5 * x2);
			right_sum[3] += (5 * y2);
		}
		else {
			right_sum[0] += x1;
			right_sum[1] += y1;
			right_sum[2] += x2;
			right_sum[3] += y2;
		}
	}

	// ������ filtered line�� �ϳ��� ������ ��� ���ϱ�
	if (right_lines_size != 0) {
		right_avg[0] = int((double)right_sum[0] / (right_lines_size+4));
		right_avg[1] = int((double)right_sum[1] / (right_lines_size+4 ));
		right_avg[2] = int((double)right_sum[2] / (right_lines_size+4 ));
		right_avg[3] = int((double)right_sum[3] / (right_lines_size+4 ));

		// ���� ������ ������ right line�� �����ϰ� ��� line�� �߰�
		r_save_lines.erase(r_save_lines.begin(), r_save_lines.begin() + 1);
		r_save_lines.push_back(Vec4i(right_avg[0], right_avg[1], right_avg[2], right_avg[3]));

		r_save_cnt++;
	}

	// ���� filtered line ��� ���ϱ�
	left_sum[0] = 0;
	left_sum[1] = 0;
	left_sum[2] = 0;
	left_sum[3] = 0;

	int left_lines_size = (int)left_lines.size();
	int min_left_x = numeric_limits<int>::max();
	int min_i = 0;
	for (int i = 0; i < left_lines_size; i++) {
		Vec4i line = left_lines[i];

		int current_min_x = min(line[0], line[2]);
		if (current_min_x < min_left_x) {
			min_left_x = current_min_x;
			min_i = i;
		}

	}
	for (int i = 0; i < left_lines_size; i++) {
		Vec4i line = left_lines[i];
		
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		if (i == min_i) {
			left_sum[0] += (5*x1);
			left_sum[1] += (5*y1);
			left_sum[2] += (5*x2);
			left_sum[3] += (5*y2);
		}
		else {
			left_sum[0] += x1;
			left_sum[1] += y1;
			left_sum[2] += x2;
			left_sum[3] += y2;
		}
	}

	// ���� filtered line�� �ϳ��� ������ ��ǥ�� ��� ���ϱ�
	if (left_lines_size != 0) {
		left_avg[0] = int((double)left_sum[0] / (left_lines_size +4));
		left_avg[1] = int((double)left_sum[1] / (left_lines_size + 4));
		left_avg[2] = int((double)left_sum[2] / (left_lines_size +4));
		left_avg[3] = int((double)left_sum[3] / (left_lines_size + 4));

		// ���� ������ ������ left line�� �����ϰ� ��� line�� �߰�
		l_save_lines.erase(l_save_lines.begin(), l_save_lines.begin() + 1);
		l_save_lines.push_back(Vec4i(left_avg[0], left_avg[1], left_avg[2], left_avg[3]));

		l_save_cnt++;
	}

	// ---------- 2) �ֱ� ���� line 10���� ��� ���ϱ�

	// ������ ���� line 10���� ��� ���ϱ�
	if (r_save_cnt > 10) {
		right_sum[0] = 0;
		right_sum[1] = 0;
		right_sum[2] = 0;
		right_sum[3] = 0;

		for (int i = 0; i < 10; i++) {
			Vec4i line = r_save_lines[i];
			if ((i ==8)) {
				right_sum[0] += (3*line[0]);
				right_sum[1] += (3*line[1]);
				right_sum[2] += (3*line[2]);
				right_sum[3] += (3*line[3]);
			}
			else {
				right_sum[0] += line[0];
				right_sum[1] += line[1];
				right_sum[2] += line[2];
				right_sum[3] += line[3];
			}
		}

		right_avg[0] = (int)((double)right_sum[0] / 12);
		right_avg[1] = (int)((double)right_sum[1] / 12);
		right_avg[2] = (int)((double)right_sum[2] / 12);
		right_avg[3] = (int)((double)right_sum[3] / 12);
	}

	// ���� ���� line 10���� ��� ���ϱ�
	if (l_save_cnt > 10) {
		left_sum[0] = 0;
		left_sum[1] = 0;
		left_sum[2] = 0;
		left_sum[3] = 0;

		for (int i = 0; i <10; i++) {
			Vec4i line = l_save_lines[i];
			if (i == 8) {
				left_sum[0] += (3*line[0]);
				left_sum[1] += (3*line[1]);
				left_sum[2] += (3*line[2]);
				left_sum[3] += (3*line[3]);
			}
			else {
				left_sum[0] += line[0];
				left_sum[1] += line[1];
				left_sum[2] += line[2];
				left_sum[3] += line[3];
			}
		}

		left_avg[0] = (int)((double)left_sum[0] / 12);
		left_avg[1] = (int)((double)left_sum[1] / 12);
		left_avg[2] = (int)((double)left_sum[2] / 12);
		left_avg[3] = (int)((double)left_sum[3] / 12);
	}

	// ---------- 3) ��ǥ line ���ϱ�

	// ������ ��ǥ line ��ǥ�� ����
	int right_x = right_avg[0];
	int right_y = right_avg[1];
	double right_dx = (double)right_avg[0] - right_avg[2];
	double right_dy = (double)right_avg[1] - right_avg[3];
	// zero divison�� ���� ����
	if (right_dx == 0.0) right_dx = 0.001;
	if (right_dy == 0.0) right_dy = 0.001;
	double right_slope = right_dy / right_dx;

	// ���� ��ǥ line ��ǥ�� ����
	int left_x = left_avg[0];
	int left_y = left_avg[1];
	double left_dx = (double)left_avg[0] - left_avg[2];
	double left_dy = (double)left_avg[1] - left_avg[3];
	// zero divison�� ���� ����
	if (left_dx == 0.0) left_dx = 0.001;
	if (left_dy == 0.0) left_dy = 0.001;
	double left_slope = left_dy / left_dx;

	// ������ ���� ��ǥ line ��ǥ ���ϱ�
	// y = mx + b �� x = (y - b) / m
	// y = m(x - x0) + y0 = mx - mx0 + y0 �� x = (y - y0) / m + x0
	int y1 = height;
	int y2 = (int)(height * 0.6);
	int right_x1 = (int)(((double)y1 - right_y) / right_slope + right_x);
	int right_x2 = (int)(((double)y2 - right_y) / right_slope + right_x);
	int left_x1 = (int)(((double)y1 - left_y) / left_slope + left_x);
	int left_x2 = (int)(((double)y2 - left_y) / left_slope + left_x);
	
	// ������ ���� ��ǥ line�� ����Ѵ�.
	line(mark, Point(right_x1, y1), Point(right_x2, y2), Scalar(0, 255, 255), 10);
	line(mark, Point(left_x1, y1), Point(left_x2, y2), Scalar(0, 255, 255), 10);

	// ������ ������ ĥ�Ѵ�.
	Point points[4];
	points[0] = Point(left_x1, y1);
	points[1] = Point(left_x2, y2);
	points[2] = Point(right_x2, y2);
	points[3] = Point(right_x1, y1);
	const Point* pts[1] = { points };
	int npts[] = { 4 };
	// warning ������ ��� ������ ���������� ǥ��
	if (warning) {
		putText(mark, "WARNING", Point(50, 50), FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 255), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 0, 255));
	}
	// safe ������ ��� ������ �ʷϻ����� ǥ��
	else {
		putText(mark, "SAFE", Point(50, 50), FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 255, 0));
	}

	//Predict Direction
	string output;
	double x, threshold = 100;
	double img_center = (double)((frame.cols / 2));
	double right_m = (double)(y1 - y2) / (double)(right_x1 - right_x2);
	double left_m = (double)(y1 - y2) /(double)(left_x1 - left_x2);
	int right_X,left_X,small_right_X, small_left_X;
	if (right_x1 < right_x2) {
		right_X = right_x2; //������ ū x
		small_right_X = right_x1; //������ ���� x
	}
	else {
		right_X = right_x1;
		small_right_X = right_x2;
	}
	if (left_x1 > left_x2) {
		left_X = left_x2;
		small_left_X = left_x1;
	}
	else {
		left_X = left_x1;
		small_left_X = left_x2;
	}
	double down_center = (left_X + right_X) >> 1;
	double up_center = (small_left_X + small_right_X) >> 1;
	double center = (down_center + up_center) /2;
	double down_len = right_X - left_X;
	if ((down_len/width) > 0.7) {
		center += 50;
	}
	//�� ������ �����ϴ� ���� ���
	x = (double)(((right_m * right_x1) - (left_m * left_x1))  / (right_m - left_m));
	//printf("x: %f, center: %f \n", x, center);
	float right_ud_x = (right_X + small_right_X) >> 1;
	float left_ud_x = (left_X + small_left_X) >> 1;
	float ratio = (float)(right_ud_x - x) / (x - left_ud_x);

	float up_ratio = (float)(small_right_X - x) / (x - small_left_X);
	float down_ratio = (float)(right_X - x) / (x - left_X);
	float avg_ratio = (up_ratio + down_ratio)/2;
	printf("ratio: %f\n",  ratio);
	/*if (x >= (center ) && x <= (center)) {
		output = "Straight";
		
	}
	else if (x > center ) {
		output = "Right Turn";
	
	}
	else if (x < center){
		output = "Left Turn";
		
	}*/
	printf("prev_center:%f\n", prev_center);
	if (center < 850) {
		output = changeDir(output, avg_ratio, 0.93, 0.64);
	}
	else if(center < 800) {
		output = changeDir(output, avg_ratio, 0.88, 0.5);
	}
	else {
		output = changeDir(output, avg_ratio, 1.25, 0.75);
	}
	//if (change_lane < 1) {
	//	if (center < 850 && prev_center>850) {
	//		output = changeDir(output, ratio, 0.85, 0.64);
	//		change_lane += 1; //���� ���� Ƚ�� ����
	//	}
	//	else if (center > 850 && prev_center > 850) {
	//		change_lane += 1;
	//		output = changeDir(output, ratio, 1.2, 0.65);
	//	}
	//	else {
	//		output = changeDir(output, ratio, 1.2, 0.65);
	//	}
	//}
	//else {
	//	if (center < 850) {
	//		output = changeDir(output, ratio, 0.75, 0.64);
	//	}
	//	else {
	//		output = changeDir(output, ratio, 1.2, 0.65);
	//	}
	//}
	//printf("change lane Ƚ��: %d\n", change_lane);
	// �̹��� ����� center�� ���� center�� ����
	prev_center = center;

	putText(mark, output, Point(50, 100), FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 255), 3);
	
}


int main(int argc, char** argv) {
	// �������� ����.
	VideoCapture video("clip3.mp4");

	// �������� ��ȿ���� Ȯ���Ѵ�.
	if (!video.isOpened()) {
		cout << "������ ������ ���ų� ã�� �� �����ϴ�." << endl;
		return -1;
	}

	// ���� line ���͸� �ʱ�ȭ�Ѵ�.
	for (int i = 0; i < 10; i++) {
		r_save_lines.push_back(Vec4i(0, 0, 0, 0));
		l_save_lines.push_back(Vec4i(0, 0, 0, 0));
	}

	// ���������κ��� ������ �о� frame�� �ִ´�.
	video.read(frame);

	// ������ ��ȿ���� Ȯ���Ѵ�.
	if (frame.empty()) {
		destroyAllWindows();
		return -1;
	}

	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //���ϴ� �ڵ� ����
	double fps = video.get(CAP_PROP_FPS);  //������
	string filename = "./result.avi";  //��� ����

	writer.open(filename, codec, fps, frame.size(), CV_8UC3);
	if (!writer.isOpened()) {
		cout << "����� ���� ���� ������ �� �� �����ϴ�. \n";
		return -1;
	}

	// ������ ������ �ʴ� ������ ���� �ҷ��´�.
	//double fps = video.get(CAP_PROP_FPS);

	// ������ ������ ������ ����Ѵ�.
	int delay = cvRound(1000 / fps);

	// ������ �ʺ�� ���̸� ����Ѵ�.
	width = frame.cols;   // 1920
	height = frame.rows;  // 1080
	
	
	
	// �������� ���� ������ ������ ����Ѵ�.
	do {
		// ���������κ��� ������ �о� frame�� �ִ´�.
		
		video.read(frame);

		// ������ ��ȿ���� Ȯ���Ѵ�.
		if (frame.empty()) {
			destroyAllWindows();
			break;
		}
		//printf("width:%d height:%d\n", width, height);

		// �÷� ������ ���� �ĺ� ������ ���͸��Ѵ�.
		Mat ROI_colors;
		filterByColor(frame, ROI_colors);

		// RGB ������ �׷��̽����� �������� ��ȯ�Ѵ�.
		cvtColor(ROI_colors, gray, COLOR_BGR2GRAY);

		// ĳ�� ������ �����Ѵ�.
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		Canny(gray, edges, 100, 200);
		namedWindow("Canny Edges", 0);
		imshow("Canny Edges", edges);
		// ������ ������ ������ �����Ѵ�.
		Point points[4];
		points[0] = Point((int)(width * 0.15), (int)(height * 0.8));
		points[1] = Point((int)(width * 0.25), (int)(height * 0.55));
		points[2] = Point((int)(width * 0.65), (int)(height * 0.55));
		points[3] = Point((int)(width * 0.8), (int)(height * 0.8));
		edges = setROI(edges, points);

		namedWindow("ROI Canny Edges",0);
		imshow("ROI Canny Edges", edges);

		//Mat uImage_edges;
		//edges.copyTo(uImage_edges);

		// line�� �����Ѵ�.
		vector<Vec4i> lines;
		HoughLinesP(edges, lines, rho, theta, hough_threshold, minLineLength, maxLineGap);

		// ������ �����Ѵ�.
		Mat mark = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		Mat detected = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		detectLane(mark, detected, lines);
		/*namedWindow("mark", 0);
		imshow("mark", mark);
		namedWindow("detected", 0);
		imshow("detected", detected);
		namedWindow("frame", 0);
		imshow("frame", frame);*/

		// ���� ���� ������ ǥ���Ѵ�.
		mark.copyTo(frame, mark);
		addWeighted(frame, 1, detected, 0.2, 0.0, result);

		// ��� ������ ����Ѵ�.
		namedWindow("final", 0);
		imshow("final", result);
		writer << result;
		char key = waitKey(delay);
		// ESC Ű�� ������ �����Ѵ�.
		// delay ���� ���� ���� ����� ���� ������ �������� ������ ����Ѵ�.
		if (key == 27)
			break;
		// c Ű�� ������ 10������ �ڷ� �̵�
		else if (key == 'c')
			video.set(cv::CAP_PROP_POS_FRAMES, video.get(cv::CAP_PROP_POS_FRAMES) + 10);
			
		// z Ű�� ������ 10������ ������ �̵�
		else if (key == 'z')  
			video.set(cv::CAP_PROP_POS_FRAMES, video.get(cv::CAP_PROP_POS_FRAMES) - 10);
		
	} while (!frame.empty());

	return 0;
}