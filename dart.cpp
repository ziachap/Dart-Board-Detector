/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// By Zia Siddique (zs13828) & Josh Jones (jj13288)

// header inclusion
#include "/usr/include/opencv2/objdetect/objdetect.hpp"
#include "/usr/include/opencv2/opencv.hpp"
#include "/usr/include/opencv2/core/core.hpp"
#include "/usr/include/opencv2/highgui/highgui.hpp"
#include "/usr/include/opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
std::vector<Rect> detectAndDisplay( Mat frame );
Mat convolve(Mat image, Mat kernel);
Mat grad(Mat dx, Mat dy);
Mat dir(Mat dx, Mat dy);
Mat thresh(Mat mag, int t);
Mat hough_circle(Mat thresh, Mat dir, int t, int minr, int maxr);
void hough_line_space(Mat thresh);
/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

// Function to score how the colours of a bounding box match a dartboard
double colour_score (Mat image, Rect face) {
	int black_score = 0;
	int white_score = 0;
	int border = 20;
	if(face.height < border || face.width < border) border = 0;

	// Loop through pixels in box with border
	for(int i=face.y + border; i < face.y + face.height-border; i++) {
		for(int j=face.x + border; j < face.x + face.width-border; j++) {
			// Threshold green/red
			Vec3b intensity = image.at<Vec3b>(i, j);
			uchar blue = intensity[0];
			uchar green = intensity[1];
			uchar red = intensity[2];

			// black and white detection
			int lower_t = 30;
			int higher_t = 140;
			if (green < lower_t && red < lower_t && blue < lower_t) black_score++;
			else if (green > higher_t && red > higher_t && blue > higher_t && blue < red && blue < green) white_score++;
		}
	}

	// Consider amount and difference between green/red pixels
	double score = 0;
	double numerator = black_score + white_score;
	double divisor = black_score - white_score;
	if (numerator == 0) score = 0;
	else {
		if (divisor < 0) divisor *= -1;
		score = (double)(numerator/pow(divisor,1.5));
	}
	if (score < 0) score *= -1;

	// So zero scores don't zero the total score
	score += 1;

	//printf("White Score: %d, Black Score: %d\n", white_score, black_score);
	return score;
}

// New Line hough function
vector<Vec4i> hough_line(Mat src) {
	Mat output(src.rows,src.cols,CV_32F);
	Mat edge, colourEdge, graySrc;
	int sizes[] = { 1000, 1000, 50 };
	Mat *hspace = new Mat(3, sizes, CV_32FC1, Scalar(0));

	// Detect edges
	Canny(src, edge, 180, 230, 3);
	cvtColor(edge, colourEdge, CV_GRAY2BGR);
	vector<Vec4i> lines;

	// Do hough lines
  HoughLinesP(edge, lines, 1, CV_PI/180, 50, 50, 10 );

	return lines;
}

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat original = imread(argv[1], CV_LOAD_IMAGE_COLOR);;

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces (Viola Jones) and Display Result
	std::vector<Rect> faces = detectAndDisplay( frame );

	// 4. Save Result Image
	//imwrite( "detected.jpg", frame );

	// 5. Hough Transform
	Mat gray_image;
 	cvtColor( original, gray_image, CV_BGR2GRAY );

	// 6. Edge Detection
	Mat kernelx = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat kernely = (Mat_<double>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	Mat dx = convolve(gray_image, kernelx);
	Mat dy = convolve(gray_image, kernely);
	Mat gradient = grad(dx,dy);
	Mat direction = dir(dx,dy);
	Mat threshold = thresh(gradient, 200);

	// 7. Hough Lines
	Mat hough_l;
	vector<Vec4i> lines = hough_line(original);
	//hough_line_space(threshold); // USED FOR HOUGH LINE IMAGE IN REPORT

	// 8. Hough Circles
	Mat hough_c = hough_circle(threshold, direction, 240, 20, 40);

	// 9. Combine Results (ViolaJaaaay!, Line, Circle)
	int max = 0;	// max score for normalisation
	int faces_score[faces.size()];	// array to store bounding box scores

	// Loop through bounding boxes and populate faces_score
	for( int i = 0; i < faces.size(); i++ )
	{
		printf("------ BOX %d ------\n",i);
		int line_count = 0;
		int circle_count = 0;

		// Work out colour score
		double colour_count = colour_score(original, faces[i]);

		//Evaluate all lines
		for(size_t l = 0; l < lines.size(); l++ ){
			Vec4i line = lines[l];

			// Work out line equation (y=mx+c)
			double divisor = line[2]-line[0];
			if (divisor == 0) divisor = 9999;
			double m = (line[3]-line[1])/divisor;
			double c = line[1]-(m*line[0]);

			// Loop though line and determine if it intersects the bounding box
			for (int x = faces[i].x; x <= faces[i].x + faces[i].width; x++){
				int y = (int)((m*x)+c);
				if (y < faces[i].y + faces[i].width && y > faces[i].y){
					line_count++;
					break;
				}
			}
		}

		// Score bounding box based on number of points in circle hough space
		for(int x = faces[i].x; x <= faces[i].x + faces[i].width; x++){
			for(int y = faces[i].y; y <= faces[i].y + faces[i].height; y++){
				float circle_value = hough_c.at<float>(y,x);
				circle_count += circle_value;
			}
		}

		// Find total score and store in score array
		double total_count;
		total_count = (line_count * circle_count) * colour_count;
		faces_score[i] = total_count;
		if(total_count > max) max = total_count;

		// Console Debug
		printf("BOX LINE SCORE: %d\n", line_count);
		printf("BOX CIRCLE SCORE: %d\n", circle_count);
		printf("BOX COLOUR SCORE: %f\n", colour_count);
		printf("-> TOTAL SCORE: %f\n", total_count);
	}

	// 10. NORMALIZE SCORES AND DETECT DARTBOARDS
	float tolerance = 0.6;
	// Set a score minimum in case there are NO dartboards
	printf("\nNormalising Scores: \n");
	if (max > 10000) {
		// Loop through bounding boxes
		for( int i = 0; i < faces.size(); i++ )
		{
			float normalized_score = ((float)faces_score[i])/(float)max;

			if (normalized_score > tolerance) {
				printf("BOX %d - [+] POSITIVE NORMALIZED SCORE: %f\n", i, normalized_score);
				// Draw box
				rectangle(original, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 255, 0, 0 ), 2);
				// DEBUG - Print box score next to box
				//stringstream ss (stringstream::in | stringstream::out);
	  		//ss << normalized_score;
	  		//string score_string = ss.str();
				//putText(original, score_string, Point(faces[i].x, faces[i].y+faces[i].height+30), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, Scalar(255,255,255), 2, 8, false );
			}
			else {
				printf("BOX %d - [-] NEGATIVE NORMALIZED SCORE: %f\n", i, normalized_score);
			}
		}
	}
	imwrite("detected.jpg", original);

	return 0;
}

/** @function detectAndDisplay */
std::vector<Rect> detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found, actually DONT
	//std::cout << faces.size() << std::endl;

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	return faces;
}

// Convolution function
Mat convolve(Mat image, Mat kernel){
		Mat output(image.rows,image.cols,CV_32F);

		int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
		int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;
		Mat paddedInput;
		copyMakeBorder( image, paddedInput,
			kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
			BORDER_REPLICATE );

		for ( int i = 0; i < image.rows; i++ ) {
			for( int j = 0; j < image.cols; j++ ) {
				float sum = 0.0;
				for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
					for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
						// find the correct indices we are using
						int imagex = i + m + kernelRadiusX;
						int imagey = j + n + kernelRadiusY;
						int kernelx = m + kernelRadiusX;
						int kernely = n + kernelRadiusY;

						// get the values from the padded image and the kernel
						int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
						double kernalval = kernel.at<double>( kernelx, kernely );

						// do the multiplication
						sum += imageval * kernalval;
					}
				}
				// set the output value as the sum of the convolution
				output.at<float>(i, j) = (float) sum;
			}
		}

		return output;
	}

	// Gradient magnitude function
	Mat grad(Mat dx, Mat dy) {
		Mat output(dx.rows,dx.cols,CV_32F);

		for(int i=0; i<dx.rows; i++) {
			 for(int j=0; j<dx.cols; j++) {
					float p_dx = dx.at<float>(i,j);
					float p_dy = dy.at<float>(i,j);

					float new_pixel = sqrt((p_dx*p_dx) + (p_dy*p_dy))*0.9;
					output.at<float>(i,j)= new_pixel;
			 }
		 }

		return output;
	}

	// Gradient direction function
	Mat dir(Mat dx, Mat dy) {
		Mat output(dx.rows,dx.cols,CV_32F);

		for(int i=0; i<dx.rows; i++) {
			 for(int j=0; j<dx.cols; j++) {
					float p_dx = dx.at<float>(i,j);
					float p_dy = dy.at<float>(i,j);

					float new_pixel = atan(p_dy / p_dx);
					output.at<float>(i,j)= new_pixel;

			 }
		 }

		return output;
	}

	// Threshold function
Mat thresh(Mat mag, int t) {
		Mat output(mag.rows,mag.cols,CV_32F);

		for(int i=0; i<mag.rows; i++) {
			 for(int j=0; j<mag.cols; j++) {
					float p_mag = mag.at<float>(i,j);
					if (p_mag > t){
						output.at<float>(i,j)= 255;
					} else {
						output.at<float>(i,j)= 0;
					}
			 }
		 }

		return output;
	}

// Circle hough function
Mat hough_circle(Mat thresh, Mat dir, int t, int minr, int maxr) {
		int sizes[] = { 1000, 1000, 50 };
		Mat *hspace = new Mat(3, sizes, CV_32FC1, Scalar(0));

			for(int i=0; i<thresh.rows; i++) {
			 for(int j=0; j<thresh.cols; j++) {
					float p_thresh = thresh.at<float>(i,j);
					// If white...
					if (p_thresh > 200){
						float p_dir = dir.at<float>(i,j);
								for (double r = minr; r <= maxr; r += 0.5){
									// Hough circle formula
									int x0 = (int)(j + r*cos(p_dir));
									int y0 = (int)(i + r*sin(p_dir));
									if (x0 >= 0 && y0 >= 0) {
										hspace->at<float>(y0,x0,r) += 1; // Add vote to hough space
									}

									x0 = (int)(j - r*cos(p_dir));
									y0 = (int)(i - r*sin(p_dir));
									if (x0 >= 0 && y0 >= 0) {
										hspace->at<float>(y0,x0,r) += 1; // Add vote to hough space
									}
								}
					}
			 }
		 }



		// Sum votes along radius dimension, then write image
		Mat himage(thresh.rows,thresh.cols,CV_32F);
		for(int i=0; i<thresh.rows; i++) {
			for(int j=0; j<thresh.cols; j++) {
				int sum = 0;
				for (int r = 0; r <= maxr; r++){
					sum =+ hspace->at<float>(i,j,r); // Add vote to hough space
				}
				himage.at<float>(i,j) = sum*50;
			}
		}
		//imwrite("Hough.jpg", himage);

		// Threshold hough image
		Mat thimage(himage.rows,himage.cols,CV_32F);
		for(int i=0; i<himage.rows; i++) {
			 for(int j=0; j<himage.cols; j++) {
					float p_himage = himage.at<float>(i,j);
					if (p_himage > 254){
						thimage.at<float>(i,j)= 255;
					} else {
						thimage.at<float>(i,j)= 0;
					}
			 }
		 }

		 return thimage;
}

// Writes to file the hough line space
void hough_line_space(Mat thresh) {
	//printf("Writing Hough Line Space...\n");
	//Mat output(thresh.rows,thresh.cols,CV_32F);
	int sizes[] = { 2000, 2000 };
	Mat *hspace = new Mat(2, sizes, CV_32FC1, Scalar(0));
	int max_p = 0;
	// Loop through thresh pixels
	for(int i=0; i<thresh.rows; i++) {
		for(int j=0; j<thresh.cols; j++) {
			float p_thresh = thresh.at<float>(i,j);
			// If white...
			if (p_thresh > 250){
				for(int theta = 0; theta < 360; theta++) {
					float p_angle = (theta * (M_PI/180));
					// Hough line formula
					int p = (int)((j*cos(p_angle)) + (i*sin(p_angle)));
					if (p >= 0) {
						if (p > max_p) max_p = p;
						hspace->at<float>(p,theta) += 1; // Add vote to hough space

					}
				}
			}
		 }
	 }

	// copy from hough space to Mat image
	Mat himage(max_p,360,CV_32F);
	for(int p=0; p<max_p; p++) {
		for(int p_dir=0; p_dir<360; p_dir++) {
			himage.at<float>(p,p_dir) = hspace->at<float>(p,p_dir);
		}
	}

	//imwrite("Hough_Line.jpg", himage);
	//printf("Done! (./Hough_Line.jpg) \n");
}
