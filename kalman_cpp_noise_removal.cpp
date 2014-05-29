#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
int main()
{
Mat input=imread("Image1.JPG",CV_LOAD_IMAGE_GRAYSCALE);
input.convertTo(input,CV_32F);


Mat noise(input.rows,input.cols,CV_32F);
input.convertTo(input,CV_32F);


float var=20; 
float Q=20;		// process noise associated with the control unit....
float R=40;    // measurement noise's covariance


randn(noise, 0,var);
Mat noisy_inp= noise + input;
Mat out_est=Mat::zeros(input.rows,input.cols, CV_32F); 
Mat A=(Mat_<float>(3,3)<<0.3,0.3,0.4,0.0,1.0,0.0,0.0,0.0,1.0); 
input.row(1).copyTo(out_est.row(1));// copying source row 1 to output row 1
input.col(1).copyTo(out_est.col(1));// copying source column 1 to output column  
1
Mat H=(Mat_<float>(1,3)<<1,0,0);
Mat I = Mat::eye(3, 3,CV_32F);
float a=0,b=0,c=0;
a=out_est.at<float>(1,0);
b=out_est.at<float>(0,1);
c=out_est.at<float>(1,1);
//Initialization of input mean and covariance
Mat x_post=(Mat_<float>(3,1)<<a,b,c);
Mat cov_post=x_post*x_post.t();
//Kalman update equations
for(int i=1;i< input.rows;i++)
{
for(int j=1;j< input.cols;j++)
{
Mat x_prior=A*x_post;
Mat cov_prior=(A*cov_post*A.t())+Q;
Mat  k_gain  =cov_prior*H.t()*((H*cov_prior*H.t())
+R).inv();
float zk = noisy_inp.at<float>(i,j);
x_post= x_prior+ (k_gain*(zk ­(H*x_prior)));
cov_post = (I­(k_gain*H))* cov_prior;
out_est.at<float>(i,j)= x_post.at<float>(1,1);
}
}
noisy_inp.convertTo(noisy_inp,CV_8U);
out_est.convertTo(out_est,CV_8U);
input.convertTo(input,CV_8U);
namedWindow("Input Image", CV_WINDOW_KEEPRATIO);
imshow("Input Image",input);
waitKey(0);
namedWindow("Noisy Image", CV_WINDOW_KEEPRATIO);
imshow("Noisy Image", noisy_inp);
waitKey(0);
namedWindow("Estimated Image", CV_WINDOW_KEEPRATIO);
imshow("Estimated Image", out_est);
waitKey(0);
return(0);
}