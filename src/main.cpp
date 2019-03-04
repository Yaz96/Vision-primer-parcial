/*
 * Title: Window Click Example
 * Class: Vision para Robot
 * Instructor: Dr. Jose Luis Gordillo (http://robvis.mty.itesm.mx/~gordillo/)

 * Institution: Tec de Monterrey, Campus Monterrey
 * Date: February, 12 2019
 *
 * Description: 
 *
 * This programs uses OpenCV http://www.opencv.org/
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>




using namespace cv;
using namespace std;

// Here we will store points
vector<Point> points;
vector<Mat> bgr_planes;
vector<Mat> yiq_planes;
vector<Mat> hsv_planes;
Mat currentImage;
Mat YIQim;
Mat HSV_image;
Mat HistB;
Mat HistR;
Mat HistG;
Mat HistY;
Mat HistI;
Mat HistQ;
Mat HistH;
Mat HistS;
Mat HistV;
Mat HistYIQ;
Mat HistHSV;
Mat gauIm;
Mat med,prom;
Mat DetBordes;
Mat lapace,LOG;
Mat aux,aux2,ero;
Mat dit;
Mat grad;
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
int kernel_size = 3;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
int erosion_elem = 0;
int erosion_size = 4;
int dilation_elem = 0;
int dilation_size = 3;
int const max_elem = 2;
int const max_kernel_size = 21;


bool hist=false, flagBck=false,fre=false, desp=false;
int MAX_KERNEL_LENGTH = 31;


// me quede en hacer los histogramas de YIQ, deberiamos hacer una forma en la que podamos tener mas puntos para muestrear mejor
// pero primero debemos intentar hacer lo que estamos haciendo pero con YIQ y ver si elimina el back mejor y con hvl 
void functionHSVtoRGB(const Mat &sourceImage ){
	Mat aux;
	cvtColor(sourceImage,aux,COLOR_HSV2BGR);
	HSV_image=aux;

}




/* This is the callback that will only display mouse coordinates */
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param);
void BN(const Mat &sourceImage, Mat &destinationImage);
void BinImg(const Mat &sourceImage, Mat &destinationImage);
void functionHSV(const Mat &sourceImage, Mat &destinationImage);
void Histos(const Mat &sourceImage,Mat &bhist,Mat &rhist,Mat &ghist);
void histActualizacion(int R, int G, int B, int Y, int I, int Q,int H, int S, int V);
int eraseBack(int counter,int R, int G, int B, int Y, int Q, int I,int H, int V, int S);
void convertImageRGBtoYIQ(const Mat &sourceImage);
void convertImageYIQtoRGB(const Mat &sourceImage);
void gaussianImage(const Mat &sourceImage, Mat &destinationImage);
void Median(const Mat &sourceImage, Mat &destinationImage);
void Edge(const Mat &sourceImage, Mat &destinationImage);
void lapaciano(const Mat &sourceImage, Mat &destinationImage, Mat &realIamge);
void lapacianoOFGaussian(const Mat &sourceImage, Mat &destinationImage, Mat &realIamge,Mat &goodImage);
void promedio(const Mat &sourceImage, Mat &destinationImage);
void Erosion( int, void*,const Mat &sourceImage, Mat &destinationImage );
void Dilation( int, void*,Mat &sourceImage, Mat &destinationImage );
void Gradient(const Mat &sourceImage);
int thr=100;

Mat FiltroKernel(const Mat &src){
	static int ind = 0;
	Mat kernel, dst;
  Point anchor;
  double delta;
  int ddepth;
  int kernel_size;
  char* window_name = "filter2D Demo";

  anchor = Point( -1, -1 );
  delta = 0;
  ddepth = -1;
	 namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	kernel_size = 3 + 2*( ind%5 );
      kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

      /// Apply filter
      filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
	return dst;
	  ind ++ ;

}

int main(int argc, char *argv[])
{
int var,var2;

	/* First, open camera device */
	VideoCapture camera;
    camera.open(0);

	/* Create imavges where captured and transformed frames are going to be stored */

	Mat BNimage;

	Mat bhist,ghist,rhist;

    /* Create main OpenCV window to attach callbacks */
    namedWindow("Image");
    setMouseCallback("Image", mouseCoordinatesExampleCallback);
	
    while (true)
	{
		/* Obtain a new frame from camera */
		if (!fre)

		currentImage = imread("/home/diego96/Desktop/castillo.jpg", CV_LOAD_IMAGE_COLOR);

		
            /* Draw all points */
            //for (int i = 0; i < points.size(); ++i) {
              // circle(currentImage, (Point)points[i], 5, Scalar( 0, 0, 255 ), CV_FILLED);
			  // cout<<(Point)points[i]<<"punt"<<endl;
	//if (i>0){ line(currentImage,(Point)points[i],(Point)points[i-1] , Scalar( 0, 0, 255 ),5,8, 0); }
            //}//for 

            /* Show image */
	var = waitKey(3);
	var2 = var==255 ? var2:var;
	


	if (var2!='f'){
		//cout<<"var2"<<var2<<endl;
	switch(var2){
	case 'g':
	
	BN(currentImage,BNimage);
	imshow("Image", BNimage);
	break;
	case 'b':
	BinImg(currentImage,BNimage);
	imshow("Image", BNimage);
	break;
	case 'p':
	currentImage.copyTo(YIQim);
	convertImageRGBtoYIQ(currentImage);
	BN(YIQim,BNimage);
	imshow("Image", BNimage);
	break;
	case 'O':
	functionHSV(currentImage, HSV_image);	
	BN(HSV_image,BNimage);
	imshow("Image", BNimage);
	break;
	case 'r':
	currentImage.copyTo(YIQim);
	convertImageRGBtoYIQ(currentImage);
	functionHSV(currentImage, HSV_image);
	imshow("Image",currentImage);
	imshow("YIQ", YIQim);
	imshow("HSV",HSV_image);

	break;
	case 'q':
	thr++;
	thr = thr>=255 ? 255:thr;
	BinImg(currentImage,BNimage);
	imshow("Image", BNimage);
	var2 = 'b';
	cout<<thr<<endl;
	break;
	case 'a':
	thr--;
	thr = thr <= 0 ? 0:thr;
	BinImg(currentImage,BNimage);
	imshow("Image", BNimage);
	var2 = 'b';
	cout<<thr<<endl;
	break;
	case 'm':
	cout<<"Ingresa 8 puntos del objeto : "<<endl;
	flagBck = true;
	var2='f';
	break;
	case 9:
	hist = false;
	fre= false;
	flagBck=false;
	var2 = 'r';
	break;
	case 'y':
	imshow("YIQ", YIQim);
	imshow("HistY",HistY);
	imshow("HistI",HistI);
	imshow("HistQ",HistQ);
	imshow("HSV",HSV_image);
	imshow("HistH",HistH);
	imshow("HistS",HistS);
	imshow("HistV",HistV);
	desp=true;
	var2='1';
	break;
    case 'n':
    gaussianImage(BNimage,gauIm);
    imshow("gaussian",gauIm);
    break;
    case 'v':
    Median(BNimage,med);
    imshow("Median",med);
    break;
    case 'u':
    Edge(BNimage,DetBordes);
    imshow("Bordes",DetBordes);
    break;
    case '1':
    lapacianoOFGaussian(currentImage,aux,aux2,LOG);
    imshow("LOG",LOG);
    break;
    case '2':
    lapaciano(BNimage,aux,lapace);
    imshow("lapaciano",lapace);
    break;
	case '3':
    promedio(BNimage,prom);
    imshow("Promedio",prom);
    break;
	case '4':
    Erosion( 0, 0,BNimage, ero );
    imshow("Erosion",ero);
    break;
	case '5':
	Dilation( 0, 0,BNimage, dit );
	imshow("Dilatation", dit);
	break;
	case '6':
	Gradient(BNimage);
	imshow("Gradient", grad);
	break;
	case 7:
	Mat dst;
	FiltroKernel(currentImage);
	imshow( window_name, dst );
	break;
	default:
	imshow("Image",currentImage);
	
	}
	}
	else{
		if (!hist){
		currentImage.copyTo(YIQim);
		convertImageRGBtoYIQ(currentImage);
		functionHSV(currentImage, HSV_image);
		Histos(currentImage,bhist,rhist,ghist);
		var2='f';
		fre = true;
		hist=true;
		//imshow("HistB",HistB);
		//imshow("HistG",HistG);
		//imshow("HistR",HistR);
		
		}

	}
	            
	/* If 'x' is pressed, exit program */
		
	
	
		if (var == 'x') {break;}
		
		
			
		//if outter
		
		
			//cout << "No image data.. " << endl;
	
	}
}


void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
	int R,G,B,Y,I,Q,H,S,V;
	static int counter=0;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            cout << "  Mouse X, Y: " << x << ", " << y <<endl;
			R=static_cast< int >( currentImage.at<Vec3b>(y,x)[2]);//Rjo
			G=static_cast< int >( currentImage.at<Vec3b>(y,x)[1]);//Verde
			B=static_cast< int >( currentImage.at<Vec3b>(y,x)[0]);//Azul
			cout << "R "<< R ;
			cout << " G "<< G ;
			cout << " B "<< B <<endl;

			/*currentImage.at<Vec3b>(y-10, x)[0] = 255;
			currentImage.at<Vec3b>(y-10, x)[1] = 0;
			currentImage.at<Vec3b>(y-10, x)[2] = 0;*/
			if(desp){
			Q=static_cast< int >( YIQim.at<Vec3b>(y,x)[2]);
			I=static_cast< int >( YIQim.at<Vec3b>(y,x)[1]);
			Y=static_cast< int >( YIQim.at<Vec3b>(y,x)[0]);
			cout << "Y "<< Y ;
			cout << " I "<< I ;
			cout << " Q "<< Q <<endl;


			V=static_cast< int >( HSV_image.at<Vec3b>(y,x)[2]);	
			S=static_cast< int >( HSV_image.at<Vec3b>(y,x)[1]);
			H=static_cast< int >( HSV_image.at<Vec3b>(y,x)[0]);
			cout << "H "<< H ;
			cout << " S "<< S ;
			cout << " V "<< V <<endl;

			}

			if (hist)
			histActualizacion( R,G,B,Y,I,Q,H,S,V);
			
			if (flagBck)
			counter=eraseBack(counter,R,G,B,Y,Q,I,H,V,S);

            /*  Draw a point */
            points.push_back(Point(x, y));
            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
    }
}


void BN(const Mat &sourceImage, Mat &destinationImage)
{
	int val,val2;
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols / 2; ++x){
		val=(sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[0]*0.1+sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[1]*0.3+sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[2]*0.6);
		val2=(sourceImage.at<Vec3b>(y, x)[0]*0.1+sourceImage.at<Vec3b>(y, x)[1]*0.3+sourceImage.at<Vec3b>(y, x)[2]*0.6
);
	for (int i = 0; i < sourceImage.channels(); ++i)
			{
destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = val;
			destinationImage.at<Vec3b>(y, x)[i] = val2;
				
			}

		}

}
void BinImg(const Mat &sourceImage, Mat &destinationImage)
{
	int val,val2;
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols / 2; ++x){
		val=(sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[0]+sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[1]+sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[2])/3;
		val2=(sourceImage.at<Vec3b>(y, x)[0]+sourceImage.at<Vec3b>(y, x)[1]+sourceImage.at<Vec3b>(y, x)[2])/3;
	for (int i = 0; i < sourceImage.channels(); ++i)
			{
destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = val>thr ? 255:0;
			destinationImage.at<Vec3b>(y, x)[i] = val2>thr ? 255:0;
				
			}

		}

}



void Histos(const Mat &sourceImage,Mat &bhist,Mat &rhist,Mat &ghist){

Mat yhist,ihist,qhist;
Mat hhist,vhist,shist;
split(YIQim,yiq_planes);
split(sourceImage,bgr_planes);
split(HSV_image,hsv_planes);

int histSize =256;
float range[] = {0,256};
const float* histRange = {range};

calcHist(&bgr_planes[0],1,0,Mat(),bhist,1,&histSize,&histRange,true,false);
calcHist(&bgr_planes[1],1,0,Mat(),ghist,1,&histSize,&histRange,true,false);
calcHist(&bgr_planes[2],1,0,Mat(),rhist,1,&histSize,&histRange,true,false);

calcHist(&yiq_planes[0],1,0,Mat(),yhist,1,&histSize,&histRange,true,false);
calcHist(&yiq_planes[1],1,0,Mat(),ihist,1,&histSize,&histRange,true,false);
calcHist(&yiq_planes[2],1,0,Mat(),qhist,1,&histSize,&histRange,true,false);

calcHist(&hsv_planes[0],1,0,Mat(),hhist,1,&histSize,&histRange,true,false);
calcHist(&hsv_planes[1],1,0,Mat(),shist,1,&histSize,&histRange,true,false);
calcHist(&hsv_planes[2],1,0,Mat(),vhist,1,&histSize,&histRange,true,false);


int histw = 512, histh = 400;
int binw = cvRound( (double) histw/histSize);
Mat histb( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat histr( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat histg( 450, 520, CV_8UC3, Scalar( 0,0,0) );

Mat histy( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat histi( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat histq( 450, 520, CV_8UC3, Scalar( 0,0,0) );

Mat histhh( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat hists( 450, 520, CV_8UC3, Scalar( 0,0,0) );
Mat histv( 450, 520, CV_8UC3, Scalar( 0,0,0) );


  //Normalize the result to [ 0, histImage.rows ]
  normalize(bhist, bhist, 0, histb.rows, NORM_MINMAX, -1, Mat() );
  normalize(ghist, ghist, 0, histg.rows, NORM_MINMAX, -1, Mat() );
  normalize(rhist, rhist, 0, histr.rows, NORM_MINMAX, -1, Mat() );

  normalize(yhist, yhist, 0, histy.rows, NORM_MINMAX, -1, Mat() );
  normalize(ihist, ihist, 0, histi.rows, NORM_MINMAX, -1, Mat() );
  normalize(qhist, qhist, 0, histq.rows, NORM_MINMAX, -1, Mat() );
  
  normalize(hhist, hhist, 0, histhh.rows, NORM_MINMAX, -1, Mat() );
  normalize(vhist, vhist, 0, histv.rows, NORM_MINMAX, -1, Mat() );
  normalize(shist, shist, 0, hists.rows, NORM_MINMAX, -1, Mat() );


  /// Draw for each channel
  int aux=5;
  string staux;
  line( histb, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
    line( histr, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
  line( histg, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );

  line( histy, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
    line( histi, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
  line( histq, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
  
  line( histhh, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
    line( hists, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );
  line( histv, Point(0,400), Point(512,400), Scalar( 255, 255, 255), 2, 8, 0  );

  for( int i = 1; i < histSize; i++ )
  {
      line( histb, Point( binw*(i-1), (histh - cvRound(bhist.at<float>(i-1))) ),
                       Point( binw*(i), histh - (cvRound(bhist.at<float>(i)) ) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );

      line( histg, Point( binw*(i-1), histh - cvRound(ghist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(ghist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histr, Point( binw*(i-1), histh - cvRound(rhist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(rhist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );

 line( histy, Point( binw*(i-1), (histh - cvRound(yhist.at<float>(i-1))) ),
                       Point( binw*(i), histh - (cvRound(yhist.at<float>(i)) ) ),
                       Scalar( 255, 255, 255), 2, 8, 0  );

      line( histi, Point( binw*(i-1), histh - cvRound(ihist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(ihist.at<float>(i)) ),
                       Scalar( 100, 255, 160), 2, 8, 0  );
      line( histq, Point( binw*(i-1), histh - cvRound(qhist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(qhist.at<float>(i)) ),
                       Scalar( 190, 130, 255), 2, 8, 0  );

	line( histhh, Point( binw*(i-1), (histh - cvRound(hhist.at<float>(i-1))) ),
                       Point( binw*(i), histh - (cvRound(hhist.at<float>(i)) ) ),
                       Scalar( 176, 138, 230), 2, 8, 0  );

      line( hists, Point( binw*(i-1), histh - cvRound(shist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(shist.at<float>(i)) ),
                       Scalar( 255, 255, 255), 2, 8, 0  );
      line( histv, Point( binw*(i-1), histh - cvRound(vhist.at<float>(i-1)) ) ,
                       Point( binw*(i), histh - cvRound(vhist.at<float>(i)) ),
                       Scalar( 255, 0, 255), 2, 8, 0  );

		if (i%10==0){
		staux = to_string(aux);
		putText(histb, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histr, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histg, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histy, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histi, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histq, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histhh, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(hists, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histv, staux, cvPoint(binw*(i-1),410), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);

		aux=aux+5;}
		else if (i%5==0){
		staux = to_string(aux);
		putText(histb, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histr, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histg, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histy, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histi, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histq, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histhh, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(hists, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		putText(histv, staux, cvPoint(binw*(i-1),425), FONT_HERSHEY_COMPLEX_SMALL, 0.45, cvScalar(255,255,255), .5, 10);
		aux=aux+5;
		}
  }

HistB = histb;
HistR = histr;
HistG= histg;

HistY = histy;
HistI= histi;
HistQ= histq;

HistH = histhh;
HistS= hists;
HistV= histv;
  /// Display
  /*namedWindow("calcHist DemoR", CV_WINDOW_AUTOSIZE );
	
  imshow("calcHist DemoR", histR );
  	namedWindow("calcHist DemoG", CV_WINDOW_AUTOSIZE );
  imshow("calcHist DemoG", histG );
      namedWindow("calcHist DemoB", CV_WINDOW_AUTOSIZE );

  imshow("calcHist DemoB", histB );
*/
}


void histActualizacion(int R,int G,int B, int Y, int I, int Q,int H, int S, int V){

	
	Mat maB = HistB.clone();
	Mat maG = HistG.clone();
	Mat maR = HistR.clone();
	Mat maY = HistY.clone();
	Mat maI = HistI.clone();
	Mat maQ = HistQ.clone();
	Mat maH = HistH.clone();
	Mat maS = HistS.clone();
	Mat maV = HistV.clone();

	int histh = 400;
	R= R*2-2;
	G=G*2-2;
	B=B*2-2;
	

	line( maB, Point( B, 0 ) ,Point( B,histh ),Scalar( 255, 0, 0), 2, 8, 0  );
	line( maG, Point( G, 0 ) ,Point( G,histh ),Scalar( 0, 255, 0), 2, 8, 0  );
	line( maR, Point( R, 0 ) ,Point( R,histh ),Scalar( 0, 0, 255), 2, 8, 0  );

	imshow("HistB",maB);
	imshow("HistG",maG);
	imshow("HistR",maR);

	if (desp){
	line( maY, Point( Y, 0 ) ,Point( Y,histh ),Scalar( 255, 0, 0), 2, 8, 0  );
	line( maI, Point( I, 0 ) ,Point( I,histh ),Scalar( 0, 255, 0), 2, 8, 0  );
	line( maQ, Point( Q, 0 ) ,Point( Q,histh ),Scalar( 0, 0, 255), 2, 8, 0  );

	line( maH, Point( H, 0 ) ,Point( H,histh ),Scalar( 255, 0, 0), 2, 8, 0  );
	line( maS, Point( S, 0 ) ,Point( S,histh ),Scalar( 0, 255, 0), 2, 8, 0  );
	line( maV, Point( V, 0 ) ,Point( V,histh ),Scalar( 0, 0, 255), 2, 8, 0  );

	imshow("HistY",maY);
	imshow("HistI",maI);
	imshow("HistQ",maQ);

	imshow("HistH",maH);
	imshow("HistS",maS);
	imshow("HistV",maV);

	}



}

int eraseBack(int counter,int R, int G, int B, int Y, int Q, int I,int H,int V,int S){
	int stddevG,stddevR,stddevB,stddevY,stddevI,stddevQ,stddevH,stddevV,stddevS ;
	int thRmx,thRmn,thGmx,thGmn,thBmx,thBmn,thYmx,thYmn,thImx,thImn,thQmx,thQmn,thHmx,thHmn,thVmx,thVmn,thSmx,thSmn;
	static int promR[8],promG[8],promB[8], promY[8],promI[8],promQ[8],promH[8],promV[8],promS[8];
	if (counter == 8){
		cout<<"Punto = "<<counter<<endl;

		
		promR[0]=(promR[1]+promR[2]+promR[3]+promR[4]+promR[5]+promR[6]+promR[7]+promR[8])/8;
		promG[0]=(promG[1]+promG[2]+promG[3]+promG[4]+promG[5]+promG[6]+promG[7]+promG[8])/8;
		promB[0]=(promB[1]+promB[2]+promB[3]+promB[4]+promB[5]+promB[6]+promB[7]+promB[8])/8;

		promY[0]=(promY[1]+promY[2]+promY[3]+promY[4]+promY[5]+promY[6]+promY[7]+promY[8])/8;
		promI[0]=(promI[1]+promI[2]+promI[3]+promI[4]+promI[5]+promI[6]+promI[7]+promI[8])/8;
		promQ[0]=(promQ[1]+promQ[2]+promQ[3]+promQ[4]+promQ[5]+promQ[6]+promQ[7]+promQ[8])/8;

		promH[0]=(promH[1]+promH[2]+promH[3]+promH[4]+promH[5]+promH[6]+promH[7]+promH[8])/8;
		promS[0]=(promS[1]+promS[2]+promS[3]+promS[4]+promS[5]+promS[6]+promS[7]+promS[8])/8;
		promV[0]=(promV[1]+promV[2]+promV[3]+promV[4]+promV[5]+promV[6]+promV[7]+promV[8])/8;

		stddevR= sqrt((pow(promR[1]-promR[0],2) +  pow(promR[2]-promR[0],2) + pow(promR[3]-promR[0],2) + pow(promR[4]-promR[0],2)+ pow(promR[5]-promR[0],2)+ pow(promR[6]-promR[0],2)+ pow(promR[7]-promR[0],2)+ pow(promR[8]-promR[0],2) )/(8-1) );
		stddevG= sqrt((pow(promG[1]-promG[0],2) +  pow(promG[2]-promG[0],2) + pow(promG[3]-promG[0],2) + pow(promG[4]-promG[0],2)+ pow(promG[5]-promG[0],2)+ pow(promG[6]-promG[0],2)+ pow(promG[7]-promG[0],2)+ pow(promG[8]-promG[0],2))/(8-1) );
		stddevB= sqrt((pow(promB[1]-promB[0],2) +  pow(promB[2]-promB[0],2) + pow(promB[3]-promB[0],2) + pow(promB[4]-promB[0],2)+ pow(promB[5]-promB[0],2)+ pow(promB[6]-promB[0],2)+ pow(promB[7]-promB[0],2)+ pow(promB[8]-promB[0],2))/(8-1) );
		
		stddevY= sqrt((pow(promY[1]-promY[0],2) +  pow(promY[2]-promY[0],2) + pow(promY[3]-promY[0],2) + pow(promY[4]-promY[0],2)+ pow(promY[5]-promY[0],2)+ pow(promY[6]-promY[0],2)+ pow(promY[7]-promY[0],2)+ pow(promY[8]-promY[0],2) )/(8-1) );
		stddevI= sqrt((pow(promI[1]-promI[0],2) +  pow(promI[2]-promI[0],2) + pow(promI[3]-promI[0],2) + pow(promI[4]-promI[0],2)+ pow(promI[5]-promI[0],2)+ pow(promI[6]-promI[0],2)+ pow(promI[7]-promI[0],2)+ pow(promI[8]-promI[0],2))/(8-1) );
		stddevQ= sqrt((pow(promQ[1]-promQ[0],2) +  pow(promQ[2]-promQ[0],2) + pow(promQ[3]-promQ[0],2) + pow(promQ[4]-promQ[0],2)+ pow(promQ[5]-promQ[0],2)+ pow(promQ[6]-promQ[0],2)+ pow(promQ[7]-promQ[0],2)+ pow(promQ[8]-promQ[0],2))/(8-1) );
		
		stddevH= sqrt((pow(promH[1]-promH[0],2) +  pow(promH[2]-promH[0],2) + pow(promH[3]-promH[0],2) + pow(promH[4]-promH[0],2)+ pow(promH[5]-promH[0],2)+ pow(promH[6]-promH[0],2)+ pow(promH[7]-promH[0],2)+ pow(promH[8]-promH[0],2))/(8-1) );
		stddevS= sqrt((pow(promS[1]-promS[0],2) +  pow(promS[2]-promS[0],2) + pow(promS[3]-promS[0],2) + pow(promS[4]-promS[0],2)+ pow(promS[5]-promS[0],2)+ pow(promS[6]-promS[0],2)+ pow(promS[7]-promS[0],2)+ pow(promS[8]-promS[0],2))/(8-1) );
		stddevV= sqrt((pow(promV[1]-promV[0],2) +  pow(promV[2]-promV[0],2) + pow(promV[3]-promV[0],2) + pow(promV[4]-promV[0],2)+ pow(promV[5]-promV[0],2)+ pow(promV[6]-promV[0],2)+ pow(promV[7]-promV[0],2)+ pow(promV[8]-promV[0],2))/(8-1) );
		
		thRmx = promR[0]+stddevR;
		thRmn = promR[0]-stddevR;
		thGmx = promG[0]+stddevG;
		thGmn = promG[0]-stddevG;
		thBmx = promB[0]+stddevB;
		thBmn = promB[0]-stddevB;

		thYmx = promY[0]+stddevY;
		thYmn = promY[0]-stddevY;
		thImx = promI[0]+stddevI;
		thImn = promI[0]-stddevI;
		thQmx = promQ[0]+stddevQ;
		thQmn = promQ[0]-stddevQ;

		thHmx = promH[0]+stddevH;
		thHmn = promH[0]-stddevH;
		thSmx = promS[0]+stddevS;
		thSmn = promS[0]-stddevS;
		thVmx = promV[0]+stddevV;
		thVmn = promV[0]-stddevV;

		/*cout<< "thRmx "<< thRmx<< " thRmn "<< thRmn << " promR "<< promR[0]<<endl;
		cout<< "thGmx "<< thGmx<< " thGmn "<< thGmn << " prom "<< promG[0]<<endl;
		cout<< "thmx "<< thBmx<< " thBmn"<< thBmn << "prom"<< promB[0]<<endl;*/

		for (int y = 0; y < currentImage.rows; ++y){
		for (int x = 0; x < currentImage.cols / 2; ++x){
if ( (currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[0] >= thBmn && currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[0] <= thBmx) && 
(currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[1] >= thGmn && currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[1] <= thGmx) &&
(currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[2] >= thRmn && currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[2] <= thRmx) ){

}
else{		
						currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[0] = 0;
						currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[1] = 0;
						currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[2] = 0;
						}

if ( (YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[0] >= thYmn && YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[0] <= thYmx) && 
(YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[1] >= thImn && YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[1] <= thImx) &&
(YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[2] >= thQmn && YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[2] <= thQmx) ){

}
else{
						YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[0] = 0;
						YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[1] = 0;
						YIQim.at<Vec3b>(y, YIQim.cols - 1 - x)[2] = 0;
						}

if ( (HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[0] >= thHmn && HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[0] <= thHmx) && 
(HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[1] >= thSmn && HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[1] <= thSmx) &&
(HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[2] >= thVmn && HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[2] <= thVmx) ){

}
else{
						HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[0] = 0;
						HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[1] = 0;
						HSV_image.at<Vec3b>(y, HSV_image.cols - 1 - x)[2] = 0;
						}

if ((currentImage.at<Vec3b>(y, x)[0] >= thBmn && currentImage.at<Vec3b>(y, x)[0] <= thBmx)&&
(currentImage.at<Vec3b>(y, x)[1] >= thGmn && currentImage.at<Vec3b>(y, x)[1] <= thGmx)&&
(currentImage.at<Vec3b>(y, x)[2] >= thRmn && currentImage.at<Vec3b>(y, x)[2] <= thRmx)){

}
	else{					currentImage.at<Vec3b>(y, x)[0] = 0;
							currentImage.at<Vec3b>(y, x)[1] = 0;
							currentImage.at<Vec3b>(y, x)[2] = 0;
		}
if ((YIQim.at<Vec3b>(y, x)[0] >= thYmn && YIQim.at<Vec3b>(y, x)[0] <= thYmx)&&
(YIQim.at<Vec3b>(y, x)[1] >= thImn && YIQim.at<Vec3b>(y, x)[1] <= thImx)&&
(YIQim.at<Vec3b>(y, x)[2] >= thQmn && YIQim.at<Vec3b>(y, x)[2] <= thQmx)){

}
	else{					YIQim.at<Vec3b>(y, x)[0] = 0;
							YIQim.at<Vec3b>(y, x)[1] = 0;
							YIQim.at<Vec3b>(y, x)[2] = 0;
		}

if ((HSV_image.at<Vec3b>(y, x)[0] >= thHmn && HSV_image.at<Vec3b>(y, x)[0] <= thHmx)&&
(HSV_image.at<Vec3b>(y, x)[1] >= thSmn && HSV_image.at<Vec3b>(y, x)[1] <= thSmx)&&
(HSV_image.at<Vec3b>(y, x)[2] >= thVmn && HSV_image.at<Vec3b>(y, x)[2] <= thVmx)){

}
	else{					HSV_image.at<Vec3b>(y, x)[0] = 0;
							HSV_image.at<Vec3b>(y, x)[1] = 0;
							HSV_image.at<Vec3b>(y, x)[2] = 0;
		}
		}
		}

		counter = 0;
		imshow("Image",currentImage);
		//convertImageYIQtoRGB(YIQim);
		imshow("YIQ", YIQim);
		functionHSVtoRGB(HSV_image);
		imshow("HSV",HSV_image);
		return counter;
	}
	else{
		promR[counter+1]= R;
		promG[counter+1]=G;
		promB[counter+1]=B;

		promY[counter+1] =Y;
		promI[counter+1] =I;
		promQ[counter+1] =Q;

		promH[counter+1] =H;
		promS[counter+1] =S;
		promV[counter+1] =V;

		counter++;
		cout<<"Punto"<<counter<<endl;
		return counter;
	}

}




void convertImageRGBtoYIQ(const Mat &sourceImage)
{
	int r,g,b;
	double ya,i,q;
	int fy,fi,fq;
	int counter = 0;

	// Create a blank YIQ image
	for (int y = 0; y < sourceImage.rows; ++y){
	
		for (int x = 0; x < sourceImage.cols / 2; ++x){
	r=sourceImage.at<Vec3b>(y, x)[2];
	g=sourceImage.at<Vec3b>(y, x)[1];
	b=sourceImage.at<Vec3b>(y, x)[0];

	ya=round(0.299900*r +0.58700*g+0.114000*b);
	i=(0.595716*r -0.274453*g-0.321264*b)/255;
	q=(0.211456*r-0.522591*g+0.311350*b)/255;
	//cout << r <<"R"<<g<<"G"<<b<<"B" <<endl;
	fy = (int) ya;
	fi= (int) round(255*(i+0.523)/1.046);
	fq=(int) round (255*(q+0.596)/1.192);
	//cout<<fy<<" fy "<<fi<<" fi "<<fq<<" fq "<<endl;
		
			
	

			// Set the YIQ pixel components
			YIQim.at<Vec3b>(y, x)[0] =  fy;		// Y component
			YIQim.at<Vec3b>(y, x)[1] = fi;		// I component
			YIQim.at<Vec3b>(y, x)[2] = fq;		// Q component
			

	r=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[2];
	g=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[1];
	b=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[0];

	ya=round(0.299900*r +0.58700*g+0.114000*b);
	i=(0.595716*r -0.274453*g-0.321264*b)/255;
	q=(0.211456*r-0.522591*g+0.311350*b)/255;

	fy = (int) ya;
	fi= (int) round(255*(i+0.523)/1.046);
	fq=(int) round (255*(q+0.596)/1.192);

			// Set the YIQ pixel components
			YIQim.at<Vec3b>(y, currentImage.cols - 1 - x)[0] = fy;		// Y component
			YIQim.at<Vec3b>(y, currentImage.cols - 1 - x)[1] = fi;		// I component
			YIQim.at<Vec3b>(y, currentImage.cols - 1 - x)[2] = fq;		// Q component

		
		
		}
	}

	
	
}


void functionHSV(const Mat &sourceImage, Mat &destinationImage){
	cvtColor(sourceImage,destinationImage,COLOR_BGR2HSV);

}


void convertImageYIQtoRGB(const Mat &sourceImage){
	int r,g,b;
	double ya,i,q;
	int fy,fi,fq;
	int counter = 0;
	Mat YIQaux= sourceImage.clone();

	// Create a blank YIQ image
	for (int y = 0; y < sourceImage.rows; ++y){
	
		for (int x = 0; x < sourceImage.cols / 2; ++x){
	q=sourceImage.at<Vec3b>(y, x)[2];
	i=sourceImage.at<Vec3b>(y, x)[1];
	ya=sourceImage.at<Vec3b>(y, x)[0];
	
	ya= ya/255;
	i= (i*1.046/255)-0.523;
	q = (q*1.192/255)-0.596;

	r=(int) round(ya+0.956*i+0.619*q);
    g=(int) round(ya-0.272*i-0.647*q);
    b=(int) round(ya-1.106*i+1.703*q);
	//cout << r <<"R"<<g<<"G"<<b<<"B" <<endl;

	//cout<<fy<<" fy "<<fi<<" fi "<<fq<<" fq "<<endl;
		
			// Set the YIQ pixel components
			YIQaux.at<Vec3b>(y, x)[0] = b;		// Y component
			YIQaux.at<Vec3b>(y, x)[1] = g;		// I component
			YIQaux.at<Vec3b>(y, x)[2] = r;		// Q component
			

	q=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[2];
	i=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[1];
	ya=currentImage.at<Vec3b>(y, currentImage.cols - 1 - x)[0];

	ya= ya/255;
	i= (i*1.046/255)-0.523;
	q = (q*1.192/255)-0.596;

	r=(int) round(ya+0.956*i+0.619*q);
    g=(int) round(ya-0.272*i-0.647*q);
    b=(int) round(ya-1.106*i+1.703*q);

			// Set the YIQ pixel components
			YIQaux.at<Vec3b>(y, currentImage.cols - 1 - x)[0] = b;		// Y component
			YIQaux.at<Vec3b>(y, currentImage.cols - 1 - x)[1] = g;		// I component
			YIQaux.at<Vec3b>(y, currentImage.cols - 1 - x)[2] = r;		// Q component

		
		
		}
	}

	YIQim=YIQaux.clone();

}

/*void yiq2rgb(double y, double i, double q){
    y=y*255;
    i=i*255;
    q=q*255;
    r2=y+0.956*i+0.619*q;
    g2=y-0.272*i-0.647*q;
    b2=y-1.106*i+1.703*q;
}*/
void gaussianImage(const Mat &sourceImage, Mat &destinationImage)
{
        for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        GaussianBlur( sourceImage, destinationImage, Size( i, i ), 0, 0 );
    }

}
void Median(const Mat &sourceImage, Mat &destinationImage)
{
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    { 
        medianBlur ( sourceImage, destinationImage, i);
    }
    
}
void Edge(const Mat &sourceImage, Mat &destinationImage)
{
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    { 
        Canny( sourceImage, destinationImage, i,i+1);
    }
}
void lapacianoOFGaussian(const Mat &sourceImage, Mat &destinationImage, Mat &realIamge, Mat &goodImage)
{
    GaussianBlur( sourceImage, sourceImage, Size(3,3), 0, 0, BORDER_DEFAULT );
    BN(sourceImage,destinationImage);
 Laplacian( destinationImage, realIamge, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
 convertScaleAbs( realIamge, goodImage );
}

void lapaciano(const Mat &sourceImage, Mat &destinationImage, Mat &realIamge)
{
   
 Laplacian( sourceImage, destinationImage, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
 convertScaleAbs( destinationImage, realIamge );
}
void promedio(const Mat &sourceImage, Mat &destinationImage)
{
	for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
       { 
		   blur( sourceImage, destinationImage, Size( i, i ), Point(-1,-1) );
	   }
}
void Erosion( int, void*,const Mat &sourceImage, Mat &destinationImage )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
  /// Apply the erosion operation
  erode( sourceImage, destinationImage, element );
}
void Dilation( int, void*,Mat &sourceImage, Mat &destinationImage )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( sourceImage, destinationImage, element );
}

void Gradient(const Mat &sourceImage)
{
	GaussianBlur( sourceImage, sourceImage, Size(3,3), 0, 0, BORDER_DEFAULT );
	//Scharr( input_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( sourceImage, grad, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad, grad ); // CV_16S -> CV_8U
 
	//Scharr( input_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	//Sobel( sourceImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	//convertScaleAbs( grad_y, abs_grad_y ); // CV_16S -> // CV_16S -> CV_8U
 
	// create the output by adding the absolute gradient images of each x and y direction
	//addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
}

