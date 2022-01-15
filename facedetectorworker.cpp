#include "facedetectorworker.h"
#include <QDebug>
#include <QThread>


FaceDetectorWorker::FaceDetectorWorker(QObject *parent) : QObject(parent),m_toggleStream(false)
{
    m_cap = new cv::VideoCapture();
    faceDetector = dlib::get_frontal_face_detector();
    //load face landmark model
    dlib::deserialize("/home/mustafa/Desktop/Dlib_Examples/FacialKeyMarksDetection/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;
}

FaceDetectorWorker::~FaceDetectorWorker()
{
    if(m_cap->isOpened())
        m_cap->release();
    delete m_cap;
}

void FaceDetectorWorker::receiveGrabFrame()
{

    if(!m_toggleStream) return;

    (*m_cap) >> m_frame_original;
    if(m_frame_original.empty())
        return;
    frameCounter++;
    cv::resize(m_frame_original, m_frameProcessed, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
    process(m_frameProcessed);

    QImage output = cvMatToQImage(m_frameProcessed);
    emit sendFrame(output);

}

void FaceDetectorWorker::receiveToggleStream()
{

    m_toggleStream =! m_toggleStream;
}

void FaceDetectorWorker::receiveCameraId(const int device)
{
    if(m_cap->isOpened())
        m_cap->release();
    m_cap->open(device);
}

void FaceDetectorWorker::process(cv::Mat &frame)
{
    std::vector<dlib::full_object_detection> FaceLandMarks;

    findFaceLandmarksAndDrawPolylines(frame, FaceLandMarks);


    // Find size of FaceLandMarks and calculate ear for each faces
    numberOfFace = FaceLandMarks.size();

    for(auto& faceLandmark: FaceLandMarks) {

        auto [LeftEyePoints, RightEyePoints] = getEyePoints(frame, faceLandmark);

                addCircleToEyePoints(frame, LeftEyePoints, RightEyePoints);

                double leftEAR = calculateEyeAspectRatio(LeftEyePoints);
                double rightEAR = calculateEyeAspectRatio(RightEyePoints);

                ear = (leftEAR + rightEAR)/ 2.0;

                if(ear < EYE_AR_THRESH) {
            COUNTER ++;
        }
        else {
            if(COUNTER >= EYE_AR_THRESH_FRAMES)
                TOTAL++;
            COUNTER = 0;
        }


    }

    ear_s = std::to_string(ear);
    blink_s = std::to_string(TOTAL);
    fps_s = std::to_string(fps);
    numberOfFace_s = std::to_string(numberOfFace);

    //draw fps on the frame
    cv::putText(frame, "ear:" + ear_s, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);
    cv::putText(frame, "blink:" + blink_s, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);
    //cv::putText(frame, "fps:" + fps_s, cv::Point(300, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);
    //cv::putText(frame, "Faces:" + numberOfFace_s, cv::Point(300, 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 5);

    emit SendeyeBlinkNumber(TOTAL);

}

QImage  FaceDetectorWorker::cvMatToQImage( const cv::Mat &inMat )
{
    switch ( inMat.type() )
    {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
        QImage image( inMat.data,
                      inMat.cols, inMat.rows,
                      static_cast<int>(inMat.step),
                      QImage::Format_ARGB32 );

        return image;
    }

        // 8-bit, 3 channel
    case CV_8UC3:
    {
        QImage image( inMat.data,
                      inMat.cols, inMat.rows,
                      static_cast<int>(inMat.step),
                      QImage::Format_RGB888 );

        return image.rgbSwapped();
    }

        // 8-bit, 1 channel
    case CV_8UC1:
    {
#if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
        QImage image( inMat.data,
                      inMat.cols, inMat.rows,
                      static_cast<int>(inMat.step),
                      QImage::Format_Grayscale8 );
#else
        static QVector<QRgb>  sColorTable;

        // only create our color table the first time
        if ( sColorTable.isEmpty() )
        {
            sColorTable.resize( 256 );

            for ( int i = 0; i < 256; ++i )
            {
                sColorTable[i] = qRgb( i, i, i );
            }
        }

        QImage image( inMat.data,
                      inMat.cols, inMat.rows,
                      static_cast<int>(inMat.step),
                      QImage::Format_Indexed8 );

        image.setColorTable( sColorTable );
#endif

        return image;
    }

    default:
        qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
        break;
    }

    return QImage();
}

void FaceDetectorWorker::findFaceLandmarksAndDrawPolylines(cv::Mat &frame, std::vector<dlib::full_object_detection>&FaceLandMarks)
{
    //define to hold detected faces
    std::vector<dlib::rectangle> faces;


    //change to dlib image format
    dlib::cv_image<dlib::bgr_pixel> dlibImage(frame);

    //detect faces at interval of skipFrames
    if(frameCounter % skipFrames == 0){
        faces = faceDetector(dlibImage);
    }


    //loop over faces
    for(int i=0; i<faces.size(); i++){

        //scale the rectangle coordinates as we did face detection on resized smaller image
        dlib::rectangle rect(faces[i].left() ,faces[i].top(), faces[i].right(), faces[i].bottom());

        //Face landmark detection
        dlib::full_object_detection faceLandmark = landmarkDetector(dlibImage, rect);
        //draw poly lines around face landmarks
        //drawPolylines(frame, faceLandmark);
        FaceLandMarks.push_back(faceLandmark);

    }

}

std::tuple<std::vector<cv::Point>, std::vector<cv::Point> > FaceDetectorWorker::getEyePoints(cv::Mat &frame, dlib::full_object_detection& landmarks)
{
       std::vector<cv::Point>LeftEyePoints;
       std::vector<cv::Point>RightEyePoints;
       cv::Mat LeftEyeHull, RightEyeHull;

       for(int i=36; i<=41; i++) {
           LeftEyePoints.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));

       }

       for(int i=42; i<=47; i++) {
           RightEyePoints.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));

       }

       auto EyePoints = std::make_tuple(LeftEyePoints, RightEyePoints);
       /*
       cv::convexHull(LeftEyePoints, LeftEyeHull);
       cv::convexHull(RightEyePoints, RightEyeHull);
       cv::drawContours(frame, LeftEyeHull, -1, cv::Scalar(0,255,0),1);
       cv::drawContours(frame, RightEyeHull, -1, cv::Scalar(0,255,0),1);
       */

       return EyePoints;
}

void FaceDetectorWorker::drawPolylines(cv::Mat &image, dlib::full_object_detection& landmarks)
{
    drawPolyline(image, landmarks, 0, 16,  false);      //jaw line
    drawPolyline(image, landmarks, 17, 21, false);      //left eyebrow
    drawPolyline(image, landmarks, 22, 26, false);      //right eyebrow
    drawPolyline(image, landmarks, 27, 30, false);      //Nose bridge
    drawPolyline(image, landmarks, 30, 35, true);       //lower nose
    drawPolyline(image, landmarks, 36, 41, true);       //left eye
    drawPolyline(image, landmarks, 42, 47, true);       //right eye
    drawPolyline(image, landmarks, 48, 59, true);       //outer lip
    drawPolyline(image, landmarks, 60, 67, true);       //inner lip
}

double FaceDetectorWorker::dist_euclidean(cv::Point& p1, cv::Point& p2)
{
   return sqrt( pow( (p1.x - p2.x), 2) + pow( (p1.y-p2.y),2 ));
}

double FaceDetectorWorker::calculateEyeAspectRatio(std::vector<cv::Point>& points)
{
    float A = dist_euclidean(points[1], points[5] );
    float B = dist_euclidean(points[2], points[4] );
    float C = dist_euclidean(points[0], points[3] );
    float ear = (A+B)/(2.0 * C);
    return ear;
}

void FaceDetectorWorker::drawPolyline(cv::Mat &image, dlib::full_object_detection& landmarks, int start, int end, bool isClosed)
{
    std::vector<cv::Point> points;
    for(int i=start; i<=end; i++){
        points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
    }
    cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
}

void FaceDetectorWorker::addCircleToEyePoints(cv::Mat &frame, std::vector<cv::Point>& LeftEyePoints, std::vector<cv::Point>& RightEyePoints)
{
    std::for_each(LeftEyePoints.begin(), LeftEyePoints.end(), [&frame](cv::Point& points) {
        cv::circle(frame, points, 1, cv::Scalar(0,255,255),2 );
    });

    std::for_each(RightEyePoints.begin(), RightEyePoints.end(), [&frame](cv::Point& points) {
        cv::circle(frame, points, 1, cv::Scalar(0,255,255),2 );
    });
}

