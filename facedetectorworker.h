#ifndef FACEDETECTORWORKER_H
#define FACEDETECTORWORKER_H

#include <QObject>
#include <QImage>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/image_processing.h>
#include<dlib/opencv.h>

class FaceDetectorWorker : public QObject
{
    Q_OBJECT
public:
    explicit FaceDetectorWorker(QObject *parent = nullptr);
    ~FaceDetectorWorker();
    QImage cvMatToQImage( const cv::Mat &inMat );

    void findFaceLandmarksAndDrawPolylines(cv::Mat &frame, std::vector<dlib::full_object_detection>&FaceLandMarks);

    std::tuple<std::vector<cv::Point>, std::vector<cv::Point>> getEyePoints(cv::Mat& frame,
                                                        dlib::full_object_detection& landmarks);
    void drawPolylines(cv::Mat &image, dlib::full_object_detection& landmarks);
    double dist_euclidean(cv::Point& p1, cv::Point& p2);
    double calculateEyeAspectRatio(std::vector<cv::Point>& points);
    void drawPolyline(cv::Mat &image, dlib::full_object_detection& landmarks, int start, int end, bool isClosed);
    void addCircleToEyePoints(cv::Mat& frame, std::vector<cv::Point>&LeftEyePoints, std::vector<cv::Point>& RightEyePoints);

signals:
    void sendFrame(const QImage& qimage);
    void SendeyeBlinkNumber(int eyeBlinkNumber);

public slots:
    void receiveGrabFrame();
    void receiveToggleStream();
    void receiveCameraId(const int id);
    void process(cv::Mat& frame);
private:
    cv::Mat m_frame_original;
    cv::Mat m_frameProcessed;
    cv::VideoCapture *m_cap;
    bool m_toggleStream;

    //define the face detector
    dlib::frontal_face_detector faceDetector;

    //define landmark detector
    dlib::shape_predictor landmarkDetector;

    //define to hold detected faces
    //std::vector<dlib::rectangle> faces;


    //Initialize eye aspect ratio and blink parameters

    double ear {0};
    int numberOfFace = {0};
    const float EYE_AR_THRESH  {0.3};
    const float EYE_AR_THRESH_FRAMES {3};
    int COUNTER {0};
    int TOTAL {0};

    //define skip frames
    int skipFrames {1};
    //initiate the tickCounter
    double tick = cv::getTickCount();
    int frameCounter {0};

    //create variable to store fps
    double fps {30.0};

    cv::String ear_s,blink_s, fps_s, numberOfFace_s;
    float FACE_DOWNSAMPLE_RATIO {2};
};

#endif // FACEDETECTORWORKER_H
