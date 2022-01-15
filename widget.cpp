#include "widget.h"
#include "ui_widget.h"
#include "facedetectorworker.h"
#include <QTimer>
#include <QDebug>


Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget),statusCamera(false)
{
    ui->setupUi(this);
    //dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    ui->label->setScaledContents(true);
    setup();
}

Widget::~Widget()
{
    thread->quit();
    while(!thread->isFinished());
    delete thread;
    delete ui;
}

void Widget::setup()
{
    thread = new QThread();
    QTimer* workerTrigger = new QTimer();
    FaceDetectorWorker* faceWorker = new FaceDetectorWorker();

    workerTrigger->setInterval(1);

    connect(workerTrigger, &QTimer::timeout, faceWorker, &FaceDetectorWorker::receiveGrabFrame);
    connect(this, &Widget::sendCameraId, faceWorker, &FaceDetectorWorker::receiveCameraId);
    connect(this, &Widget::sendToggleStream, faceWorker, &FaceDetectorWorker::receiveToggleStream);
    connect(ui->toggleButton, &QPushButton::clicked, this, &Widget::receiveToggleStream);
    connect(faceWorker, &FaceDetectorWorker::sendFrame, this, &Widget::receiveFrame);
    connect(faceWorker, &FaceDetectorWorker::SendeyeBlinkNumber, this, &Widget::receiveEyeBlinkNumber);
    connect(thread, &QThread::finished, faceWorker, &QThread::deleteLater);
    connect(thread, &QThread::finished, workerTrigger, &QTimer::deleteLater);
    connect(thread, SIGNAL(started()), workerTrigger, SLOT(start()));

    workerTrigger->start();
    faceWorker->moveToThread(thread);
    workerTrigger->moveToThread(thread);

    thread->start();

    emit sendCameraId(0);


}


void Widget::receiveFrame(const QImage& frame)
{
    ui->label->setPixmap(QPixmap::fromImage(frame));
}

void Widget::receiveToggleStream()
{

    if(!ui->toggleButton->text().compare("stop"))
        ui->toggleButton->setText("run");
    else
        ui->toggleButton->setText("stop");
    emit sendToggleStream();
}

void Widget::receiveEyeBlinkNumber(int eyeBlinkNumber)
{

    //ui->eyeBlinkNum->setText(QString::number(eyeBlinkNumber));
    ui->lcdNumber->display(QString::number(eyeBlinkNumber));
}

