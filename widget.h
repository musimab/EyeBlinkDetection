#ifndef WIDGET_H
#define WIDGET_H
#include <QImage>
#include <QThread>
#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    void setup();
signals:
    void sendToggleStream();
    void sendCameraId(const int);

public slots:
    void receiveFrame(const QImage& frame);
    void receiveToggleStream();
    void receiveEyeBlinkNumber(int eyeBlinkNumber);

private:
    Ui::Widget *ui;
    QThread* thread;
    bool statusCamera;
};
#endif // WIDGET_H
