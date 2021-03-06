QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17


# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += /home/mustafa/Downloads/dlib-19.22/dlib/all/source.cpp \
    facedetectorworker.cpp
INCLUDEPATH +=/home/mustafa/Downloads/dlib-19.22


LIBS += /home/mustafa/Downloads/dlib-19.22/build/dlib/libdlib.a
LIBS += -pthread
CONFIG += link_pkgconfig
PKGCONFIG += x11

#CONFIG += link_pkgconfig
PKGCONFIG += opencv4

SOURCES += \
    main.cpp \
    widget.cpp

HEADERS += \
    facedetectorworker.h \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
