from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtGui import QImage, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QSize, QCoreApplication
import sys

class TestProvider(QQuickImageProvider):
    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Image)

    def requestImage(self, id, size, requestedSize):
        print(f"requestImage called! id={id}, size={size}, requestedSize={requestedSize}")
        img = QImage(100, 100, QImage.Format.Format_RGB888)
        img.fill(0xFF0000)
        # Check PySide6 signature expectation
        if size is not None and hasattr(size, "setWidth"):
            size.setWidth(100)
            size.setHeight(100)
        return img

app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()
p = TestProvider()
engine.addImageProvider("test", p)
engine.loadData(b"import QtQuick 2.15\nImage { source: \"image://test/live\" }\n")
app.processEvents()
