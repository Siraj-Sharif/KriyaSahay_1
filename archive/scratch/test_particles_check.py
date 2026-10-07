import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()
qml = b'''
import QtQuick 2.15
import QtQuick.Particles 2.15

Rectangle {
    width: 600
    height: 400
    color: "#000000"
    ParticleSystem { id: sys }
}
'''
engine.loadData(qml)
if engine.rootObjects():
    print("QtQuick.Particles 2.15 IS AVAILABLE!")
else:
    print("QtQuick.Particles 2.15 FAILED")
