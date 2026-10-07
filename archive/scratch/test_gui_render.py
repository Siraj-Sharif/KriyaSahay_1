import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path("pc/src").resolve()))

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtCore import QTimer

from neurogrip.camera.mock_camera import MockCamera
from neurogrip.communication.mock_serial import MockSerialInterface
from neurogrip.config.settings import AppConfig
from neurogrip.frontend.bridge import NeuroGripBridge
from neurogrip.frontend.image_provider import FrameImageProvider

app = QGuiApplication(sys.argv)
app.setApplicationName("NeuroGrip")

QQuickStyle.setStyle("Basic")
engine = QQmlApplicationEngine()

image_provider = FrameImageProvider()
engine.addImageProvider("neurogrip", image_provider)

camera = MockCamera()
serial = MockSerialInterface()
config = AppConfig.default()

bridge = NeuroGripBridge(
    image_provider=image_provider,
    config=config,
    camera=camera,
    serial=serial,
)
engine.rootContext().setContextProperty("bridge", bridge)

qml_path = Path("pc/src/neurogrip/frontend/qml/Main.qml").resolve()
engine.load(str(qml_path))

root_objects = engine.rootObjects()
if not root_objects:
    print("Failed to load QML")
    sys.exit(1)

window = root_objects[0]
window.show()

bridge.start_pipeline()

def capture():
    # Process events to ensure QML layout & paint
    app.processEvents()
    time.sleep(0.5)
    app.processEvents()
    
    img = window.grabWindow()
    out_path = Path("scratch/gui_snapshot.png").resolve()
    img.save(str(out_path))
    print(f"Saved snapshot to {out_path}, size: {img.width()}x{img.height()}")
    bridge.stop_pipeline()
    app.quit()

QTimer.singleShot(2500, capture)
app.exec()
