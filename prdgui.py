from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PIL import Image
import sys


class Window(QMainWindow):
    def __init__(self, worker_func, width, height):
        super().__init__()

        self.acceptDrops()

        # set the title
        self.setWindowTitle("progrockdiffusion")

        # setting  the geometry of window
        self.setGeometry(500, 500, width, height)

        # creating label
        self.label = QLabel(self)

        # loading image
        self.pixmap = QPixmap(width, height)

        # adding image to label
        self.label.setPixmap(self.pixmap)
        # self.label.setText('')

        # Optional, resize label to image size
        self.label.resize(self.pixmap.width(), self.pixmap.height())

        self.worker_func = worker_func

        QTimer.singleShot(0, self.runWorker)

        # show all the widgets
        self.show()

    def runWorker(self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.worker_func)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.close)
        # self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

    def setPixmap(self, img):
        pixmap = pil2pixmap(img)
        self.label.setPixmap(pixmap)

    def setText(self, txt):
        self.label.setText(txt)


window = None


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(QPixmap)

    def __init__(self, worker_func):
        super().__init__()
        self.func = worker_func

    def run(self):
        self.func()
        self.finished.emit()


def run_gui(worker_func, width, height):
    # create pyqt5 app
    App = QApplication(sys.argv)

    # create the instance of our Window
    global window
    window = Window(worker_func, width, height)

    # start the app
    sys.exit(App.exec())


def update_image(img):
    window.setPixmap(img)


def pil2pixmap(im):
    if im.mode == "RGB":
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
    elif im.mode == "RGBA":
        r, g, b, a = im.split()
        im = Image.merge("RGBA", (b, g, r, a))
    elif im.mode == "L":
        im = im.convert("RGBA")
    # Bild in RGBA konvertieren, falls nicht bereits passiert
    im2 = im.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    pixmap = QPixmap.fromImage(qim)
    return pixmap
