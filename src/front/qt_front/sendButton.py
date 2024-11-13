import sys
import os
from PySide2.QtWidgets import QWidget
from PySide2.QtCore import Qt, Slot
from PySide2.QtGui import QPainter, QColor, QPixmap
from PySide2.QtCore import QPoint, QTimer, Signal

class SendButton(QWidget):
    sendMessageSignal = Signal()
    def __init__(self, parent=None):

        super().__init__(parent)
        #self.setGeometry(100, 100, 200, 50)
        self.setFixedSize(50, 50)
        self.setStyleSheet("background-repeat: no-repeat; border-radius: 25px;")
        self._png1 = os.path.abspath((os.path.curdir)) + u"\icon\icon_button\sendmsg1.png"
        self._png2 = os.path.abspath((os.path.curdir)) + u"\icon\icon_button\sendmsg2.png"
        self.pix1 = QPixmap(self._png1)
        self.pix2 = QPixmap(self._png2)
        self.pix1 = self.pix1.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.pix2 = self.pix2.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.up = True
        self.hover = True

        self.point = QPoint(0, 0)
        self.timer1 = QTimer(self)
        self.timer1.setInterval(180)
        self.timer1.timeout.connect(self.timeout1)
        self.timer2 = QTimer(self)
        self.timer2.setInterval(180)
        self.timer2.timeout.connect(self.timeout2)
        self.timer1.start()



    def paintEvent(self, event):
        # Custom painting code here
        p = QPainter(self)
        if self.hover:
            p.drawPixmap(self.point, self.pix2)
        else:
            p.drawPixmap(self.point, self.pix1)
        pass
        return super().paintEvent(event)

    def mousePressEvent(self, event):
        # Handle mouse press events
        self.startAnimationSlot()
        self.sendMessageSignal.emit()
        pass
        return super().mousePressEvent(event)

    @Slot()
    def startAnimationSlot(self):
        # Start the animation
        self.timer1.stop()
        self.timer2.start()
        pass

    def enterEvent(self, event):
        # Handle mouse enter events
        self.setCursor(Qt.PointingHandCursor)
        self.hover = True
        self.update()
        return super().enterEvent(event)
        pass

    def leaveEvent(self, event):
        # Handle mouse leave events
        self.setCursor(Qt.ArrowCursor)
        self.hover = False
        self.update()
        return super().leaveEvent(event)
        pass

    @Slot()
    def timeout1(self):
        # Handle timer timeout
        if self.point.y() == 0 + 2 or self.point.y() == 0 - 2:
            self.up = not self.up

        if self.up:
            self.point.setY(int(self.point.y()) + 1)
        else:
            self.point.setY(int(self.point.y()) - 1)

        self.update()
        pass

    @Slot()
    def timeout2(self):
        # Handle timer timeout
        if(self.point.x() > 60):
            self.point.setX(0)
            self.point.setY(0)
            self.timer2.stop()
            self.timer1.start()

        self.point.setX(self.point.x()+5)
        self.point.setY(self.point.y()-1)

        self.update()
        pass
