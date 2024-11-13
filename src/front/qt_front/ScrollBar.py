from PySide2.QtWidgets import QWidget
from PySide2.QtCore import QEvent, Qt
from PySide2.QtCore import QTimer
from PySide2.QtGui import QColor, QMouseEvent, QPaintEvent, QResizeEvent, QPainter, QPainterPath, QPen, QBrush, QPalette, QWheelEvent
from PySide2.QtCore import QSize, QPoint
from PySide2.QtCore import Signal
from PySide2.QtCore import QRect
from PySide2.QtCore import Slot

WIDTH = 10# 拖动条宽度
SIZE_ALTER = 1# 拖动条每次变化的长度
SHOW_NOW_MESSAGE = 30# 显示最新消息的阀值

class ScrollBar(QWidget):


    toBottomSignal = Signal()
    toTopSignal = Signal()
    toCancelSignal = Signal()
    moveSignal = Signal(int)
    
    def __init__(self, height: int, parent: QWidget  = None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.timeOutSlot)

        self.length = height - WIDTH
        self.color = QColor("#87CEFF")
        self.color.setAlpha(0)
        self.oldsize = QSize(-1,-1)

        self.is_drag = False
        self.drag = QPoint(0,0)
        self.pos_y = 0
        self.length = 0



    def resize(self, size: QSize) -> None:
        #####self.resize(WIDTH, size.height())
        self.move(size.width() - 15, 0)
        return super().resize(WIDTH, size.height())

    def setLength(self):
        if self.length > 0:
            self.length -= SIZE_ALTER
            if self.pos_y+self.length+SHOW_NOW_MESSAGE>self.height()-WIDTH:
                if not self.is_drag:
                    self.toBottom()
            self.update()

    def isBottom(self):
        return self.pos_y == self.height() - WIDTH - self.length
    
    def toBottom(self):
        self.toBottomSignal.emit()
        self.pos_y = self.height() - WIDTH - self.length


    def resizeEvent(self, event: QResizeEvent) -> None:
        if self.oldsize.height() == -1:
            self.oldsize = event.size()
            return super().resizeEvent(event)
        
        self.length = self.length + event.size().height() - self.oldsize.height()
        self.oldsize = event.size()
        return super().resizeEvent(event)
    
    def paintEvent(self, event: QPaintEvent) -> None:
        pai = QPainter(self)
        pai.setRenderHint(QPainter.Antialiasing, True)
        pai.setPen(Qt.NoPen)
        pai.setBrush(self.color)

        pai.drawChord(QRect(0, self.pos_y, WIDTH, WIDTH), 0, 180 * 16)
        pai.drawRect(0, self.pos_y + WIDTH / 2, WIDTH, self.length)
        pai.drawChord(QRect(0, self.pos_y + self.length, WIDTH, WIDTH), 180 * 16, 180 * 16)
        return super().paintEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.pos().x() > WIDTH:
                return
            self.is_drag = True
            self.drag = event.pos()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.is_drag = False
        
        return super().mouseReleaseEvent(event)
    
    def scrollMove(self, y: int):
        if y<=0:
            self.toTopSignal.emit()
            self.pos_y = 0
        elif y+self.length>self.height()-WIDTH:
            self.toBottomSignal.emit()
            self.pos_y = self.height() - WIDTH - self.length
        else:
            self.toCancelSignal.emit()
            self.pos_y = y
        return
    
    @Slot(int)
    def wheelEventSlot(self, moveCount: int):
        if moveCount < 0:
            return
        old_y = self.pos_y
        temp_pos_y = self.pos_y + moveCount
        self.scrollMove(temp_pos_y)
        self.moveSignal.emit(self.pos_y - old_y)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.color.setAlpha(180)
        self.timer.stop()
        self.timer.start()

        if self.is_drag:
            old_y = self.pos_y
            move_y = event.pos().y() - self.drag.y()
            temp_pos_y = self.pos_y + move_y

            self.scrollMove(temp_pos_y)
            self.drag = event.pos()
            self.moveSignal.emit(self.pos_y - old_y)
        
        self.update()

        return super().mouseMoveEvent(event)
    
    def enterEvent(self, event: QEvent) -> None:
        self.color.setAlpha(180)
        self.timer.stop()
        self.timer.start()
        self.update()
        return super().enterEvent(event)
    
    @Slot()
    def timeOutSlot(self):
        if self.color.alpha() != 0:
            self.color.setAlpha(self.color.alpha()-10)
        else:
            self.timer.stop()
        self.update()
        
        return