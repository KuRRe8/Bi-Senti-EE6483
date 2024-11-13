import os
from PySide2.QtWidgets import QWidget as QWidget
from PySide2.QtWidgets import QPushButton as QPushButton
from PySide2.QtGui import QBitmap as QBitmap, QResizeEvent
from PySide2.QtGui import QPainter as QPainter
from PySide2.QtGui import QPalette as QPalette
from PySide2.QtGui import QBrush as QBrush
from PySide2.QtGui import QPixmap as QPixmap
from PySide2.QtCore import QEvent, Qt as Qt
from PySide2.QtCore import QResource
from PySide2.QtCore import QRect as QRect
from PySide2.QtCore import QSize as QSize
from PySide2.QtCore import QPoint as QPoint
from PySide2.QtCore import QTimer as QTimer
from PySide2.QtGui import QColor as QColor
from PySide2.QtCore import Slot as Slot
from PySide2.QtGui import QFont as QFont
from PySide2.QtCore import Signal as Signal
from PySide2.QtGui import QFontMetrics as QFontMetrics
from PySide2.QtGui import QFont as QFont


_ME_OFFICES = 110# 自己的消息位置偏移量
_BEGIN_X = 20# 聊天记录开始位置
_FONT_SPACE = 15# 字体间距
_MAX_LENGTH = 640# 显示的最大长度
_HEIGHT_LINE = 30# 每行的高度
_FONT_COUNT_MAX = 40# 每行显示的最大字数
_INIT_LINE_HEIGHT = 50# 第一行的高度
MAX_IMAGE_SIZE = QSize(640,350)# 显示图片的最大大小
NULL_IMAGE_SIZE = QSize(100,100)# 没有图片时显示的图片大小

class OneMsg(QWidget):
    finishPaintSignal = Signal()
    def __init__(self, pos: QPoint, message: str, is_text: bool, is_me: bool, is_history:bool, parent: QWidget  = None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.message = message
        self.is_text = is_text
        self.is_me = is_me
        self.fps = 0


        if is_text: 
            line = len(message) // _FONT_COUNT_MAX
            if len(message) % _FONT_COUNT_MAX != 0:
                line += 1
            if len(message) >= _FONT_COUNT_MAX:
                self.resize(QSize(_MAX_LENGTH ,_INIT_LINE_HEIGHT + (_HEIGHT_LINE * (line - 1))))
            else:
                self.resize(QSize(len(message) * _FONT_SPACE + _BEGIN_X*2, _INIT_LINE_HEIGHT))
        else:
            NotImplementedError("Not implemented yet")

        if is_me:
            point = QPoint(parent.width() - self.width() - _ME_OFFICES, pos.y())
        else:
            point = QPoint(100, pos.y())
    
        if is_history:
            self.move(point)
        else:
            if is_me:
                self.animationTimer_me = QTimer(self)
                self.animationTimer_me.setInterval(20)
                self.animationTimer_me.timeout.connect(self.animationMeTimeOutSlot)
                self.move(point.x()+30, point.y())
                self.animationTimer_me.start()
            else:
                self.animationTimer_other = QTimer(self)
                self.animationTimer_other.setInterval(20)
                self.animationTimer_other.timeout.connect(self.animationOtherTimeOutSlot)
                self.move(point.x()-30, point.y())
                self.animationTimer_other.start()

        if is_me:
            self.coloroff = QColor("#6495ED")
        else:
            self.coloroff = QColor("#D4D4D4")
        self.coloroff.setAlpha(150)

        if is_me:
            self.coloron = QColor("#6495ED")
        else:
            self.coloron = QColor("#A9A9A9")
        self.coloron.setAlpha(180)


        pal = QPalette(self.palette())
        pal.setBrush(QPalette.Background, QBrush(self.coloroff))
        self.setPalette(pal)

        self.finishPaintSignal.connect(self.finishPaintSlot)

    @Slot()
    def finishPaintSlot(self):
        return
        self.setVisible(True)



    def resizeEvent(self, event: QResizeEvent) -> None:
        #return super().resizeEvent(event)
        bmp = QBitmap(self.size())
        bmp.fill()
        p = QPainter(bmp)
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black)
        p.drawRoundedRect(bmp.rect(), 20, 20)
        p.end()
        self.setMask(bmp)
        return super().resizeEvent(event)
    
    def paintEvent(self, event):
        #return super().paintEvent(event)
        pai = QPainter(self)
        pai.begin(self)
        if self.is_text:
            if self.is_me:
                pai.setPen(QColor("#FFE4C4"))
            else:
                pai.setPen(QColor("#282828"))
            
            pai.setFont(QFont("Microsoft YaHei", 12))
            px = _BEGIN_X
            py = _HEIGHT_LINE

            fon = QFont("Microsoft YaHei", 12)
            fm = QFontMetrics(fon)
            a_width = fm.width("a")
            width_all = {}
            for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ,.?!@#$%^&*()_+-=[]{}|;:<>?/\\':
                chat_width = fm.width(char)
                width_all[char] = chat_width
            for char in self.message:
                if px >= _MAX_LENGTH - _BEGIN_X:
                    px = _BEGIN_X
                    py += _HEIGHT_LINE
                pai.drawText(px, py, char)
                if char in width_all:
                    px += width_all[char]
                else:
                    px += _FONT_SPACE-5
        else:
            NotImplementedError("Not implemented yet")
        pai.end()
        super().paintEvent(event)
        return self.finishPaintSignal.emit()

    def enterEvent(self, event: QEvent) -> None:
        pal = QPalette(self.palette())
        pal.setBrush(QPalette.Background, QBrush(self.coloron))
        self.setPalette(pal)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        pal = QPalette(self.palette())
        pal.setBrush(QPalette.Background, QBrush(self.coloroff))
        self.setPalette(pal)
        return super().leaveEvent(event)
    
    @Slot()
    def animationMeTimeOutSlot(self):
        self.move(self.x()-5, self.y())
        self.fps += 1
        if self.fps >= 5:
            self.animationTimer_me.stop()

    @Slot()
    def animationOtherTimeOutSlot(self):
        self.move(self.x()+5, self.y())
        self.fps += 1
        if self.fps >= 5:
            self.animationTimer_other.stop()