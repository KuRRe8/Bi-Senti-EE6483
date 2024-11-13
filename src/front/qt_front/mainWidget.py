import os
from PySide2.QtWidgets import QWidget as QWidget
from PySide2.QtWidgets import QPushButton as QPushButton
from PySide2.QtGui import QBitmap as QBitmap
from PySide2.QtGui import QPainter as QPainter
from PySide2.QtGui import QPalette as QPalette
from PySide2.QtGui import QBrush as QBrush
from PySide2.QtGui import QPixmap as QPixmap
from PySide2.QtCore import Qt as Qt
from PySide2.QtCore import QResource, Slot
from PySide2.QtCore import QRect as QRect
from PySide2.QtCore import Signal as Signal
from PySide2.QtWidgets import QLineEdit as QLineEdit
from PySide2.QtWidgets import QGridLayout as QGridLayout
from PySide2.QtWidgets import QVBoxLayout as QVBoxLayout
from PySide2.QtWidgets import QHBoxLayout as QHBoxLayout
from PySide2.QtWidgets import QFrame as QFrame
from PySide2.QtWidgets import QLabel as QLabel
from PySide2.QtWidgets import QScrollArea as QScrollArea
from PySide2.QtWidgets import QScrollBar as QScrollBar
from PySide2.QtWidgets import QSizePolicy as QSizePolicy
from PySide2.QtWidgets import QSpacerItem as QSpacerItem
from PySide2.QtCore import QThread as QThread

import chatThread
import ui_mainWidget
import inputLineEdit
import sendButton
import historyWidget

class MainWidget(QWidget):
    startAnimationSignal = Signal()
    SendMessageSignal = Signal(str)# will connnect to chatThread.mySendMessageSlot
    updateMyselfRecordSignal = Signal(int,int,str,str)
    def __init__(self):


        QWidget.__init__(self)
        #ret = QResource.registerResource(os.getcwd()+'\\rc.rcc')
        self.ui = ui_mainWidget.Ui_mainWidget()
        self.ui.setupUi(self)
        #self.ui.sendButton.clicked.connect(self._slotSendMsg)
        #self.ui.inputlineEdit.returnPressed.connect(self._slotSendMsg)
        self.ui.gridLayout.removeWidget(self.ui.widgetHistory)
        self.chatrecordwidget = historyWidget.ChatRecordWidget(self)
        self.chatrecordwidget.setObjectName(u"chatrecordwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chatrecordwidget.sizePolicy().hasHeightForWidth())
        self.chatrecordwidget.setSizePolicy(sizePolicy)

        self.ui.gridLayout.addWidget(self.chatrecordwidget, 0, 1, 1, 1)

        self.ui.widgetHistory.deleteLater()
        self.ui.sendButton.deleteLater()
        self.ui.inputlineEdit.deleteLater()
        #self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.bmp = QBitmap()
        self.qpai = QPainter()
        self.qpal = QPalette()

        self.chatT = chatThread.ChatThread()
        self.SendMessageSignal.connect(self.chatT.mySendMessageSlot)
        self.chatT.updateRecordSignal.connect(self.chatrecordwidget.updateRecordSlot)
        self.WorkerThread = QThread()
        self.chatT.moveToThread(self.WorkerThread)
        self.WorkerThread.start()

        self.profile = QPushButton(self)
        self.profile.setFixedSize(10, 10)
        self.profile.setStyleSheet("""
            QPushButton {
                border-radius: 50px;
                background-color: red;
                width: 10px;
                height: 10px;
            }
        """)
        self.inputline = inputLineEdit.inputLineEdit(self)
        self.sendbutton = sendButton.SendButton(self)

        self.ui.widgetInputContainer.layout().addWidget(self.profile)
        self.ui.widgetInputContainer.layout().addWidget(self.inputline)
        self.ui.widgetInputContainer.layout().addWidget(self.sendbutton)

        self.inputline.returnPressed.connect(self.returnPressed_Slot)
        self.sendbutton.sendMessageSignal.connect(self.returnPressed_Slot)
        self.startAnimationSignal.connect(self.sendbutton.startAnimationSlot)
        self.updateMyselfRecordSignal.connect(self.chatrecordwidget.updateRecordSlot)

    def __del__(self):
        self.WorkerThread.quit()
        self.WorkerThread.wait()
        self.deleteLater()

    """
    def resizeEvent(self, event: ui_mainWidget.QResizeEvent) -> None:
        bmp = QBitmap(self.size())
        bmp.fill()
        p = QPainter(bmp)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black)
        p.drawRoundedRect(bmp.rect(), 15, 15)
        self.setMask(bmp)
        
        qp = QPalette(self.palette())
        qp.setBrush(QPalette.Background, QBrush(QPixmap(":/img/image/preview.jpg").scaled(self.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation)))

        return super().resizeEvent(event)
    """
    
    """
    def paintEvent(self, event: ui_mainWidget.QPaintEvent) -> None:
        self.bmp = QBitmap(self.size())
        self.bmp.fill()
        if self.qpai.isActive():
          self.qpai.end()  # End the current drawing if the painter is active
        self.qpai.begin(self.bmp)
        self.qpai.setRenderHint(QPainter.Antialiasing, True)
        self.qpai.setPen(Qt.NoPen)
        self.qpai.setBrush(Qt.black)
        self.qpai.drawRoundedRect(self.bmp.rect(), 15, 15)
        self.setMask(self.bmp)
        
        self.qpal = QPalette(self.palette())
        self.qpal.setBrush(QPalette.Background, QBrush(QPixmap(":/img/image/preview.jpg").scaled(self.size(), Qt.IgnoreAspectRatio, Qt.FastTransformation)))
        return super().paintEvent(event)
    """
    
    def paintEvent(self, event: ui_mainWidget.QPaintEvent) -> None:
        qpai = QPainter(self)
        qpai.drawPixmap(self.rect(), QPixmap(":/img/image/preview.jpg"), QRect())
        return super().paintEvent(event)
    
    #update history box
    def _slotSendMsg(self):
        #ui = ui_mainWidget.Ui_mainWidget()
        #ui.historyBrowser.append(ui.inputlineEdit.text())
        Qmessage = self.ui.inputlineEdit.text()
        self.ui.inputlineEdit.clear()

    @Slot()
    def returnPressed_Slot(self):
        if self.inputline.text() == "":
            return
        self.startAnimationSignal.emit()
        msg = self.inputline.text()
        if len(msg) > 160:
            msg = msg[:160]
        
        self.SendMessageSignal.emit(msg)
        self.updateMyselfRecordSignal.emit(99,0,"",msg)
        self.inputline.clear()
