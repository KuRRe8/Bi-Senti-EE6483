from PySide2.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSizePolicy, QScrollArea, QScrollBar, QFrame
from PySide2.QtGui import QBitmap, QPainter, QPalette, QBrush, QPixmap, QColor, QFont
from PySide2.QtCore import Qt, QPoint, QTimer, Signal, QSize, QRect
from PySide2.QtCore import QObject, Slot


import os
import sys
import enum
import ScrollBar, onemsg

OFFSET_HEIGHT = 105 # 距离父类窗口的底边距
LINE_SPACE = 20     #行间距
KEEP_HISTORY = 800  #保留的聊天记录历史阀值
MOVE_DISTANCE = 10  #聊天记录每次移动的距离
INIT_LINE_HEIGHT = 50 #第一行高
FONT_COUNT_MAX = 40 #每行最大字数
HEIGHT_LINE = 30 #每行高

class RecordType(enum.Enum):
    Image = 0
    Text = 1

class MoveType(enum.Enum):
    BOTTOM = 0
    TOP = 1
    CANCEL = 2

class historyRecord():
    def __init__(self, id: int, photo: int, name: str, message: str) -> None:
        self.type = RecordType.Text
        self.id = id
        self.photo = 0
        self.name = str("")
        self.message = message

class RecordWidgetStruct():
    def __init__(self, id: int, historyCount: int, cw: onemsg.OneMsg, NamePhotoWidget: None, HeadPhotoWidget: None) -> None:
        self.id = id
        self.historyCount = historyCount
        self.cw = cw
        self.NamePhotoWidget = None
        self.HeadPhotoWidget = None

class ChatRecordWidget(QWidget):

    wheelEventSignal = Signal(int)


    def __init__(self, parent: QWidget  = None):
        super().__init__(parent)

        self.id = 99 #记录的ID
        self.historyCount = 0 #记录的数量
        self.record_y = 10 #新消息的虚拟坐标
        self.move_type = False
        self.move_count = 0 #上下移动的次数
        self.move_timer = None
        self.old_size = QSize()
        self.mt = MoveType.CANCEL

        #QMap<int, historyRecord>* recordMap_Message;// 存放聊天数据
        #QMap<int, RecordWidgetStruct>* recordMap_Widget;// 存放聊天小窗体
        self.recordMap_Message = {} 
        self.recordMap_Widget = {}


        self.movetimer = QTimer(self)
        self.movetimer.setInterval(15)
        self.movetimer.timeout.connect(self.moveTimeOutSlot)


        self.scrollBar = ScrollBar.ScrollBar(self.parent().height() - OFFSET_HEIGHT, self)
        self.scrollBar.toBottomSignal.connect(self.toBottomSlot)
        self.scrollBar.toTopSignal.connect(self.toTopSlot)
        self.scrollBar.toCancelSignal.connect(self.toCancelSlot)
        self.scrollBar.moveSignal.connect(self.moveSlot)
        ret = self.wheelEventSignal.connect(self.scrollBar.wheelEventSlot)

        self.resize(self.parent().size())

    def resize(self, size: QSize):
        self.scrollBar.resize(self.size())
        return super().resize(size.width(), size.height() - OFFSET_HEIGHT)
    
    def resizeEvent(self, event: ScrollBar.QResizeEvent) -> None:

        for key, value in self.recordMap_Widget.items():
            value.cw.move(QPoint(event.size().width() - (self.old_size.width() - value.cw.pos().x()), value.cw.pos().y()))
            # value.hpw
            # value.npw
        self.old_size = event.size()
        return super().resizeEvent(event)

    def wheelEvent(self, event: ScrollBar.QWheelEvent) -> None:
        moveCount = event.angleDelta().y() / 120
        self.wheelEventSignal.emit(-moveCount)
        return super().wheelEvent(event)
    
    ##def PixmapMessageShowSlot

    def updateRecordSlot(self, id: int, photo: int, name: str, message: str):
        self.historyCount +=1
        self.newRecord((self.historyCount), id, photo, name, RecordType.Text , message)
    
    @Slot()
    def toBottomSlot(self):
        self.mt = MoveType.BOTTOM
    
    @Slot()
    def toTopSlot(self):
        self.mt = MoveType.TOP
    
    @Slot()
    def toCancelSlot(self):
        self.mt = MoveType.CANCEL
    
    @Slot(int)
    def moveSlot(self, move_count: int):
        if len(self.recordMap_Widget) == 0:
            return
        self.move_count += move_count
        self.movetimer.start()
        self.move_type = False

    def virtualHeight(self):
        return self.height() - INIT_LINE_HEIGHT
    
    def showNowMessage(self, record_y: int):
        if record_y <= self.virtualHeight():
            return
        
        if self.scrollBar.isBottom():
            self.move_count += 1
        else:
            self.scrollBar.setLength()

        self.move_type = True
        self.movetimer.start()

    def newRecord(self, historyCount: int, id: int, photo: int, name: str, type: RecordType, message: str):

        assert type == RecordType.Text
        if len(self.recordMap_Widget)>0:
            lastkey = list(self.recordMap_Widget.keys())[-1]
            self.record_y = self.recordMap_Widget[lastkey].cw.y() + self.recordMap_Widget[lastkey].cw.height() + LINE_SPACE
        
        is_text = True 

        if self.id == id:
            cw = onemsg.OneMsg(QPoint(0, self.record_y), message, is_text, True, False, self)
            #npw
            #hpw
        else:
            cw = onemsg.OneMsg(QPoint(0, self.record_y), message, is_text, False, False, self)
            #npw
            #hpw

        #connect hpw

        tmp = cw.isVisible()
        tmp = cw.setVisible(True)
        

        #npw
        #hpw

        self.showNowMessage(self.record_y + cw.height())

        self.recordMap_Message.update({historyCount: historyRecord(id=id, photo=photo, name=name, message=message)})
        self.recordMap_Widget.update({historyCount: RecordWidgetStruct(id=id, historyCount=historyCount, cw=cw, NamePhotoWidget=None, HeadPhotoWidget=None)})
        


    def addHistoryRecord_Widget(self, historyCount: int, historyRcd: historyRecord, pos_y: int):
        is_text = True 

        if self.id == historyRcd.id:
            cw = onemsg.OneMsg(QPoint(0, self.record_y), historyRcd.message, is_text, True, True, self)
            #npw
            #hpw
        else:
            cw = onemsg.OneMsg(QPoint(0, self.record_y), historyRcd.message, is_text, False, True, self)
            #npw
            #hpw

        #connect hpw

        cw.setVisible(True)
        #npw
        #hpw

        self.showNowMessage(self.record_y + cw.height())

        self.recordMap_Widget.update({historyCount: RecordWidgetStruct(id=historyRcd.id, historyCount=historyCount, oneMsg=cw, NamePhotoWidget=None, HeadPhotoWidget=None)})
        
    def delRecordMap_Widget(self, key: int):
        self.recordMap_Widget.pop(key)

    def isAdjustFinish(self):
        ret = False
        if self.mt == MoveType.BOTTOM:
            lastkey = list(self.recordMap_Widget.keys())[-1]
            if self.recordMap_Widget[lastkey].historyCount == self.historyCount and self.recordMap_Widget[lastkey].cw.y() + self.recordMap_Widget[lastkey].cw.height() <= self.virtualHeight():
                ret = True
                self.movetimer.stop()
            else:
                self.move_count += 1
        elif self.mt == MoveType.TOP:
            firstkey = list(self.recordMap_Widget.keys())[0]
            if self.recordMap_Widget[firstkey].historyCount == 1 and self.recordMap_Widget[firstkey].cw.y() >= 0:
                ret = True
                self.movetimer.stop()
            else:
                self.move_count -= 1
        elif self.mt == MoveType.CANCEL:
            ret = True
            self.movetimer.stop()
        
        return ret
    
    def isMoveOne_Up(self, rws: RecordWidgetStruct) ->bool:
        return rws.cw.y() + rws.cw.height() >= self.virtualHeight() and rws.cw.y() + rws.cw.height() - MOVE_DISTANCE < self.virtualHeight()
    
    def ismoveOne_Down(self, rws: RecordWidgetStruct) -> bool:
        return rws.cw.y() + rws.cw.height() <= self.virtualHeight() and rws.cw.y() + rws.cw.height() - MOVE_DISTANCE > self.virtualHeight()

    def setRecordMoveSleep(self):
        sleep = abs(self.move_count)
        direction = 1 if self.move_count >= 0 else -1
        
        if(sleep > 5):
            MOVE_DISTANCE = 50 * direction
        elif (sleep > 3):
            MOVE_DISTANCE = 30 * direction
        elif (sleep > 2):
            MOVE_DISTANCE = 20 * direction
        else:
            MOVE_DISTANCE = 10 * direction

    def record_ADD_DEL(self):
        firstkey = list(self.recordMap_Widget.keys())[0]
        lastkey = list(self.recordMap_Widget.keys())[-1]

        rws_first_temp = self.recordMap_Widget[firstkey]
        rws_last_temp = self.recordMap_Widget[lastkey]

        if self.move_count > 0:
            if rws_first_temp.cw.y() < -KEEP_HISTORY:
                self.delRecordMap_Widget(rws_first_temp.historyCount)
            if rws_last_temp.historyCount < self.historyCount:
                hr_temp = self.recordMap_Message[rws_last_temp.historyCount + 1]
                if rws_last_temp.cw.y() + rws_last_temp.cw.height() + LINE_SPACE + self.getMessageHeight(hr_temp) < self.height()+KEEP_HISTORY:
                    self.addHistoryRecord_Widget(rws_last_temp.historyCount + 1, hr_temp, rws_last_temp.cw.y() + rws_last_temp.cw.height() + LINE_SPACE)
        
        elif self.move_count < 0:
            if rws_last_temp.cw.y() + rws_last_temp.cw.height() > KEEP_HISTORY + self.height():
                self.delRecordMap_Widget(rws_last_temp.historyCount)

            if rws_first_temp.historyCount != 1:
                hr_temp = self.recordMap_Message[rws_first_temp.historyCount - 1]
                if rws_first_temp.cw.y() - LINE_SPACE - self.getMessageHeight(hr_temp) < -KEEP_HISTORY:
                    self.addHistoryRecord_Widget(rws_first_temp.historyCount - 1, hr_temp, rws_first_temp.cw.y() - LINE_SPACE - self.getMessageHeight(hr_temp))

    def getMessageHeight(self, record: historyRecord):
        height = 0
        if record.type == RecordType.Text:
            line = len(record.message) // FONT_COUNT_MAX
            if len(record.message) % FONT_COUNT_MAX:
                line += 1
            height = INIT_LINE_HEIGHT + (HEIGHT_LINE * (line - 1))
        else:
            NotImplementedError()
        
        return height
    
    def moveCross(self):
        firstkey = list(self.recordMap_Widget.keys())[0]
        lastkey = list(self.recordMap_Widget.keys())[-1]
        if self.move_count>0:
            if self.recordMap_Widget[lastkey].cw.y() + self.recordMap_Widget[lastkey].cw.height() <= self.virtualHeight():
                self.move_count = 0
                self.movetimer.stop()
                return True
        elif self.move_count<0:
            if self.recordMap_Widget[firstkey].cw.y() >= 0:
                self.move_count = 0
                self.movetimer.stop()
                return True
        
        return False

    @Slot()
    def moveTimeOutSlot(self):
        self.record_ADD_DEL( )
        if self.move_count==0 and self.isAdjustFinish():
            
            return
        
        self.setRecordMoveSleep()

        if self.moveCross():
            return
        
        for key, value in self.recordMap_Widget.items():
            self.controlCount(value)
            value.cw.move(value.cw.x(), value.cw.y() - MOVE_DISTANCE)
            # value.hpw
            # value.npw

    def controlCount(self, rws: RecordWidgetStruct):
        if self.move_count>0:
            if self.isMoveOne_Up(rws):
                self.move_count -=1
                if self.move_type:
                    self.move_type = False
                    self.scrollBar.setLength()
        else:
            if self.ismoveOne_Down(rws):
                self.move_count +=1

    def enterEvent(self, event: ScrollBar.QEvent) -> None:
        print("enterEvent of ChatRecordWidget")
        return super().enterEvent(event)
    
    def leaveEvent(self, event: ScrollBar.QEvent) -> None:
        print("leaveEvent of ChatRecordWidget")
        return super().leaveEvent(event)