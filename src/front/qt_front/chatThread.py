from PySide2.QtCore import QObject, QThread, Signal, Slot
import os, sys, time

class ChatThread(QObject):
    updateRecordSignal = Signal(int,int,str,str)
    def __init__(self):
        super().__init__()
    
    @Slot()
    def mySendMessageSlot(self, msg):
        modified_msg = "I received: " + msg
        time.sleep(1)
        print(modified_msg)
        self.updateRecordSignal.emit(0,0,"",modified_msg)
    
    @Slot()
    def _mySendMessageSlot(self, msg):

        #实现逻辑全放在这里面就行，输入msg，输出response，尽量别写太多代码在里面，业务和界面分离
        #完成后把 函数签名 _mySendMessageSlot 改为 mySendMessageSlot 就行
        response = "????????????"





        self.updateRecordSignal.emit(0,0,"",response)