# This Python file uses the following encoding: utf-8
import sys
from PySide2.QtWidgets import QApplication
import mainWidget

def main():
    app = QApplication([])
    window = mainWidget.MainWidget()
    window.show()
    sys.exit(app.exec_())