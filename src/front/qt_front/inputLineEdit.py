from PySide2.QtWidgets import QLineEdit, QWidget
from PySide2.QtGui import QFont
from PySide2.QtCore import Qt
from PySide2.QtGui import QPalette, QBrush

class inputLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        qPalette = QPalette()
        qPalette.setBrush(QPalette.Text, QBrush(Qt.white))
        

        self.setFont(QFont("Times New Roman", 16))
        self.setPalette(qPalette)
        self.setFocus()
        self.setStyleSheet("background:transparent; border-width:0; border-style:outset")
        self.setPlaceholderText("Chat here!")
    
        #self.layout().addWidget(self.lineedit)

    #def paintEvent(self, event):
        # Custom painting code here
        #pass

    """
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Handle the enter key press event here
            # Your code goes here
            self.clear()
            pass
        else:
            super().keyPressEvent(event)
    """

    