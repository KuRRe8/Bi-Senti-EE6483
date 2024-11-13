# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainWidget.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import rc_rc

class Ui_mainWidget(object):
    def setupUi(self, mainWidget):
        if not mainWidget.objectName():
            mainWidget.setObjectName(u"mainWidget")
        mainWidget.resize(1090, 730)
        mainWidget.setStyleSheet(u"")
        self.gridLayout = QGridLayout(mainWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalSpacerHistoryLeft = QSpacerItem(132, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacerHistoryLeft, 0, 0, 1, 1)

        self.widgetHistory = QWidget(mainWidget)
        self.widgetHistory.setObjectName(u"widgetHistory")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetHistory.sizePolicy().hasHeightForWidth())
        self.widgetHistory.setSizePolicy(sizePolicy)

        self.gridLayout.addWidget(self.widgetHistory, 0, 1, 1, 1)

        self.horizontalSpacerHistoryRight = QSpacerItem(131, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacerHistoryRight, 0, 2, 1, 1)

        self.horizontalSpacerInputLeft = QSpacerItem(132, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacerInputLeft, 1, 0, 1, 1)

        self.widgetInputContainer = QWidget(mainWidget)
        self.widgetInputContainer.setObjectName(u"widgetInputContainer")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.widgetInputContainer.sizePolicy().hasHeightForWidth())
        self.widgetInputContainer.setSizePolicy(sizePolicy1)
        self.widgetInputContainer.setMinimumSize(QSize(800, 50))
        self.horizontalLayout = QHBoxLayout(self.widgetInputContainer)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.inputlineEdit = QLineEdit(self.widgetInputContainer)
        self.inputlineEdit.setObjectName(u"inputlineEdit")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.inputlineEdit.sizePolicy().hasHeightForWidth())
        self.inputlineEdit.setSizePolicy(sizePolicy2)
        font = QFont()
        font.setFamily(u"Times New Roman")
        font.setPointSize(18)
        self.inputlineEdit.setFont(font)
        self.inputlineEdit.setDragEnabled(True)

        self.horizontalLayout.addWidget(self.inputlineEdit)

        self.sendButton = QPushButton(self.widgetInputContainer)
        self.sendButton.setObjectName(u"sendButton")
        sizePolicy1.setHeightForWidth(self.sendButton.sizePolicy().hasHeightForWidth())
        self.sendButton.setSizePolicy(sizePolicy1)
        self.sendButton.setMinimumSize(QSize(40, 40))
        self.sendButton.setMaximumSize(QSize(40, 40))
        self.sendButton.setContextMenuPolicy(Qt.NoContextMenu)
        icon = QIcon()
        icon.addFile(u"icon/icon_button/sendmsg2.png", QSize(), QIcon.Normal, QIcon.Off)
        self.sendButton.setIcon(icon)
        self.sendButton.setIconSize(QSize(50, 50))
        self.sendButton.setAutoDefault(False)
        self.sendButton.setFlat(False)

        self.horizontalLayout.addWidget(self.sendButton)


        self.gridLayout.addWidget(self.widgetInputContainer, 1, 1, 1, 1)

        self.horizontalSpacerInputRight = QSpacerItem(131, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacerInputRight, 1, 2, 1, 1)


        self.retranslateUi(mainWidget)

        self.sendButton.setDefault(False)


        QMetaObject.connectSlotsByName(mainWidget)
    # setupUi

    def retranslateUi(self, mainWidget):
        mainWidget.setWindowTitle(QCoreApplication.translate("mainWidget", u"Group6483", None))
        self.inputlineEdit.setInputMask("")
        self.inputlineEdit.setPlaceholderText(QCoreApplication.translate("mainWidget", u"Chat here!", None))
#if QT_CONFIG(tooltip)
        self.sendButton.setToolTip(QCoreApplication.translate("mainWidget", u"Send Msg", None))
#endif // QT_CONFIG(tooltip)
        self.sendButton.setText("")
    # retranslateUi

