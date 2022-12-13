# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibration.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(700, 700)
        self.ViewProjectorPattern = QtWidgets.QLabel(Dialog)
        self.ViewProjectorPattern.setGeometry(QtCore.QRect(360, 70, 320, 240))
        self.ViewProjectorPattern.setFrameShape(QtWidgets.QFrame.Box)
        self.ViewProjectorPattern.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ViewProjectorPattern.setAlignment(QtCore.Qt.AlignCenter)
        self.ViewProjectorPattern.setObjectName("ViewProjectorPattern")
        self.LabelProjectorPatternRect = QtWidgets.QLabel(Dialog)
        self.LabelProjectorPatternRect.setEnabled(True)
        self.LabelProjectorPatternRect.setGeometry(QtCore.QRect(360, 330, 320, 20))
        self.LabelProjectorPatternRect.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelProjectorPatternRect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LabelProjectorPatternRect.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelProjectorPatternRect.setObjectName("LabelProjectorPatternRect")
        self.LabelCameraImage = QtWidgets.QLabel(Dialog)
        self.LabelCameraImage.setEnabled(True)
        self.LabelCameraImage.setGeometry(QtCore.QRect(20, 50, 320, 20))
        self.LabelCameraImage.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelCameraImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LabelCameraImage.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelCameraImage.setObjectName("LabelCameraImage")
        self.ViewProjectorPatternRect = QtWidgets.QLabel(Dialog)
        self.ViewProjectorPatternRect.setGeometry(QtCore.QRect(360, 350, 320, 240))
        self.ViewProjectorPatternRect.setFrameShape(QtWidgets.QFrame.Box)
        self.ViewProjectorPatternRect.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ViewProjectorPatternRect.setAlignment(QtCore.Qt.AlignCenter)
        self.ViewProjectorPatternRect.setObjectName("ViewProjectorPatternRect")
        self.LabelProjectorPattern = QtWidgets.QLabel(Dialog)
        self.LabelProjectorPattern.setEnabled(True)
        self.LabelProjectorPattern.setGeometry(QtCore.QRect(360, 50, 320, 20))
        self.LabelProjectorPattern.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelProjectorPattern.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LabelProjectorPattern.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelProjectorPattern.setObjectName("LabelProjectorPattern")
        self.ViewCameraImageRect = QtWidgets.QLabel(Dialog)
        self.ViewCameraImageRect.setGeometry(QtCore.QRect(20, 350, 320, 240))
        self.ViewCameraImageRect.setFrameShape(QtWidgets.QFrame.Box)
        self.ViewCameraImageRect.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ViewCameraImageRect.setAlignment(QtCore.Qt.AlignCenter)
        self.ViewCameraImageRect.setObjectName("ViewCameraImageRect")
        self.LabelCameraImageRect = QtWidgets.QLabel(Dialog)
        self.LabelCameraImageRect.setEnabled(True)
        self.LabelCameraImageRect.setGeometry(QtCore.QRect(20, 330, 320, 20))
        self.LabelCameraImageRect.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelCameraImageRect.setFrameShadow(QtWidgets.QFrame.Raised)
        self.LabelCameraImageRect.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelCameraImageRect.setObjectName("LabelCameraImageRect")
        self.ViewCameraImage = QtWidgets.QLabel(Dialog)
        self.ViewCameraImage.setGeometry(QtCore.QRect(20, 70, 320, 240))
        self.ViewCameraImage.setFrameShape(QtWidgets.QFrame.Box)
        self.ViewCameraImage.setFrameShadow(QtWidgets.QFrame.Plain)
        self.ViewCameraImage.setAlignment(QtCore.Qt.AlignCenter)
        self.ViewCameraImage.setObjectName("ViewCameraImage")
        self.LabelStatus = QtWidgets.QLabel(Dialog)
        self.LabelStatus.setGeometry(QtCore.QRect(20, 20, 661, 20))
        self.LabelStatus.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelStatus.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LabelStatus.setScaledContents(False)
        self.LabelStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelStatus.setObjectName("LabelStatus")
        self.ToolButtonMainFolder = QtWidgets.QToolButton(Dialog)
        self.ToolButtonMainFolder.setGeometry(QtCore.QRect(520, 610, 40, 20))
        self.ToolButtonMainFolder.setObjectName("ToolButtonMainFolder")
        self.LabelSceneInfo = QtWidgets.QLabel(Dialog)
        self.LabelSceneInfo.setGeometry(QtCore.QRect(20, 670, 151, 20))
        self.LabelSceneInfo.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelSceneInfo.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LabelSceneInfo.setScaledContents(False)
        self.LabelSceneInfo.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelSceneInfo.setObjectName("LabelSceneInfo")
        self.LabelMainFolderVis = QtWidgets.QLabel(Dialog)
        self.LabelMainFolderVis.setGeometry(QtCore.QRect(140, 610, 371, 20))
        self.LabelMainFolderVis.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelMainFolderVis.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LabelMainFolderVis.setScaledContents(False)
        self.LabelMainFolderVis.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelMainFolderVis.setObjectName("LabelMainFolderVis")
        self.LabelPatternFolder = QtWidgets.QLabel(Dialog)
        self.LabelPatternFolder.setGeometry(QtCore.QRect(20, 640, 111, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LabelPatternFolder.setFont(font)
        self.LabelPatternFolder.setObjectName("LabelPatternFolder")
        self.LabelMainFolder = QtWidgets.QLabel(Dialog)
        self.LabelMainFolder.setGeometry(QtCore.QRect(20, 610, 111, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LabelMainFolder.setFont(font)
        self.LabelMainFolder.setObjectName("LabelMainFolder")
        self.LabelPatternFolderVis = QtWidgets.QLabel(Dialog)
        self.LabelPatternFolderVis.setGeometry(QtCore.QRect(140, 640, 421, 20))
        self.LabelPatternFolderVis.setFrameShape(QtWidgets.QFrame.Panel)
        self.LabelPatternFolderVis.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.LabelPatternFolderVis.setScaledContents(False)
        self.LabelPatternFolderVis.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelPatternFolderVis.setObjectName("LabelPatternFolderVis")
        self.PushButtonPrevious = QtWidgets.QPushButton(Dialog)
        self.PushButtonPrevious.setGeometry(QtCore.QRect(570, 610, 21, 21))
        self.PushButtonPrevious.setObjectName("PushButtonPrevious")
        self.PushButtonNext = QtWidgets.QPushButton(Dialog)
        self.PushButtonNext.setGeometry(QtCore.QRect(660, 610, 21, 21))
        self.PushButtonNext.setObjectName("PushButtonNext")
        self.PushButtonSave = QtWidgets.QPushButton(Dialog)
        self.PushButtonSave.setGeometry(QtCore.QRect(630, 670, 51, 21))
        self.PushButtonSave.setObjectName("PushButtonSave")
        self.PushButtonDetectChess = QtWidgets.QPushButton(Dialog)
        self.PushButtonDetectChess.setGeometry(QtCore.QRect(570, 640, 51, 21))
        self.PushButtonDetectChess.setObjectName("PushButtonDetectChess")
        self.PushButtonRun = QtWidgets.QPushButton(Dialog)
        self.PushButtonRun.setGeometry(QtCore.QRect(570, 670, 51, 21))
        self.PushButtonRun.setObjectName("PushButtonRun")
        self.RadioButtonValid = QtWidgets.QRadioButton(Dialog)
        self.RadioButtonValid.setGeometry(QtCore.QRect(600, 610, 51, 20))
        self.RadioButtonValid.setObjectName("RadioButtonValid")
        self.LabelAlpha = QtWidgets.QLabel(Dialog)
        self.LabelAlpha.setGeometry(QtCore.QRect(180, 670, 51, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LabelAlpha.setFont(font)
        self.LabelAlpha.setObjectName("LabelAlpha")
        self.HorizontalSliderAlpha = QtWidgets.QSlider(Dialog)
        self.HorizontalSliderAlpha.setGeometry(QtCore.QRect(230, 670, 281, 22))
        self.HorizontalSliderAlpha.setMaximum(100)
        self.HorizontalSliderAlpha.setOrientation(QtCore.Qt.Horizontal)
        self.HorizontalSliderAlpha.setObjectName("HorizontalSliderAlpha")
        self.LabelAlphaValue = QtWidgets.QLabel(Dialog)
        self.LabelAlphaValue.setGeometry(QtCore.QRect(530, 670, 31, 20))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.LabelAlphaValue.setFont(font)
        self.LabelAlphaValue.setObjectName("LabelAlphaValue")
        self.RadioButtonDetectChessLoad = QtWidgets.QRadioButton(Dialog)
        self.RadioButtonDetectChessLoad.setGeometry(QtCore.QRect(630, 640, 51, 20))
        self.RadioButtonDetectChessLoad.setObjectName("RadioButtonDetectChessLoad")

        self.retranslateUi(Dialog)
        self.HorizontalSliderAlpha.valueChanged['int'].connect(self.LabelAlphaValue.setNum)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Calibration"))
        self.ViewProjectorPattern.setText(_translate("Dialog", "NULL"))
        self.LabelProjectorPatternRect.setText(_translate("Dialog", "Rectified Projector"))
        self.LabelCameraImage.setText(_translate("Dialog", "ChessBoard Camera"))
        self.ViewProjectorPatternRect.setText(_translate("Dialog", "NULL"))
        self.LabelProjectorPattern.setText(_translate("Dialog", "ChessBoard Pattern"))
        self.ViewCameraImageRect.setText(_translate("Dialog", "NULL"))
        self.LabelCameraImageRect.setText(_translate("Dialog", "Rectified Camera"))
        self.ViewCameraImage.setText(_translate("Dialog", "NULL"))
        self.LabelStatus.setText(_translate("Dialog", "Current Scene"))
        self.ToolButtonMainFolder.setText(_translate("Dialog", "..."))
        self.LabelSceneInfo.setText(_translate("Dialog", "Scene detected: 0/0"))
        self.LabelMainFolderVis.setText(_translate("Dialog", "${main_folder}"))
        self.LabelPatternFolder.setText(_translate("Dialog", "pattern_folder:"))
        self.LabelMainFolder.setText(_translate("Dialog", "main_folder:"))
        self.LabelPatternFolderVis.setText(_translate("Dialog", "${main_folder}/pat"))
        self.PushButtonPrevious.setText(_translate("Dialog", "<"))
        self.PushButtonNext.setText(_translate("Dialog", ">"))
        self.PushButtonSave.setText(_translate("Dialog", "Save"))
        self.PushButtonDetectChess.setText(_translate("Dialog", "Detect"))
        self.PushButtonRun.setText(_translate("Dialog", "Run"))
        self.RadioButtonValid.setText(_translate("Dialog", "Valid"))
        self.LabelAlpha.setText(_translate("Dialog", "alpha:"))
        self.LabelAlphaValue.setText(_translate("Dialog", "0.0"))
        self.RadioButtonDetectChessLoad.setText(_translate("Dialog", "Load"))