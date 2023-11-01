import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QToolButton, QPushButton, QLineEdit, QTextEdit
from PyQt5 import uic
import numpy as np
from datetime import datetime
import joblib
from track_utils import *

pipe_lr = joblib.load(open("emotion_classifier_pipe_lr_03_june_2021.pkl", "rb"))


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        uic.loadUi("nlp.ui", self)

        self.Home = self.findChild(QToolButton, "HomeButton")
        self.submitButton = self.findChild(QPushButton, "submitButton")
        self.lineEdit = self.findChild(QLineEdit, "lineEdit")
        self.textEdit = self.findChild(QTextEdit, "textEdit")

        self.submitButton.clicked.connect(self.submit)
        self.clearButton.clicked.connect(self.clear)

        self.show()

    def submit(self):

        # raw_text = "I'm expecting an extremely important phone call any minute now #terror #opportunity"
        raw_text = self.lineEdit.text()
        self.lineEdit.setText("")

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

        emoji_icon = emotions_emoji_dict[prediction]

        self.textEdit.append("Original Text:")
        self.textEdit.append(raw_text)
        self.textEdit.append("\nPrediction:")
        self.textEdit.append("{}:{}".format(prediction, emoji_icon))
        self.textEdit.append("\nConfidence: {}".format(np.max(probability)))
        self.textEdit.append("\nPrediction Probability")

        for i in range(len(probability[0])):
            self.textEdit.append("{}: {}".format(pipe_lr.classes_[i], probability[0][i]))

    def clear(self):
        self.textEdit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
