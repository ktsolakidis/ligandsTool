import sys
import pandas as pd
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow,QInputDialog, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QWidget, QComboBox, QLabel, QLineEdit, QHBoxLayout

from DBSCANenh import perform_dbscan_clustering
from MAHALANOBISenh import mahalanobis_plot

class ClusteringGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.outliers_df = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Clustering Analysis Tool')
        self.setGeometry(100, 100, 500, 500)

        layout = QVBoxLayout()

        # Button to open CSV
        self.btnOpenCSV = QPushButton('Open CSV', self)
        self.btnOpenCSV.clicked.connect(self.openFileDialog)
        self.btnOpenCSV.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.btnOpenCSV)
        layout.setAlignment(self.btnOpenCSV, Qt.AlignCenter)

        # List Widget for showing columns
        self.listWidget = QListWidget(self)
        self.listWidget.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.listWidget)

        # ComboBox for choosing clustering method
        self.methodComboBox = QComboBox(self)
        self.methodComboBox.addItems(['DBSCAN', 'Mahalanobis'])
        self.methodComboBox.currentTextChanged.connect(self.onMethodChanged)
        layout.addWidget(self.methodComboBox)
        layout.setAlignment(self.methodComboBox, Qt.AlignCenter)

        # DBSCAN parameters
        self.dbscanLayout = QHBoxLayout()
        self.epsLabel = QLabel('Eps:')
        self.epsInput = QLineEdit(self)
        self.minSamplesLabel = QLabel('Min Samples:')
        self.minSamplesInput = QLineEdit(self)
        self.dbscanLayout.addWidget(self.epsLabel)
        self.dbscanLayout.addWidget(self.epsInput)
        self.dbscanLayout.addWidget(self.minSamplesLabel)
        self.dbscanLayout.addWidget(self.minSamplesInput)
        layout.addLayout(self.dbscanLayout)

        # Mahalanobis parameter
        self.mahalanobisLayout = QHBoxLayout()
        self.thresholdLabel = QLabel('Threshold:')
        self.thresholdInput = QLineEdit(self)
        self.mahalanobisLayout.addWidget(self.thresholdLabel)
        self.mahalanobisLayout.addWidget(self.thresholdInput)
        layout.addLayout(self.mahalanobisLayout)

        # Button to run clustering
        self.btnRunClustering = QPushButton('Run Clustering', self)
        self.btnRunClustering.clicked.connect(self.runClustering)
        self.btnRunClustering.setEnabled(False)
        self.btnRunClustering.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.btnRunClustering)
        layout.setAlignment(self.btnRunClustering, Qt.AlignCenter)

        # Button to save outliers
        self.btnSaveOutliers = QPushButton('Save Outliers', self)
        self.btnSaveOutliers.clicked.connect(self.saveOutliers)
        self.btnSaveOutliers.setEnabled(False)
        self.btnSaveOutliers.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.btnSaveOutliers)
        layout.setAlignment(self.btnSaveOutliers, Qt.AlignCenter)

        # Set main layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initially set DBSCAN as default and adjust visibility of parameter fields
        self.methodComboBox.setCurrentIndex(0)
        self.onMethodChanged('DBSCAN')
        self.updateButtonStyles()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            self.loadCSV(fileName)

    def loadCSV(self, filePath):
            try:
                self.df = pd.read_csv(filePath, sep='[;,]',engine='python',on_bad_lines='skip', quotechar='"', escapechar='\\')
                self.listWidget.clear()
                for column in self.df.columns:
                    self.listWidget.addItem(column)
                self.btnRunClustering.setEnabled(True)
                self.updateButtonStyles()
            except Exception as e:
                print("Error loading CSV:", e)


    def onMethodChanged(self, method):
        is_dbscan = method == 'DBSCAN'
        is_mahalanobis = method == 'Mahalanobis'
        for w in [self.epsLabel, self.epsInput, self.minSamplesLabel, self.minSamplesInput]:
            w.setVisible(is_dbscan)
        for w in [self.thresholdLabel, self.thresholdInput]:
            w.setVisible(is_mahalanobis)

    def runClustering(self):
        selectedColumns = [item.text() for item in self.listWidget.selectedItems()]
        if not selectedColumns:
            print("No columns selected")
            return

        chosenMethod = self.methodComboBox.currentText()
        if chosenMethod == 'DBSCAN':
            try:
                eps = float(self.epsInput.text())
                min_samples = int(self.minSamplesInput.text())
                self.outliers_df = perform_dbscan_clustering(self.df, selectedColumns, eps, min_samples)
            except ValueError:
                print("Invalid DBSCAN parameters")
        elif chosenMethod == 'Mahalanobis':
            try:
                threshold = float(self.thresholdInput.text())
                self.outliers_df = mahalanobis_plot(self.df, selectedColumns, threshold)
            except ValueError:
                print("Invalid Mahalanobis threshold")

        print(self.outliers_df)

        if self.outliers_df is not None :
            self.btnSaveOutliers.setEnabled(True)
        else:
            self.btnSaveOutliers.setEnabled(False)
            print("No outliers found or error in clustering.")
        self.updateButtonStyles()

    def saveOutliers(self):
        if self.outliers_df is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Outliers CSVs", "")
            
            if folder_path:
                base_name, ok = QInputDialog.getText(self, "Enter Base Name", "Enter the base name for CSV files:")
                
                if ok:
                    for i, outliers_df in enumerate(self.outliers_df):
                        file_name = f"{base_name}_{i}.csv"
                        file_path = os.path.join(folder_path, file_name)
                        outliers_df.to_csv(file_path, index=False)
                        
                    print(f"Outliers saved to {folder_path}")
                else:
                    print("Base name not provided.")
            else:
                print("No folder selected.")
        else:
            print("No outliers to save.")



            

    def updateButtonStyles(self):
        buttonStyleEnabled = """
            QPushButton {
                background-color: #2074e3;
                color: white;
                border-radius: 5px;
                padding: 5px 40px;
                min-height: 20px;
                font-size: 14px;
            }
        """
        buttonStyleDisabled = """
            QPushButton {
                background-color: #cccccc;
                color: #666666;
                border-radius: 5px;
                padding: 5px 40px;
                min-height: 20px;
                font-size: 14px;
            }
            
        """


        csvButtonStyle = """
            QPushButton {
                background-color: #1D6F42;
                color: white;
                border-radius: 5px;
                padding: 5px 20px;
                font-size: 14px;
            }
        """


        saveButtonStyle = """
            QPushButton {
                background-color: #1D6F42;
                color: white;
                border-radius: 5px;
                padding: 5px 20px;
                font-size: 14px;
            }
        """

        # Update only the buttons that should be dynamically styled based on their enabled state
        self.btnRunClustering.setStyleSheet(buttonStyleEnabled if self.btnRunClustering.isEnabled() else buttonStyleDisabled)
        self.btnSaveOutliers.setStyleSheet(buttonStyleEnabled if self.btnSaveOutliers.isEnabled() else buttonStyleDisabled)

        # Set a consistent style for the 'Open CSV' button
        self.btnOpenCSV.setStyleSheet(csvButtonStyle)


def main():
    app = QApplication(sys.argv)
    ex = ClusteringGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
