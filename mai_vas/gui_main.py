# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 18:25:55 2025

@author: Stepan

Main GUI script for MAI-VAS with PySide2
"""
import sys

from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QPushButton, 
    QPlainTextEdit, 
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, 
    QProgressBar,
)

from PySide2.QtGui import QFont
    
from gui_widgets import PathWidget, ModelChoiceWidget,  ImageDialog, ModeSelectorWidget
from gui_worker import MaiVasWorker, MaiVasModel
 
class MainWindow(QMainWindow):
    
    start_process = Signal()
    error_signal = Signal(str) # For error handling within the mai-vas class
    
    def __init__(self):
        super().__init__()
        
        # Init basics
        self.mai_vas = MaiVasModel()
        self.thread = QThread()
        self.worker = MaiVasWorker(self.mai_vas)
        
        # Process threading
        self.start_process.connect(self.worker.run)
        self.worker.finished.connect(self.finish_execution)
        self.worker.image.connect(self.show_image_dialog)
        self.worker.update.connect(self.message)
        
        self.worker.moveToThread(self.thread)
        self.thread.start()

        # GUI
        self.setWindowTitle("MAI-VAS GUI")
        self.resize(1000, 600)
        self.setStyleSheet('QMainWindow {background-color:rgb(250, 250, 250)}')
        
        # FONT
        self.dialogue_font = QFont('Arial', 10)
        self.button_font = QFont('Arial', 12)
        self.label_font = QFont('Arial', 10)
        
        # Calculate button
        self.calculate_button = QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.execute_mai_vas)
        self.calculate_button.setMinimumSize(100, 50)
        self.calculate_button.setFont(self.button_font)
        self.calculate_button.setToolTip('Run MAI-VAS with selected parameters.')
        self.calculate_button.setStyleSheet('QPushButton {background-color:rgb(76, 175, 80); color:white; border:1px solid black}' \
                                            'QPushButton:hover {background-color: rgb(102, 187, 106)}' \
                                            'QPushButton:disabled {background-color: rgb(204, 204, 204); border:1px solid rgb(204, 204, 204)}')      

        # Terminate button
        self.terminate_button = QPushButton('Cancel')
        self.terminate_button.clicked.connect(self.terminate_mai_vas)
        self.terminate_button.setEnabled(False)
        self.terminate_button.setMinimumSize(100, 50)
        self.terminate_button.setStyleSheet('QPushButton {color:red}')
        self.terminate_button.setFont(self.button_font)
        self.terminate_button.setToolTip('Stop execution.')

        # Progress bar init
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,100)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.progress_bar.hide()
        
        # Run button layout
        self.run_buttons = QWidget()
        l = QGridLayout()
        l.setContentsMargins(0,0,0,0) # Removes margin to be left justified.
        l.addWidget(self.progress_bar,0, 0, 1, 2)
        l.addWidget(self.calculate_button,1,1)
        l.addWidget(self.terminate_button,1,0)
        self.run_buttons.setLayout(l)
        
        # Terminal window
        self.dialogue_box = QPlainTextEdit()
        self.dialogue_box.setReadOnly(True)
        self.dialogue_box.setStyleSheet('QPlainTextEdit {border: 2px solid black}')
        self.dialogue_box.setFont(self.dialogue_font)

        # Initial text
        self.message('MAI-VAS GUI Ready:')
        self.message('Please select how you would like to load the data, model parameters, and save locations.')
        
        # Save location
        self.save_path_widget = PathWidget(default = self.mai_vas.save_path, dirs_only = True, label = 'Choose where to save the result file:', font = self.label_font)
        self.save_path_widget.path_box.setPlaceholderText(f'default: {self.mai_vas.save_path}')
        self.save_path_widget.path_signal.connect(self.worker.set_save_path)
        self.save_path_widget.error_signal.connect(self.message)
        self.save_path_widget.error_signal.connect(self.worker.stop)
        self.save_path_widget.setToolTip(f'Choose were you would like for the output to be saved (default is {self.save_path_widget.default})')
        
        # Data location
        self.data_path_widget = PathWidget(default = self.mai_vas.data_path, dirs_only = True, label = 'Select the location of data:', font = self.label_font)
        self.data_path_widget.path_box.setPlaceholderText(f'default: {self.mai_vas.data_path}')
        self.data_path_widget.path_signal.connect(self.worker.set_data_path)
        self.data_path_widget.error_signal.connect(self.message)
        self.data_path_widget.error_signal.connect(self.worker.stop)
        self.data_path_widget.setToolTip(f'Choose were your data csv is located (default is {self.data_path_widget.default})')
        
        # Model selector
        self.model_widget = ModelChoiceWidget(label = 'Select model parameters:', font = self.label_font)
        self.model_widget.view_signal.connect(self.worker.set_view)
        self.model_widget.format_signal.connect(self.worker.set_format)
        self.worker.view_signal.connect(self.model_widget.view_box.setCurrentText)
        self.worker.format_signal.connect(self.model_widget.image_format_box.setCurrentText)
        self.model_widget.view_box.hide() # Hide viewbox as the default init does not need it.
        
        # Mode selection
        self.data_input_kind_widget = ModeSelectorWidget(items_list = ['From folder', 'Single-image','From metadata'],
                                                        label = 'Select data loading mode:', font = self.label_font)
        self.data_input_kind_widget.menu.currentIndexChanged.connect(self.worker.set_mode)                      # Actually changed mode
        self.data_input_kind_widget.menu.currentIndexChanged.connect(self.data_path_widget._change_dialog_mode) # FileDialog slot 
        self.data_input_kind_widget.menu.currentIndexChanged.connect(self.hide_model_choice_box)                # Hides model selector

        # Menu bar layout
        self.menu_widget = QWidget()
        l = QVBoxLayout()
        l.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        l.setSpacing(25)
        
        l.addWidget(self.data_input_kind_widget)
        l.addWidget(self.model_widget)
        l.addSpacing(25)
        l.addWidget(self.data_path_widget)
        l.addWidget(self.save_path_widget)
        l.addSpacing(25)
        l.addWidget(self.run_buttons)

        self.menu_widget.setLayout(l)

        # Entire window layout
        l = QHBoxLayout()
        l.addWidget(self.menu_widget, 1)
        l.addWidget(self.dialogue_box, 2)

        w = QWidget()
        w.setLayout(l)
        
        # Display central widget
        self.setCentralWidget(w)
       
    # Method to parse messages onto the GUI
    def message(self, s):
        self.dialogue_box.appendPlainText(s)
    
    # Execute code upon button press
    def execute_mai_vas(self):
        self.calculate_button.setEnabled(False)
        self.terminate_button.setEnabled(True)
        self.progress_bar.show()
        self.start_process.emit()
        
    # Finalise code execution
    def finish_execution(self):
        if not self.worker._running:
            self.worker.restart()
        self.calculate_button.setEnabled(True)
        self.terminate_button.setEnabled(False)
        self.progress_bar.hide()
        self.progress_bar.reset()
        self.message('Ready: \n')

    # Method to handle the terminate button
    def terminate_mai_vas(self):
        self.terminate_button.setEnabled(False)
        self.message('Terminating execution...')
        self.worker.stop()
         
    # Handles what options are displayed based on mode
    def hide_model_choice_box(self, mode):
        if mode == 2:
            self.model_widget.view_box.show()
            self.model_widget.image_format_box.show()
            self.model_widget.label_widget.show()
        elif mode == 0:
            self.model_widget.view_box.hide()
            self.model_widget.image_format_box.show()
            self.model_widget.label_widget.show()
        elif mode == 1:
            self.model_widget.view_box.hide()
            self.model_widget.image_format_box.hide()
            self.model_widget.label_widget.hide()
            
    # Displays image after execution in image-mode
    def show_image_dialog(self, image):
        self.image_dialog = ImageDialog(image)
        self.image_dialog.show()
            
    # Kill method
    def closeEvent(self, event):
        if self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            
        super().closeEvent(event)
             
    # Handle enter key press event to always run calculate button
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.calculate_button.click() 
    
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()