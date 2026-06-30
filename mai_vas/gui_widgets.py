import os
import sys
from pathlib import Path

from PySide2.QtCore import Slot, Signal
from PySide2.QtWidgets import (
    QPushButton, 
    QVBoxLayout, QHBoxLayout,
    QWidget, 
    QLineEdit, 
    QFileDialog,
    QComboBox,
    QDialog,
    QLabel
)

from PySide2.QtGui import QImage, QPixmap

class PathWidget(QWidget):
    """
    Responsible for the save and data paths.
    
    """
    
    path_signal  = Signal(str)     # Signal to change model path option       
    error_signal = Signal(str)     # Signal for errors
    
    def __init__(self, default, dirs_only = False, label = '', font = None):
        super().__init__()
        
        self.path      = os.path.normpath(default)        # Main path
        self.default   = os.path.normpath(default)        # Resets to this if box is empty
        self.dirs_only = dirs_only
        self.label     = label
        self.font      = font

        # Widget for the textbox
        self.path_box = QLineEdit()
        self.path_box.setMaxLength(300)
        self.path_box.setPlaceholderText('')
        self.path_box.editingFinished.connect(self._emit_path)
        
        # Widget for browse button
        self.browse_button = QPushButton('Browse')
        self.browse_button.clicked.connect(self._browse_files)

        # Label
        self.label_widget = QLabel(self.label)
        self.label_widget.setFont(self.font)
        
        # Create Qline + Browse widget
        l = QHBoxLayout()
        l.addWidget(self.path_box)
        l.addWidget(self.browse_button)

        h = QVBoxLayout()
        h.setContentsMargins(0,0,0,0)
        h.setSpacing(5)
        h.addWidget(self.label_widget)
        h.addLayout(l)
        
        self.setLayout(h)
        
    def _emit_path(self):
        self.path = self.path_box.text()
        if self.path == '':
            self.path = self.default
        self.path_signal.emit(self.path)
        
    
    def _browse_files(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(self.default)
        
        if self.dirs_only:
            dialog.setFileMode(QFileDialog.Directory)
                    
        dialog.setViewMode(QFileDialog.ViewMode.List)
        
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
        
            self.path = filename
            self.path_box.setText(filename)
            self._emit_path()
            
    @Slot(int)
    def _change_dialog_mode(self, mode):
        """
        The slot allows for the dialog to dynamically switch between dirs only and files only dependant on the mode selected.
        """
        if mode == 2 or mode == 1:
            self.dirs_only = False
        else:
            self.dirs_only = True


class ModelChoiceWidget(QWidget):
    """
    Widget for model selection
    
    """
    
    view_signal   = Signal(str)
    format_signal = Signal(str)
    
    def __init__(self, label = '', font = None):
        super().__init__()
        
        self.label = label
        self.font = font
        
        # Handle view
        self.view_box = QComboBox()
        self.view_box.addItems(['CC', 'MLO'])
        self.view_box.activated.connect(self._emit_view_parameter)
        self.view_box.setToolTip('Choose view model.')
        
        # Handle image format
        self.image_format_box = QComboBox()
        self.image_format_box.addItems(['RAW', 'PRO'])
        self.image_format_box.activated.connect(self._emit_format_parameter)
        self.image_format_box.setToolTip('Choose which mammography format you want to use.')

        # Add label
        self.label_widget = QLabel(self.label)
        self.label_widget.setFont(self.font)
        
        l = QHBoxLayout()
        l.addWidget(self.view_box)
        l.addWidget(self.image_format_box)

        v = QVBoxLayout()
        v.setContentsMargins(0,0,0,0) # Removes margin. Affects the layout rather than widget itself
        v.addWidget(self.label_widget)
        v.addLayout(l)
        
        self.setLayout(v)
    
    def _emit_view_parameter(self):
        self.view_signal.emit(self.view_box.currentText())
        
    def _emit_format_parameter(self):
        self.format_signal.emit(self.image_format_box.currentText())

class ModeSelectorWidget(QWidget):
    def __init__(self, items_list, label = '', font = None):
        super().__init__()

        self.items_list = items_list
        self.label = label
        self.font = font

        # Init dropdown menu
        self.menu = QComboBox()
        self.menu.addItems(self.items_list)
        self.menu.setToolTip("""Choose the mode for image loading: 
                                        From folder - select folder with data, 
                                        Single-image - select single image,
                                        From metadata - selects a csv with paths to data""")
        
        # Init label
        self.menu_label = QLabel(self.label)
        self.menu_label.setFont(self.font)

        l = QHBoxLayout()
        l.setContentsMargins(0,0,0,0)
        l.addWidget(self.menu_label)
        l.addWidget(self.menu)

        self.setLayout(l)   
        
class ImageDialog(QDialog):
    def __init__(self, image):
        super().__init__()
        
        self.setWindowTitle('Image')
        self.resize(1000,1000)
        
        self.image = image
        
        label = QLabel(self)
        pixmap = self.pil_to_pixmap(self.image)
        label.setPixmap(pixmap)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        
    def pil_to_pixmap(self, pil_image):
        rgb_image = pil_image.convert("RGBA")
        data = rgb_image.tobytes("raw", "RGBA")
        w, h = rgb_image.size
        qimage = QImage(data, w, h, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)