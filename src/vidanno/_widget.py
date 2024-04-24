
from typing import TYPE_CHECKING

from vidanno.vidanno_gui import Ui_Form as gui
from qtpy.QtWidgets import (QWidget,QVBoxLayout, QFrame, QSizePolicy, QSlider, QComboBox,QLineEdit, QProgressBar, QLabel, QCheckBox, QGridLayout)
from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import skvideo.io
import traceback
import os
import cv2
import numpy as np
from functools import partial
import matplotlib.colors as mcolors

if TYPE_CHECKING:
    import napari


class VidAnnoWidget(QWidget, gui):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.setLayout(QVBoxLayout())
        self.gui = gui()
        self.pixseq_ui = QFrame()

        self.gui.setupUi(self.pixseq_ui)
        self.layout().addWidget(self.pixseq_ui)

        self.gui.create_label.clicked.connect(self.update_label_dict)
        self.gui.add_label.clicked.connect(self.add_new_label)

        self.viewer.bind_key('c', func = self.copy_selected_shapes)

        self.viewer.bind_key("Control-Left", lambda event: self.move_selected_shapes(viewer=self.viewer, key="left"))
        self.viewer.bind_key("Control-Right", lambda event: self.move_selected_shapes(viewer=self.viewer, key="right"))
        self.viewer.bind_key("Control-Up", lambda event: self.move_selected_shapes(viewer=self.viewer, key="up"))
        self.viewer.bind_key("Control-Down",  lambda event: self.move_selected_shapes(viewer=self.viewer, key="down"))


        # self.viewer.bind_key('c', func=lambda event: self._copy_shapes(viewer=self.viewer, mode="active"), overwrite=True)
        # self.viewer.bind_key('Control-c', func=lambda event: self._copy_shapes(viewer=self.viewer, mode="all"), overwrite=True)

        self.label_dict = {}    

        self.load_sample_data()


    def add_new_label(self):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            if "Shapes" not in layer_names:
                self.viewer.add_shapes(ndim=3, features={"label_name":[]})

            shapes_layer = self.viewer.layers["Shapes"]
            label_name = self.gui.add_label_name.currentText()

            if label_name in self.label_dict.keys():

                label_type = self.label_dict[label_name]["label_type"]
                label_colour = self.label_dict[label_name]["label_colour"]

                if label_type.lower() == "box":
                    shapes_layer.mode = 'add_rectangle'
                if label_type.lower() == "line":
                    shapes_layer.mode = "add_line"
                if label_type.lower() == "polygon":
                    shapes_layer.mode = "add_polygon_lasso"

                shapes_layer.current_edge_color = list(mcolors.to_rgb(label_colour.lower()))
                shapes_layer.current_edge_width = 5
                shapes_layer.current_properties = {"label_name":[label_name]}
                
        except:
            print(traceback.format_exc())


    def update_label_dict(self):

        try:

            label_name = self.gui.create_label_name.text()
            label_type = self.gui.create_label_type.currentText()
            label_colour = self.gui.create_label_colour.currentText()

            self.label_dict[label_name] = {"label_type":label_type,
                                           "label_colour":label_colour,
                                           }
            
            label_names = self.label_dict.keys()

            combo = self.gui.add_label_name
            combo.clear()
            combo.addItems(label_names)

        except:
            print(traceback.format_exc())




    def move_selected_shapes(self, viewer=None, key="left", distance=1):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            if "Shapes" in layer_names:

                shape_layer = self.viewer.layers["Shapes"]

                shapes = self.shapes_layer.data.copy()

                selected_shape_indices = list(shape_layer.selected_data)

                if len(selected_shape_indices) > 0:

                    for shape_index in selected_shape_indices:

                        selected_shape = shapes[shape_index].copy()

                        if key == "left":
                            selected_shape[:, 2] -= distance
                        elif key == "right":
                            selected_shape[:, 2] += distance
                        elif key == "up":
                            selected_shape[:, 1] -= distance
                        elif key == "down":
                            selected_shape[:, 1] += distance

                        shapes[shape_index] = selected_shape
                    
                    # Update the shape's data in the layer
                    shape_layer.data = shapes

                    # Refresh the layer to update the view
                    shape_layer.refresh()
                    shape_layer.selected_data = set(selected_shape_indices)

        except:
            print(traceback.format_exc())
            pass



    def copy_selected_shapes(self, viewer):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            n_frames = int(self.viewer.dims.range[0][1])
            current_step = list(self.viewer.dims.current_step)
            current_frame = current_step[0]
            next_frame = current_frame + 1

            if "Shapes" in layer_names:

                shape_layer = self.viewer.layers["Shapes"]

                shapes = shape_layer.data

                selected_shape_indices = list(shape_layer.selected_data)

                if len(selected_shape_indices) > 0:

                    new_selection_indices = []

                    for shape_index in selected_shape_indices:

                        selected_shape = shapes[shape_index].copy()
                        selected_shape[:, 0] = next_frame

                        shapes.append(selected_shape)
                        new_selection_indices.append(len(shapes)-1)

                    shape_layer.data = shapes

                    current_step[0] = next_frame
                    self.viewer.dims.current_step = current_step

                    shape_layer.refresh()
                    shape_layer.selected_data = set(new_selection_indices)
                    


        except:
            print(traceback.format_exc())



    def _copy_shapes(self, viewer = None, mode ="active"):

        try:

            layer_names = [layer.name for layer in self.viewer.layers]

            n_frames = int(self.viewer.dims.range[0][1])
            current_frame = self.viewer.dims.current_step[0]

            if mode == "active":
                frame_list = [current_frame + 1]
            else:
                frame_list = np.arange(n_frames).tolist()
                frame_list.pop(current_frame)
 
            if "Shapes" in layer_names:

                shape_layer = self.viewer.layers["Shapes"]

                shapes = shape_layer.data.copy()

                shape_indeces = [index for index, shape in enumerate(shapes) if shape[0][0] == current_frame]

                if len(shape_indeces) > 0:

                    for frame in frame_list:

                        for shape_index in shape_indeces:

                            active_shape = shapes[shape_index].copy()
                            active_shape[:, 0] = frame

                            shapes.append(active_shape)

                    shape_layer.data = shapes
                    print(shapes)
        except:
            print(traceback.format_exc())



    
    def load_sample_data(self):
        try:
            path = "pandas.mp4"

            frames = []

            cap = cv2.VideoCapture(path)
            ret = True

            while ret:
                ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
                if ret:
                    frames.append(img)

            video = np.stack(frames, axis=0) # dimensions (T, H, W, C)

            self.image_layer = self.viewer.add_image(video)
            self.shapes_layer = self.viewer.add_shapes(ndim=3, properties={"label_name":[]})

            self.shapes_layer.events.data.connect(self.update_shapes)

        except:
            print(traceback.format_exc())
  

    def update_shapes(self, event):

        try:

            shapes_layer = self.viewer.layers["Shapes"]
            shapes = shapes_layer.data

            print(shapes_layer.properties)

            if event.action == "adding":
                pass
            if event.action == "added":
                shape_index = len(shapes)

                label_name = self.gui.add_label_name.currentText()

                if label_name in self.label_dict.keys():

                    # print(True)

                    label_colour = self.label_dict[label_name]["label_colour"]
                    # edge_colour = list(mcolors.to_rgb(label_colour.lower()))
                    # edge_colour.append(1)
                    
                    # shapes_layer.edge_color[-1] = edge_colour

                    # shapes_layer.refresh()

                    # print(shapes_layer.edge_color)

                    # print(shapes_layer.text)



            if event.action == "changing":
                pass
            if event.action =="changed":
                pass
        except:
            print(traceback.format_exc())
