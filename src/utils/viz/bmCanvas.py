"""bmCanvas.py: Canvas Class for Brain-Mapping Visualization"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/09/2018"


import sys
import os.path
from itertools import cycle
if sys.version_info >= (3,0):
    from PyQt5 import QtGui, QtWidgets
else:
    from PyQt4 import QtGui, QtWidgets
import numpy as np
import vispy
from vispy import app, scene
from vispy import visuals
from vispy.visuals.transforms import STTransform
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.filters import Alpha


# Internal
from utils.util import PRT
from utils.const import RenderingMethod

MARKER_TYPES= ['disc', 'arrow', 'ring', 'clobber', 'square', 'diamond', 'vbar', 'hbar',
                'cross', 'tailed_arrow', 'x', 'triangle_up', 'triangle_down', 'star',
                'o', '+', 's', '-', '|', '->', '>', '^', 'v', '*',]


# create colormaps that work well for translucent and additive volume rendering
class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """

    def __repr__(self):
        return "TransFire"

class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """

    def __repr__(self):
        return "TransGrays"



class BMCanvas(scene.SceneCanvas):
    """BMCanvas Class
        - inherits VisPy SceneCanvas, for drawing 3D(Volumetric) data
    """
    custom_axis = None
    volume = None
    # Setup colormap iterators
    all_cmaps = get_colormaps()
    good_cmap_keys = ['viridis', 'hsl', 'grays', 'light_blues', 'orange',
                      'RdBu', 'GrBu_d', 'hot']  # coolwarm
    def __init__(self, **args):
        """init"""
        try:
            self.logwin = args.pop('logWindow')
        except KeyError:
            pass
        super(BMCanvas, self).__init__(**args)

        # unfreeze to allow additional attributes to the calss inheriting scene.SceneCanvas
        self.unfreeze()

        # init colormaps for canvas
        self.init_colormaps()

        self.index = 0
        self.text_y = 30
        self.markers = {}
        self.markers_legend = {}
        self.key_bindings = {}

        # create view
        #initDataF = "/home/mykim/Downloads/electronic_brain-512.png"
        initDataF = None
        self.init_view(initDataF)

    def init_colormaps(self):
        """init colormaps for canvas chosen by list good_cmap_keys"""

        self.filtered_cmaps = dict((k, self.all_cmaps[k]) for k in self.good_cmap_keys if k in self.all_cmaps)
        self.filtered_cmaps['magenta'] = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        self.filtered_cmaps['cyan'] = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        self.filtered_cmaps['yellow'] = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        self.opaque_cmaps = cycle(self.filtered_cmaps)
        self.translucent_cmaps = cycle([TransFire(), TransGrays()])
        self.opaque_cmap = next(self.opaque_cmaps)
        self.translucent_cmap = next(self.translucent_cmaps)


    def init_view(self, initF=None):
        """initialize 3-D VisPy View Space

        :param showInitData: initialize View Space with initial data if provided
        """
        self.cView = self.central_widget.add_view()

        if initF:
            from scipy import misc
            initData = np.flipud(misc.imread(initF))
            #self.volume = scene.visuals.Volume(initData, parent=self.cView.scene, threshold=0.225, emulate_texture=False)
            self.volume = scene.visuals.Volume(initData, parent=self.cView.scene, threshold=0.225)
        else:
            self.volume = None

        self.marker = None

        self.transform = STTransform(translate=(50., 50., 50.), scale=(100, 100, 100, 1))

        self.cAxis = scene.visuals.XYZAxis(parent=self.cView.scene)
        self.cAxis.transform = self.transform.as_matrix()

        self.mainDescription = scene.visuals.Text("[R:x, G:y, B:z]", pos=(10, self.text_y), anchor_x='left',
                                            bold=True, font_size=10, color='white', parent=self.cView)
        self.text_y += 20


    def add_scalebar(self, data, val=5*1e4):

        z, y, x = data.shape
        yr = min(int(float(y) * 0.1), 30)
        xr = min(int(float(x) * 0.3), 300)
        zr = min(z, 30)

        data_new = np.zeros((z, y + yr, x), dtype=data.dtype)
        data_new[:, 0:y, 0:x] = data.copy()

        # mark scale bar to data
        thick = 5
        for i in range(x-xr, x):
            for j in range(y, y+yr):
                for k in range(zr):
                    data_new[k][j][i] = val

        if "scalebar" not in self.mainDescription.text:
            # calculate scale
            self.mainDescription.text = "%s [scalebar(x,y,z): %dx%dx%d pixel]"%(self.mainDescription.text,
                                                                               xr, yr, zr)

        return data_new


    def set_volume(self, data, alpha=1.0, clim=None, scalebar=False):
        """set data to the view
        NOTE: raw data should have (z, y, x) format

        Parameters
        ------------
        data: a numpy array (should be 3D)
        alpha: transperancy of volume
        """
        if scalebar:
            # add scale bar
            data = self.add_scalebar(data)

        # update axis
        vz, vy, vx = [s/2.0 for s in data.shape]

        if self.volume is None:
            #self.volume = scene.visuals.Volume(data, parent=self.cView.scene, emulate_texture=False, clim=clim)
            self.volume = scene.visuals.Volume(data, parent=self.cView.scene, clim=clim)
            self.volume.attach(Alpha(alpha))
            self.cView.camera = scene.TurntableCamera(parent=self.cView.scene, fov=60., center=(vx,vy,vz))

        else:
            self.volume.visible = False
            if clim is None:
                self.volume.set_data(data, (np.min(data), np.max(data)))
            else:
                self.volume.set_data(data, clim)
            self.volume.visible = True

        self.transform = STTransform(translate=(vx, vy, vz), scale=(vx, vy, vz))     # (y, z, x) axis
        self.cAxis.transform = self.transform.as_matrix()


    def delete_marker(self, key, binding=None):
        del self.markers[key]
        del self.markers_legend[key]
        if binding is not None:
            del self.key_bindings[binding]



#self.overlay_marker(self.mPoints, VizMarkers.GFP_MICROGLIA, "m", color=[1.0, 0.0, 1.0, 0.7])

    def overlay_marker(self, data, name, binding, color=[1.0, 0.0, 0.0, 0.6],
                       alpha=1.0, size=7, symbol='o'):

        """set marker on top of volume
            default symbol: 10 (circle)
        """
        try:
            data = data[..., [2, 1, 0]]     # (z, y, x) -> (x, y, z)
        except IndexError:
            raise IndexError

        if binding in self.key_bindings:
            old_name = self.key_bindings[binding]
            # FIX: there could be updates on the points! so update!
            #if old_name == name:
            #    # nothings to do
            #    return

            #self.log("Key Binding Overlaps! Overwritting to %s"%old_name, PRT.WARNING)
            # overwrite
            self.markers[name] = self.markers.pop(old_name)
            self.markers_legend[name] = self.markers_legend.pop(old_name)
            self.key_bindings[binding] = name

        if name in self.markers.keys():
            self.markers[name].visible = False
            self.markers[name].set_data(data, edge_color=None, face_color=color, size=size,
                                        symbol=symbol)
            self.markers[name].attach(Alpha(alpha))
            self.markers[name].visible = True
            self.markers_legend[name].text = "[%s] %s (%d)"%(binding, name, len(data))

        else:
            marker = scene.visuals.Markers()
            marker.set_data(data, edge_color=None, face_color=color, size=size, symbol=symbol)
            marker.attach(Alpha(alpha))
            self.cView.add(marker)
            self.markers[name] = marker
            legend = scene.visuals.Text("[%s] %s (%d)"%(binding, name, len(data)),
                                                    pos=(10, self.text_y), bold=True, anchor_x='left',
                                                    font_size=10, color=color[:3], parent=self.cView)
            self.markers_legend[name] = legend
            self.text_y += 15
            self.key_bindings[binding] = name


    def create_text(self, pos, text, font_size, color, bold=False, parent=None):
        if parent is None:
            parent = self.cView
        vText = scene.visuals.Text(text, pos=pos, bold=bold, anchor_x='left',
                                  font_size=font_size, color=color, parent=parent)
        return vText


    def set_volume_style(self, method=None, cmapToggle=False, logwin=None):
        """set volume visualization style with rendering method"""

        cmap = None
        if not cmapToggle:
            assert method is not None
            cmap = self.opaque_cmap if method in [RenderingMethod.MIP] else self.translucent_cmap
        else:
            if self.volume is not None:
                if self.volume.method in [RenderingMethod.MIP]:
                    cmap = self.opaque_cmap = next(self.opaque_cmaps)
                else:
                    cmap = self.translucent_cmap = next(self.translucent_cmaps)

        if self.volume is not None:
            msg = "setting "
            if method is not None:
                msg += "rendering method to [{}] ".format(method)
                self.volume.method = method
            if cmap is not None:
                msg += "colormap to [{}]".format(cmap)
                self.volume.cmap = self.filtered_cmaps[cmap]
            try:
                self.log(msg, PRT.STATUS2)
            except AttributeError:
                PRT.p(msg, PRT.STATUS2)


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""
        if self.logwin is None:
            return

        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()


    def toggle_marker(self, key, isOverlayMarker=False):
        """toggle visibility of marker

        Parameters
        ------------
        key: marker key
        isOverlayMarker: whether the marker is Main or Overlay
        """
        if key in self.markers.keys():
            self.markers[key].visible = not self.markers[key].visible

    def on_key_press(self, event):
        """overwrite on_key_press action"""

        for key in self.key_bindings:
            if event.text == key:
                self.toggle_marker(self.key_bindings[key])

        if event.text == ' ':
            if self.marker:
                self.index = (self.index + 1) % (len(MARKER_TYPES))
                self.marker.symbol = MARKER_TYPES[self.index]
                self.update()
