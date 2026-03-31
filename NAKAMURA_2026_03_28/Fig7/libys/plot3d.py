from matplotlib import artist
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


class Axes3DForceZOrder(Axes3D):

    @artist.allow_rasterization
    def draw(self, renderer):
        # draw the background patch
        self.patch.draw(renderer)
        self._frameon = False

        # first, set the aspect
        # this is duplicated from `axes._base._AxesBase.draw`
        # but must be called before any of the artist are drawn as
        # it adjusts the view limits and the size of the bounding box
        # of the axes
        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        # add the projection matrix to the renderer
        self.M = self.get_proj()
        renderer.M = self.M
        renderer.vvec = self.vvec
        renderer.eye = self.eye
        renderer.get_axis_position = self.get_axis_position
        print(renderer)
        for col in self.collections:
            col.do_3d_projection(renderer)
        for patch in self.patches:
            patch.do_3d_projection(renderer)

        if self._axis3don:
            # Draw panes first
            for axis in self._get_axis_list():
                axis.draw_pane(renderer)
            # Then axes
            for axis in self._get_axis_list():
                axis.draw(renderer)

        # Then rest
        Axes.draw(self, renderer)
