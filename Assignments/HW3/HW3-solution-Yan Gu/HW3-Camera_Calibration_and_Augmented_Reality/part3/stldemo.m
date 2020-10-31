% This is a test demo for plotting a 3D model in xyz axis.
[x, y, z, c] = stlread('opera_de_sydney.stl');
patch(x, y, z, c)