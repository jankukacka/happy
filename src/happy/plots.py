# ------------------------------------------------------------------------------
#  File: plots.py
#  Author: Jan Kukacka
#  Date: 7/2018
# ------------------------------------------------------------------------------
#  Plot utilities and standard plot functions
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.interpolate import interp1d
from .misc import ensure_list


__transparent = '#00000000'


def hide_ticks(axes=None, axis='both'):
    '''
    Hide ticks and their labels.

    # Arguments:
        - axes: axes object to hide ticks on. plt.gca() by default.
        - axis: 'x', 'y', 'both' (default). Which axis will be affected.
    '''
    if axes is None:
        axes = plt.gca()

    if axis == 'both' or axis == 'y':
        axes.set_yticklabels([])
        axes.yaxis.set_tick_params(length=0)
    if axis == 'both' or axis == 'x':
        axes.set_xticklabels([])
        axes.xaxis.set_tick_params(length=0)


def set_spine_thickness(axes=None, thickness=1):
    '''
    Allows setting thickness of spines of given axes

    # Arguments:
        - axes: axes object. Default is plt.gca()
        - thickness: positive float. Desired thicnkess.
    '''
    if axes is None:
        axes = plt.gca()

    for spine in axes.spines.values():
        spine.set_linewidth(thickness)


def scalebar(axes, res, ax_shape, len_cm=.5, color='w'):
    '''
    Displays scalebar on the plot.

    # Arguments:
        - axes: axes object to display scalebar on.
        - res: image resolution in horizontal dimension in microns
        - ax_shape: (height, width) of the displayed image
        - len_cm: float. Lenght of the scalebar in centimeters
        - color: valid matplotlib color. Color of the scalebar.
    '''
    len_pixels = len_cm * 10000 / res
    left = 0.05 * ax_shape[1]
    height_cm = .15
    height_in = height_cm * 0.393701
    ## Compute height and left-bottom point in display coordinates
    dpi_bottom = axes.transAxes.transform([0.05,0.95])
    dpi_height = axes.figure.dpi_scale_trans.transform([height_in]*2)
    # print(dpi_bottom)
    # print(dpi_height)

    ## Compute new bottom as bottom - height (shifted by height up)
    bottom = axes.transData.inverted().transform(dpi_bottom - dpi_height)
    # print(bottom)
    ## Since transData considers origin in lower left corner, we have to
    ## compute it like this:
    bottom = ax_shape[0] - bottom[1]
    ## Infer height
    height_pixels = 0.95 * ax_shape[0] - bottom
    # print(height_pixels)
    axes.add_artist(plt.Rectangle([left,bottom], len_pixels, height_pixels, color=color))


def cmap(*keypoints):
    '''
    Returns a linear segmented colormap from given keypoints.

    # Arguments
        - keypoints: Array-like of keypoints which can be either a color or a
            2-tuple of (index, color), where index is in range [0;1] and
            provides the location of the keypoint. If the index is not given,
            keypoints will be distributed equally on the range. `color` is any
            color format accepted by matplotlib, and `'t'` can be used for
            transparent color.

    # Returns:
        - colormap

    # Examples:
        Colormap ranging from transparent to red
        >>> cmap = happy.plots.cmap(['t', 'r'])
        Colormap with positions specified and transparent upper end
        >>> cmap = happy.plots.cmap([(0, 'black'), (0.2, 'xkcd: green'),
                                     (254/255, 'r'), (1, 't')])
    '''
    ## For compatibility with old version where keypoints was not *args parameter
    if len(keypoints) == 1:
        keypoints = keypoints[0]
    ## *keypoints is a tuple but we need to allow assignment in the transparent
    ## color replacement
    keypoints = list(ensure_list(keypoints))

    ## Replace color 't' for transparent
    for i in range(len(keypoints)):
        if keypoints[i] == 't':
            keypoints[i] = __transparent
        else:
            try:
                if keypoints[i][1] == 't':
                    ## Replace whole keypoint since old keypoint may be
                    ## immutable (e.g. a tuple)
                    keypoints[i] = (keypoints[i][0], __transparent)
            except:
                pass
    colormap = mc.LinearSegmentedColormap.from_list('', colors=keypoints)
    return colormap


def cmap_clip(colors=None, clip_up=True, clip_down=True):
    '''
    Creates colormap with gradient of the two given colors and transparent color
    for max and min value. Ideal for overlays on images.

    # Arguments:
        - colors: Array-like with colors of length at least 2. If None, 'red'
            and 'blue' are used.
        - clip_up: Bool. If True, highest values will be clipped. Default True.
        - clip_down: Bool. If True, lowest values will be clipped. Default True.
    '''
    if colors is None:
        colors = ('red', 'blue')
    keypoints = []
    if clip_down:
        keypoints.append((0, __transparent))
        keypoints.append((1/256, colors[0]))
    else:
        keypoints.append((0, colors[0]))

    ## Intermediate colors
    intermediate_spacing = 256 / (len(colors)-1)
    for i in range(len(colors)-2):
        keypoints.append(((i+1)*intermediate_spacing / 256, colors[i+1]))

    if clip_up:
        keypoints.append((255/256, colors[-1]))
        keypoints.append((1, __transparent))
    else:
        keypoints.append((1, colors[-1]))
    return cmap(keypoints)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def normalize_color_list(colors, list_len):
    '''
    Ensures that the variable "colors" contains a list of colors of length
    list_len. If colors is a single color, it is repeated. If it is a list
    of colors already, it is expanded to the right lenght by repeating the
    last element or trimmed if it is too long.
    Also ensures that if colors is a single color represented as a list, it
    will be correctly understood.

    # Arguments:
        - colors: Color or list of colors
        - list_len: positive int. Length of the color list to be produced.

    # Returns:
        - color_list: List of colors of lenght list_len.
    '''
    if mc.is_color_like(colors):
        colors = [colors]
    else:
        colors = ensure_list(colors)
    if len(colors) < list_len:
        ## Expand short list
        colors = colors + [colors[-1]]*(list_len-len(colors))
    if len(colors) > list_len:
        ## Truncate long list
        colors = colors[:list_len]
    return colors


def smooth_plot(axes, x, y=None, smoothing=5, **kwargs):
    '''
    Smoothed version of pyplot.plot(). Plots single line on the given axes.

    # Arguments:
        - axes: axes obejct.
        - x, (y): arrays of shape (n,) with x and y coordinates. If y is missing,
            x is considered y and range(0,n) is considered x.
        - smoothing: positive int. how many samples should be interpolated
            between each pair of points?
        - kwargs: passed to normal plot.

    # Returns:
        - list of Line objects
    '''
    length = x.size
    if y is None:
        y = x
        x = np.arange(length)
    x_new = np.linspace(x.min(),x.max(), (length-1)*smoothing+1, endpoint=True)
    y_new = interp1d(x, y, kind='quadratic')
    return axes.plot(x_new, y_new(x_new), marker='o', markevery=smoothing, **kwargs)


def smooth_fill_between(x, y1, y2=0, smoothing=5, axes=None, **kwargs):
    '''
    Smoothed version of pyplot.fill_between(). Plots single shaded region on the
    given axes.

    # Arguments:
        - axes: axes obejct.
        - x, (y1, y2): arrays of shape (n,) with x and y coordinates. If y2 is missing,
            it is considered to be zero
        - smoothing: positive int. how many samples should be interpolated
            between each pair of points?
        - axes: axes object to plot on. If None, plt.gca() is used.
        - kwargs: passed to normal plot.

    # Returns:
        - A PolyCollection with plotted polygons
    '''
    if axes is None:
        axes = plt.gca()
    length = x.size
    x_new = np.linspace(x.min(),x.max(), (length-1)*smoothing+1, endpoint=True)
    y1_new = interp1d(x, y1, kind='quadratic')
    y2_new = interp1d(x, y2, kind='quadratic')
    return axes.fill_between(x_new, y1_new(x_new), y2_new(x_new), **kwargs)


def imshow(axes, img, alpha, cmap=None, norm=None, **kwargs):
    '''
    Extending pyplot.imshow with ability to give alpha as an array

    # Arguments:
        - axes: axes object.
        - img: array of shape (height, width) or (height, width, channels) where
            channels is either 1, 3 or 4. If it is 4 (alpha is given too), then
            the original alpha and given alpha are multiplied.
        - alpha: array of shape (height, width) or scalar (behaves like normal
            imshow then)
        - cmap: colormap to use. Defaults to rcParams default.
        - norm: normalization function to use. Defaults to Normalize.
        - kwargs: passed to pyplot.imshow.
    '''
    ## TODO: Change signature to allow axes=None as in other functions
    ## Set defaults
    if cmap is None:
        ## returns default colormap
        cmap = matplotlib.cm.get_cmap()
    if norm is None:
        norm = mc.Normalize()

    ## Get RGBA representation of the input
    if img.ndim > 2 and img.shape[2] == 1:
        img = img[...,0]
    if img.ndim == 2:
        img_rgba = cmap(norm(img))
    if img.ndim == 3 and img.shape[2] == 3:
        ## Image is already given as colors, append alpha channels
        img_rgba = np.concatenate([img, np.ones(img.shape[:-1] + (1,))],
                                  axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        ## Image is already in the correct format
        img_rgba = img

    ## Blend alpha
    img_rgba[...,-1] = img_rgba[...,-1] * alpha
    return axes.imshow(img_rgba, **kwargs)


def boxplot(data, ax=None, edge_color=None, fill_color=None, **kwargs):
    '''
    Extends the functionality of standard pyplot boxplot function by spcifying
    colors.
    Inspired by https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color

    # Arguments:
        - data: array or sequence of vectors. (Same as original)
        - ax: axes object. If None, plt.gca() is used.
        - edge_color: Color or list of colors for lines and box borders
        - fill_color: Color or list of colors for filling the boxes
        - kwargs: Passed to pyplot.boxplot

    # Retuns:
        - dict of boxplot artists (same as original).
    '''
    if ax is None:
        ax = plt.gca()
    use_patch_artist = fill_color is not None
    bp = ax.boxplot(data, patch_artist=use_patch_artist, **kwargs)

    if edge_color is not None:
        edge_color = normalize_color_list(edge_color, len(bp['boxes']))
        ## Single elements
        for element_type in ['boxes', 'means', 'medians']:
            for element, color in zip(bp[element_type], edge_color):
                plt.setp(element, color=color)

        ## Paired elements
        for element_type in ['whiskers', 'caps']:
            for i, color in enumerate(edge_color):
                plt.setp(bp[element_type][2*i], color=color)
                plt.setp(bp[element_type][2*i+1], color=color)

        ## Fliers (use "markeredgecolor")
        for flier, color in zip(bp['fliers'], edge_color):
            plt.setp(flier, markeredgecolor=color)

    if fill_color is not None:
        fill_color = normalize_color_list(fill_color, len(bp['boxes']))
        for patch, color in zip(bp['boxes'], fill_color):
            patch.set(facecolor=color)
    return bp


def matlab_cmap():
    '''
    Returns matlab colormap
    '''
    return cmap([[0.2081,0.1663,0.5292],[0.2116238095,0.1897809524,0.5776761905],[0.212252381,0.2137714286,0.6269714286],[0.2081,0.2386,0.6770857143],[0.1959047619,0.2644571429,0.7279],[0.1707285714,0.2919380952,0.779247619],[0.1252714286,0.3242428571,0.8302714286],[0.0591333333,0.3598333333,0.8683333333],[0.0116952381,0.3875095238,0.8819571429],[0.0059571429,0.4086142857,0.8828428571],[0.0165142857,0.4266,0.8786333333],[0.032852381,0.4430428571,0.8719571429],[0.0498142857,0.4585714286,0.8640571429],[0.0629333333,0.4736904762,0.8554380952],[0.0722666667,0.4886666667,0.8467],[0.0779428571,0.5039857143,0.8383714286],[0.079347619,0.5200238095,0.8311809524],[0.0749428571,0.5375428571,0.8262714286],[0.0640571429,0.5569857143,0.8239571429],[0.0487714286,0.5772238095,0.8228285714],[0.0343428571,0.5965809524,0.819852381],[0.0265,0.6137,0.8135],[0.0238904762,0.6286619048,0.8037619048],[0.0230904762,0.6417857143,0.7912666667],[0.0227714286,0.6534857143,0.7767571429],[0.0266619048,0.6641952381,0.7607190476],[0.0383714286,0.6742714286,0.743552381],[0.0589714286,0.6837571429,0.7253857143],[0.0843,0.6928333333,0.7061666667],[0.1132952381,0.7015,0.6858571429],[0.1452714286,0.7097571429,0.6646285714],[0.1801333333,0.7176571429,0.6424333333],[0.2178285714,0.7250428571,0.6192619048],[0.2586428571,0.7317142857,0.5954285714],[0.3021714286,0.7376047619,0.5711857143],[0.3481666667,0.7424333333,0.5472666667],[0.3952571429,0.7459,0.5244428571],[0.4420095238,0.7480809524,0.5033142857],[0.4871238095,0.7490619048,0.4839761905],[0.5300285714,0.7491142857,0.4661142857],[0.5708571429,0.7485190476,0.4493904762],[0.609852381,0.7473142857,0.4336857143],[0.6473,0.7456,0.4188],[0.6834190476,0.7434761905,0.4044333333],[0.7184095238,0.7411333333,0.3904761905],[0.7524857143,0.7384,0.3768142857],[0.7858428571,0.7355666667,0.3632714286],[0.8185047619,0.7327333333,0.3497904762],[0.8506571429,0.7299,0.3360285714],[0.8824333333,0.7274333333,0.3217],[0.9139333333,0.7257857143,0.3062761905],[0.9449571429,0.7261142857,0.2886428571],[0.9738952381,0.7313952381,0.266647619],[0.9937714286,0.7454571429,0.240347619],[0.9990428571,0.7653142857,0.2164142857],[0.9955333333,0.7860571429,0.196652381],[0.988,0.8066,0.1793666667],[0.9788571429,0.8271428571,0.1633142857],[0.9697,0.8481380952,0.147452381],[0.9625857143,0.8705142857,0.1309],[0.9588714286,0.8949,0.1132428571],[0.9598238095,0.9218333333,0.0948380952],[0.9661,0.9514428571,0.0755333333],[0.9763,0.9831,0.0538]])
