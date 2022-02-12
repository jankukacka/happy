# ------------------------------------------------------------------------------
#  File: io.py
#  Author: Jan Kukacka
#  Date: 11/2018
# ------------------------------------------------------------------------------
#  Common functions for input/output operations
# ------------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
import os

## For cached file access
__file_cache = {}


def load(filename, **kwargs):
    '''
    Loads file. Handles multiple formats:
        - .mat
        - .nii / .nii.gz
        - .mha
        - .hdr
        - .npy / .npz
        - .pkl
        - .json
        - .png
        - .jpeg / .jpg
        - .tiff / .tif
        - .bmp
        - .gif
        - dicom series (folder with .dcm)
        - .txt
    and selects the appropriate method to open them.

    # Arguments
    - filename: string with the file path or file_info dictionary containing
        the "filename" record.
    - kwargs:
        - key: string. Key in .mat or .npz file.
        - backend: string. For image files, may specify backend library (PIL or imageio)
    '''
    ## Handle cases when file_info is given
    if isinstance(filename, dict):
        return load(**filename)

    if not Path(filename).exists():
        raise FileNotFoundError(str(filename))
        # raise Exception('File not found')

    filename_str = str(filename).lower()

    if filename_str.endswith('.mat'):
        return load_matlab(filename, **kwargs)

    if (filename_str.endswith('.nii.gz')
        or filename_str.endswith('.nii')
        or filename_str.endswith('.mha')
        or filename_str.endswith('.hdr')):
        return load_nifti(filename)

    if filename_str.endswith('.npz') or filename_str.endswith('.npy'):
        return load_numpy(filename, **kwargs)

    if filename_str.endswith('.pkl'):
        return load_pickle(filename)

    if filename_str.endswith('.json'):
        return load_json(filename)

    if filename_str.endswith('.txt'):
        return load_txt(filename)

    if (filename_str.endswith('.png')
        or filename_str.endswith('.jpeg')
        or filename_str.endswith('.jpg')
        or filename_str.endswith('.tif')
        or filename_str.endswith('.tiff')
        or filename_str.endswith('.bmp')
        or filename_str.endswith('.gif')):
        return load_image(filename, **kwargs)

    if filename_str.endswith('.csv'):
        return load_csv(filename, **kwargs)

    if filename_str.endswith('.nrrd'):
        return load_nrrd(filename, **kwargs)

    ## DICOMs: filename is a name of a folder that contains some .dcm files
    if (os.path.isdir(filename)
        and any(file.endswith('.dcm') for file in os.listdir(filename))):
        return load_dicom_series(filename)

    raise Exception('Could not read the file')


def save(filename, data, overwrite=False, parents=False, **kwargs):
    '''
    Saves data to a file. Supports various file formats:
        - Nifti: .nii.gz
        - gif: .gif
        - pickle: .pkl
        - nrrd: .nrrd
        - numpy: .npy or .npz
        - json: .json
        - png: .png
        - csv: .csv
        - txt: .txt

    # Arguments:
        - filename: string with the file path or file_info dictionary containing
            the "filename" record.
        - data: array or dict or whatever suitable data to save.
        - overwrite: bool. If the file exists, is it okay to overwrite it?
            Default False.
        - parents: bool. Should the parent directories be created if they don't
            already exist? Default False.
        - kwargs:
            - is_vector: for nifti images. See save_nifti for details.
            - spacing: for nifti images. See save_nifti for details.
            - key: string. Key in .mat or .npz file.
    '''
    ## file_info dict
    if isinstance(filename, dict):
        filename = filename['filename']

    filename = Path(filename)

    ## Check if file exists
    ## Note: not safe to race conditions. If overwrite is False but some other
    ##       process creates the file after this check, it will get overwritten.
    if filename.exists() and not overwrite:
        try:
            import os
            raise OSError(os.errno.EEXIST, 'File exists. Use overwrite=True to overwrite it.')
        except AttributeError:
            import errno
            raise OSError(errno.EEXIST, 'File exists. Use overwrite=True to overwrite it.')

    ## Check if directory exists
    if not filename.parent.exists() and parents:
        filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.name.endswith('.nii.gz'):
        save_nifti(filename, data, **kwargs)
    elif filename.name.endswith('.npy'):
        save_numpy(filename, data)
    elif filename.name.endswith('.npz'):
        save_numpy_archive(filename, data)
    elif filename.name.endswith('.nrrd'):
        save_nrrd(filename, data)
    elif filename.name.endswith('.pkl'):
        save_pickle(filename, data)
    elif filename.name.endswith('.gif'):
        save_gif(filename, data, **kwargs)
    elif filename.name.endswith('.json'):
        save_json(filename, data, **kwargs)
    elif filename.name.endswith('.png'):
        save_png(filename, data, **kwargs)
    elif filename.name.endswith('.csv'):
        save_csv(filename, data, **kwargs)
    elif filename.name.endswith('.txt'):
        save_txt(filename, data)
    else:
        raise Exception('Unsupported filetype.')


def load_file_cached_local(filename, copy=True, **kwargs):
    '''
    Works just as load(), but uses internal cache to avoid re-loading the same
    file multiple times. There is no cache size control in place, so use with
    caution.
    '''
    global __file_cache
    ## Handle file_info dictionaries
    if type(filename) == dict:
        key = filename['filename']
    else:
        key = filename
    if key not in __file_cache:
        __file_cache[key] = load(filename, **kwargs)
    if not copy:
        return __file_cache[key]
    else:
        return deepcopy(__file_cache[key])


def load_pickle(filename):
    '''
    Loads pickle file by its filename.
    '''
    ## Pickle files
    import pickle
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_pickle(filename, data):
    '''
    Save pickle file by its filename.
    '''
    ## Pickle files
    import pickle
    with open(filename, 'wb') as file:
        return pickle.dump(data, file, protocol=4)


def load_nifti(filename):
    '''
    Loads nifti file by its filename.
    '''
    ## Nifti files
    ## TODO: include spacing / resolution information
    import SimpleITK as sitk
    img = sitk.ReadImage(str(filename))
    return sitk.GetArrayFromImage(img)


def save_nifti(filename, data, is_vector=False, spacing=None, direction=None,
               origin=None):
    '''
    Saves "data" into nifti file "filename".

    # Arguments:
        - filename: string or Path with the full path to the output file.
        - data: 2D or 3D array with the image to save.
        - is_vector: bool. indicates if it is 3D image (False) or 2D
            multichannel image (True).
        - spacing: (optional) int or iterable of ints giving voxel spacing in
            the image in milimeters.
        - direction: (optional) information about the dimension ordering.
            e.g. 9-tuple of zeros and ones
        - origin: (optional) information about the image coord origin.
    '''
    ## Nifti files
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(data, isVector=is_vector)

    if spacing is not None:
        ndims = data.ndim
        if is_vector:
            ndims -= 1
        if not hasattr(spacing, '__iter__'):
            spacing = [spacing] * ndims
        img.SetSpacing(spacing)

    if direction is not None:
        img.SetDirection(direction)

    if origin is not None:
        img.SetOrigin(origin)

    sitk.WriteImage(img, str(filename))


def load_numpy(filename, **kwargs):
    '''
    Loads numpy file by its filename and key (if it is a npz archive).
    '''
    import numpy as np
    file = np.load(filename)

    try:
        keys = file.keys()
        if 'key' not in kwargs:
            ## Is there a single key?
            if len(keys) == 1:
                return file[keys[0]]
            else:
                return file
        else:
            return file[kwargs['key']]
    except AttributeError:
        ## npy file has no keys
        return file


def save_numpy(filename, data):
    '''
    Saves a numpy array to a npy file
    '''
    import numpy as np
    np.save(filename, data)


def save_numpy_archive(filename, data):
    '''
    Saves a dict of arrays to a npz archive
    '''
    import numpy as np
    np.savez_compressed(filename, **data)


def load_matlab(filename, **kwargs):
    '''
    Loads matlab file by its filename and possibly key
    '''
    ## Matlab files
    import scipy.io
    import numpy as np
    try:
        file = scipy.io.loadmat(filename)
    except NotImplementedError:
        ## Thrown for h5py mat files
        import h5py
        file = h5py.File(filename,'r')

    if 'key' not in kwargs:
        ## Try automatically estimate the relevant key
        ## (the only one not starting with "__")
        keys = file.keys()
        keys = list(filter(lambda x: not x.startswith('__'), keys))
        if len(keys) == 1:
            ## Convert hdf files to ndarray but avoid unnecessary copy
            if isinstance(file[keys[0]], np.ndarray):
                return file[keys[0]]
            else:
                return np.array(file[keys[0]])
        else:
            return file
    else:
        ## Convert hdf files to ndarray but avoid unnecessary copy
        if isinstance(file[kwargs['key']], np.ndarray):
            return file[kwargs['key']]
        else:
            return np.array(file[kwargs['key']])


def load_json(filename):
    '''
    Loads json files
    '''
    import json
    with open(filename, 'r') as file:
        return json.load(file)


def save_json(filename, data):
    '''
    Saves json files
    '''
    import json
    with open(filename, 'w') as file:
        json.dump(data, file)


def load_image(filename, backend=None):
    '''
    Loads image, using a specific backend if needed

    # Arguments:
        - backend: 'PIL' or 'imageio'
    '''
    import numpy as np

    def _load_imageio(filename):
        import imageio
        return imageio.imread(filename)
    def _load_pil(filename):
        from PIL import Image
        return np.array(Image.open(filename))

    backends =  {'imageio': _load_imageio, 'PIL': _load_pil}

    if backend is not None:
        return backends[backend](filename)

    else:
        for backend in backends.values():
            try:
                return backend(filename)
            except ImportError:
                ## If some backend is not installed, try falling back to another
                pass
    raise Error('No suitable backend found. Install PIL or imageio to read images.')


def load_png(filename):
    '''
    Loads png files
    '''
    import imageio
    return imageio.imread(filename)


def save_png(filename, data, **kwargs):
    '''
    Saves png files
    '''
    import imageio
    return imageio.imwrite(filename, data, **kwargs)


def save_gif(filename, data, **kwargs):
    '''
    Saves gif image.

    # Arguments:
        - data: iterable with frames. Each frame should be of shape (height, width),
            (h,w,3) or (h,w,4).
        - filename: Output filename (full path). Should end with '.gif'
    '''
    import imageio
    imageio.mimsave(filename, data, **kwargs)


def load_dicom_series(folder_path):
    '''
    Loads a series of dicom images from a folder specified by folder_path.
    Based on https://simpleitk.readthedocs.io/en/master/Examples/DicomSeriesReader/Documentation.html

    # Arguments:
        - folder_path: Path to the folder with .dcm files to load.

    # Returns:
        - np array with image series
    '''
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image)

def ensure_folder(path):
    '''Make sure a folder exists. Returns False if it cannot be created.'''
    import os
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            # print "Couldn't create output directory."
            return False
    return True

def load_csv(filename, **kwargs):
    '''
    Loads csv files.

    kwargs can use keyword 'pandas'. If True, Pandas module will be used.
    '''
    if 'pandas' in kwargs and kwargs['pandas']:
        import pandas
        return pandas.read_csv(filename)

    import csv
    data = []
    if 'delimiter' not in kwargs:
        kwargs['delimiter'] = ';'
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, **kwargs)
        for row in reader:
            data.append(row)
    return data


def save_csv(filename, data, **kwargs):
    '''
    Saves csv files.

    If data is Pandas.DataFrame, DataFrame.to_csv() function is used.
    Otherwise, this functionality is not implemented yet.
    '''
    try:
        ## Try if it is pandas DataFrame
        data.to_csv(filename, **kwargs)
    except AttributeError as e:
        raise NotImplementedError('csv saving is only supported for Pandas data frames.')


def load_nrrd(filename, header=False):
    '''
    Loads nrrd files

    # Arguments:
        - header: bool. If True, returns also header dictionary as a second
            return item
    '''
    import nrrd
    data, file_header = nrrd.read(filename)
    if not header:
        return data
    else:
        return data, file_header


def save_nrrd(filename, data):
    '''
    Saves nrrd files
    '''
    import nrrd
    nrrd.write(str(filename), data)


def save_txt(filename, data):
    '''
    Saves txt files
    '''
    with open(filename, 'w') as file:
        file.write(data)

def load_txt(filename):
    '''
    Saves txt files
    '''
    with open(filename, 'r') as file:
        return file.read()
