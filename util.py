import cv2, os
import numpy as np
import numpy.typing as npt
import tensorflow as tf

def get_frames(filepath:str) -> npt.NDArray:
    """ Retrieves frames from the video at filepath.
    
    Args:
      filepath: location of an openCV-importable video
    
    Returns:
      An ndarray(n_frames, height, width, depth) containing the frames.
    """
    capture = cv2.VideoCapture(filepath)
    frames = []
    while True:
        ret, frame = capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    return np.stack(frames, axis=0)

def build_frames_dataset() -> tf.data.Dataset:
    """ Returns a data pipeline for frames from the MineRL dataset
    
    The Dataset stages are:
    1. Collect the filepaths of all minerl mp4s
    2. Convert the file contents into batches of frames.
    3. Unbatch the frames.
    """
    # TODO: Tensorflow does not have a stable video decode function, so I use
    #   openCV.  This forces a py_function in the data pipeline.  According to
    #   Tensorflow, this inhibits thread parallelization.  Two solutions:
    #     1. Pre-process the videos into .npy files using GNU parallel
    #     2. Use Tensorflow's tfio experimental decode_video()

    # STEP 1
    filepaths = []
    for root, dirs, files in os.walk(os.environ['MINERL_DATA_ROOT']):
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                filepaths.append(os.path.join(root, file))
    dataset = tf.data.Dataset.list_files(filepaths)
    # STEP 2
    dataset = dataset.map(lambda x: tf.py_function(lambda tf_string: get_frames(tf_string.numpy().decode('utf-8')), inp=[x], Tout=tf.uint8))
    # STEP 3
    dataset = dataset.unbatch()
    return dataset