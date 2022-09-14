import cv2, os
import numpy as np
import numpy.typing as npt

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

def retrieve_minerl_frames() -> npt.NDArray:
    """ 
    """