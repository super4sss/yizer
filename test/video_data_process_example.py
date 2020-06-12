from __future__ import print_function
import sys
import numpy as np
import os
import imageio
import cv2
import imageio_ffmpeg
import moviepy
Height = 1080
Width = 1920

file_dir = "F:/video_process_files/VID_20191221_103534.mp4"
with imageio.get_reader(file_dir,  'ffmpeg') as vid:
    nframes = vid.get_meta_data()['nframes']
    for i, frame in enumerate(vid):
        n_frames = i
        frame = cv2.resize(frame, (Width, Height), interpolation = cv2.INTER_CUBIC)
        imageio.imwrite("F:/output"+'/frame_%d.jpg' %i, frame)
    np.save('nframes.npy', n_frames)
