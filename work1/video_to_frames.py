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

# file_dir = "F:/video_action_recognition/video_process_files/VID_20191221_110038.mp4"
file_dir = "F:/video_action_recognition/video_process_files/VID_20191221_103534.mp4"
with imageio.get_reader(file_dir,  'ffmpeg') as vid:
    nframes = vid.get_meta_data()['nframes']

    for i, frame in enumerate(vid):

        # n_frames = i
        while (i%30==0):

          frame = cv2.resize(frame, (Width, Height), interpolation = cv2.INTER_CUBIC)
          # box=(320,320,640,600)
          # crop=frame[320:320+280,320:320+320]
          # crop=frame[265:265+299,656:656+299]
          # crop=frame[265:265+299,1017:1017+299]
          crop = frame[328:328 + 299, 326:326 + 299]
          # imageio.imwrite("F:/video_action_recognition/frame_output/v4mouse1"+'/frame_%d.jpg' %i, crop)
          # imageio.imwrite("F:/video_action_recognition/frame_output/v4mouse2"+'/frame_%d.jpg' %i, crop)
          # imageio.imwrite("F:/video_action_recognition/frame_output/v4-1"+'/frame_%d.jpg' %i, frame)
          imageio.imwrite("F:/video_action_recognition/frame_output/v4mouse3"+'/frame_%d.jpg' %i, crop)
          break
    # np.save('nframes.npy', n_frames)
