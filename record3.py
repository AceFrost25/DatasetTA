"""
record2.py
---------------

Main Function for recording a video sequence into cad (color-aligned-to-depth) 
images and depth images

Using librealsense SDK 2.0 with pyrealsense2 for SR300 and D series cameras
"""

# record for 40s after a 5s count down
# or exit the recording earlier by pressing q

RECORD_LENGTH = 10

import png
import pyrealsense2 as rs
import json
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import time
import os
import sys

def make_directories(folder):
    if not os.path.exists(folder + "JPEGImages/"):
        os.makedirs(folder + "JPEGImages/")
    if not os.path.exists(folder + "depth/"):
        os.makedirs(folder + "depth/")

def print_usage():
    print("Usage: record2.py <foldername>")
    print("foldername: path where the recorded data should be stored at")
    print("e.g., record2.py LINEMOD/mug")     

if __name__ == "__main__":
    try:
        folder = sys.argv[1] + "/"
    except:
        print_usage()
        exit()

    FileName = 0
    make_directories(folder)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Infrared stream configuration
    
    # Start pipeline
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    infrared_frame = frames.get_infrared_frame()

    # IR Intrinsics
    intr = infrared_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
    }

    with open(folder + 'intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)

    align_to = rs.stream.infrared
    align = rs.align(align_to)

    T_start = time.time()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        infrared_frame = aligned_frames.get_infrared_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not infrared_frame:
            continue

        d = np.asanyarray(aligned_depth_frame.get_data())
        c = np.asanyarray(infrared_frame.get_data())
   
        # Visualize count down
        if time.time() - T_start > 5:
            filecad = folder + "JPEGImages/%s.jpg" % FileName
            filedepth = folder + "depth/%s.png" % FileName
            cv2.imwrite(filecad, c)
            with open(filedepth, 'wb') as f:
                writer = png.Writer(width=d.shape[1], height=d.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = d.tolist()
                writer.write(f, zgray2list)

            FileName += 1
        if time.time() - T_start > RECORD_LENGTH + 5:
            pipeline.stop()
            break

        if time.time() - T_start < 5:
            cv2.putText(c, str(5 - int(time.time() - T_start)), (240, 320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        if time.time() - T_start > RECORD_LENGTH:
            cv2.putText(c, str(RECORD_LENGTH + 5 - int(time.time() - T_start)), (240, 320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('INFRARED IMAGE', c)

        # press q to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.stop()
            break

    # Release everything if job is finished
    cv2.destroyAllWindows()

