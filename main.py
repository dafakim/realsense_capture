
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

savepath='C:/Users/CEO/Desktop/'

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop

while True:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    depth_image_3d = np.dstack(
        (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    realdepth=np.dot(depth_image,depth_scale)
    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap))
    cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Align Example', images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

ct=datetime.today().strftime("%Y%m%d%H%M%S")
color_image_name=savepath+'/'+ct+'_color.jpg'
depth_image_name=savepath+'/'+ct+'_depth.jpg'
depth_csv_name=savepath+'/'+ct+'_depth.csv'
depth_raw_name=savepath+'/'+ct+'_depth.raw'

cv2.imwrite(color_image_name, color_image)
cv2.imwrite(depth_image_name, depth_colormap)
np.savetxt(depth_csv_name,realdepth, delimiter=',')
np.savetxt(depth_raw_name,realdepth, delimiter=',')
