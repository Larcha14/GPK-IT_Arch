import cv2
import pandas as pd
import numpy as np
from bagpy import bagreader


class VideoData:
    """Video information storage"""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.frames = []


def get_video_size(img_data: pd.DataFrame) -> tuple:
    """Extracts width and height of the video frames"""
    return img_data['width'][0], img_data['height'][0]


def get_rgb_vd(rgb_data: pd.DataFrame) -> VideoData:
    """Decodes RGB video data"""
    rgb_video = VideoData()
    w, h = get_video_size(rgb_data)
    for img_data in rgb_data['data']:
        byte_string = img_data[2:-1].encode('latin1')
        escaped_string = byte_string.decode('unicode_escape')
        byte_string = escaped_string.encode('latin1')
        rgb_array = np.frombuffer(byte_string, np.uint8)
        rgb_frame = rgb_array.reshape((h, w, -1))
        bgr_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)  # GBR
        rgb_video.frames.append(bgr_image)
    rgb_video.width, rgb_video.height = w, h
    print(f'Extracted RGB video. Size = {w}x{h}. Total frames count = {len(rgb_video.frames)}')
    return rgb_video


def get_depth_vd(depth_data: pd.DataFrame) -> VideoData:
    """Decodes DEPTH video data"""
    depth_video = VideoData()
    w, h = get_video_size(depth_data)
    for img_data in depth_data['data']:
        byte_string = img_data[2:-1].encode('latin1')
        escaped_string = byte_string.decode('unicode_escape')
        byte_string = escaped_string.encode('latin1')
        depth_array = np.ndarray((h, w), dtype=np.uint16, buffer=byte_string)
        depth_transformed = cv2.convertScaleAbs(depth_array, alpha=255 / 65535)
        depth_colored = cv2.applyColorMap(depth_transformed, cv2.COLORMAP_TURBO)
        depth_video.frames.append(depth_colored)
    depth_video.width, depth_video.height = w, h
    print(f'Extracted DEPTH video. Size = {w}x{h}. Total frames count = {len(depth_video.frames)}')
    return depth_video


def save_video(filename: str, video: VideoData, fps: int = 15, codec: str = 'mp4v', colored: bool = True) \
        -> None:
    """Saves video file"""
    print(f'Opening OpenCV video writer. CODEC={codec}, FILENAME={filename}, FPS={fps}')
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec),
                             fps, (video.width, video.height), colored)
    for frame in video.frames:
        writer.write(frame)

    print(f'Closing OpenCV VideoWriter for {filename}.')
    writer.release()
    del writer


# Entry point here
if __name__ == '__main__':
    # Reading .bag file
    bagfile = bagreader('test.bag')

    # Extracting raw topics data
    raw_rgb = bagfile.message_by_topic('/device_0/sensor_1/Color_0/image/data')
    raw_depth = bagfile.message_by_topic('/device_0/sensor_0/Depth_0/image/data')

    # Converting raw data to Pandas DataFrames
    rgb_df = pd.read_csv(raw_rgb)
    depth_df = pd.read_csv(raw_depth)

    # Extracting video data from DataFrames
    rgb_vd = get_rgb_vd(rgb_df)
    depth_vd = get_depth_vd(depth_df)

    # Saving videos in .mp4 format
    save_video(filename='rgb_video.mp4', video=rgb_vd)
    save_video(filename='depth_video.mp4', video=depth_vd)
