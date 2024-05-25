import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLOWorld

model = YOLOWorld('/home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/models/yolov8s-world.pt')  # or choose yolov8m/l-world.pt


if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)

            for result in results:
                boxes = result.boxes
                print(boxes.cls)
                print(boxes.conf)
                for i, box in enumerate(boxes):
                    cls, conf = int(box.cls), float(box.conf)
                    # print(box.xyxy.cpu().numpy())
                    [[x1, y1, x2, y2]] = box.xyxy.cpu().numpy()
                    # print(cls, conf)
                    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.circle(color_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 2, (0, 255, 0), 2)
                    label = result.names[cls]
                    text = str(label) + "{:.2f}".format(conf)
                    cv2.putText(
                        color_image,
                        text,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 200, 200),
                        1
                    )

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
