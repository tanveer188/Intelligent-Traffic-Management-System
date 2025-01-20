import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO  # Updated import for YOLO from ultralytics

def process_video_and_analyze(video_files):
    model = YOLO('yolov8n.pt')  # Load the YOLO model from ultralytics

    # Create a VideoCapture object for each video
    caps = [cv2.VideoCapture(video_path) for video_path in video_files]
    
    # Get video properties from the first video (assuming all videos have the same dimensions)
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    # Create a window to display the split screen
    cv2.namedWindow('Video Output', cv2.WINDOW_NORMAL)

    # Set a higher frame rate for output
    output_fps = min(fps * 2, 60)  # Double the original fps, capped at 60

    try:
        while True:
            frames = []
            total_vehicles = 0

            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frames.append(np.zeros((height, width, 3), dtype=np.uint8))  # Add a black frame if video ends
                else:
                    # Perform detection and tracking
                    results = model.track(frame, persist=True)
                    # Filter results to only include vehicles
                    vehicle_results = [result for result in results[0].boxes if result.cls in [2, 3, 5, 7]]  # Assuming class IDs for vehicles
                    total_vehicles = len(vehicle_results)

                    # Visualize the results
                    annotated_frame = frame.copy()
                    for vehicle in vehicle_results:
                        x1, y1, x2, y2 = map(int, vehicle.xyxy[0])  # Get bounding box coordinates
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                    # Determine traffic light color based on vehicle count
                    if total_vehicles > 10:  # Example threshold for high vehicle count
                        traffic_light_color = 'green'
                    else:
                        traffic_light_color = 'red'

                    # Display vehicle count and traffic light color on the frame
                    cv2.putText(annotated_frame, f'Vehicles: {total_vehicles}', (width - 400, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    circle_color = (0, 0, 255) if traffic_light_color == 'red' else (0, 255, 0)  # Red for 'red', Green for 'green'
                    cv2.circle(annotated_frame, (50, 50), 20, (0, 0, 0), -1)  # Draw a filled black circle as the border
                    cv2.circle(annotated_frame, (50, 50), 18, circle_color, -1)  # Draw a filled circle at the top left

                    frames.append(annotated_frame)

            # Create a split screen by stacking frames vertically
            if frames:
                # Resize frames to fit in a 2x2 grid
                resized_frames = [cv2.resize(frame, (width // 2, height // 2)) for frame in frames]
                split_frame = np.vstack((np.hstack(resized_frames[:2]), np.hstack(resized_frames[2:])))
                cv2.imshow('Video Output', split_frame)

            # Adjust wait time based on output fps
            if cv2.waitKey(int(1000 / output_fps)) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred while processing videos: {e}")
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


# Example usage
def main():
    video_folder = 'videos'
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    # Process video and analyze vehicles
    process_video_and_analyze(video_files)


if __name__ == '__main__':
    main() 
