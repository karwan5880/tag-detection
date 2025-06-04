import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

def extract_annotated_frames_and_labels(
    model_path,
    video_path,
    output_images_dir,
    output_labels_dir,
    imgsz=640,
    conf_threshold=0.25,
    iou_threshold=0.7,
    image_prefix="frame_",
    label_prefix="frame_"
):
    """
    Extracts frames from a video that contain detected objects, saves them as PNG,
    and saves their corresponding YOLO format labels.

    Args:
        model_path (str): Path to your trained YOLO model weights (e.g., 'my_trained_model/weights/best.pt').
        video_path (str): Path to the input video file (e.g., 'my_inference_images/sample.mp4').
        output_images_dir (str): Directory where extracted images will be saved.
        output_labels_dir (str): Directory where YOLO label .txt files will be saved.
        imgsz (int): Image size to resize input images to for inference.
        conf_threshold (float): Confidence threshold for object detection.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
        image_prefix (str): Prefix for the output image filenames.
        label_prefix (str): Prefix for the output label filenames.
    """

    # --- 1. Validate Paths and Create Directories ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_images_dir}")
    print(f"Output labels will be saved to: {output_labels_dir}")

    # --- 2. Load the YOLO Model ---
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 3. Open the Video File ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n--- Video Information ---")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Estimated duration: {total_frames / fps:.2f} seconds")
    print(f"-------------------------")

    # --- 4. Process Video Frame by Frame ---
    frame_count = 0
    saved_count = 0

    print("\nStarting frame processing and inference...")
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            ret, frame = cap.read() # Read a frame

            if not ret: # End of video or error
                break

            # Perform inference on the current frame
            # We pass the frame (NumPy array) directly as source
            # verbose=False suppresses per-frame console output from predict
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False, # Suppress detailed predict output per frame
                device=0 # Ensure GPU is used if available
            )

            # Access the results for the current frame (there's only one result object as we passed one frame)
            r = results[0]

            # Check if any objects were detected in this frame
            if r.boxes: # r.boxes is not None and not empty
                saved_count += 1

                # Construct filenames
                frame_filename = os.path.join(output_images_dir, f"{image_prefix}{frame_count:05d}.png")
                label_filename = os.path.join(output_labels_dir, f"{label_prefix}{frame_count:05d}.txt")

                # # Save the frame as PNG
                # cv2.imwrite(frame_filename, frame)

                # Save the labels in YOLO format
                with open(label_filename, 'w') as f:
                    for box in r.boxes:
                        # box.xywhn gives normalized [x_center, y_center, width, height]
                        # box.cls gives class ID (as a tensor, so convert to int)
                        # box.conf gives confidence (as a tensor, so convert to float)
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        class_id = int(box.cls[0])
                        
                        # Write to file: class_id center_x center_y width height
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            frame_count += 1
            pbar.update(1) # Update the progress bar

    # --- 5. Cleanup and Summary ---
    cap.release()
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections saved: {saved_count}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Labels saved to: {output_labels_dir}")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your trained model weights (e.g., from your Colab training, then downloaded)
    # Make sure this path is correct on your local machine
    TRAINED_MODEL_PATH = 'train4_results/content/runs/detect/train4/weights/best.pt'

    # Path to your input video file
    INPUT_VIDEO_PATH = 'sample.mp4'

    # Output directories for images and labels
    OUTPUT_IMAGES_DIR = 'extracted_dataset/images'
    OUTPUT_LABELS_DIR = 'extracted_dataset/labels'

    # Inference parameters
    IMG_SIZE = 640       # Must be consistent with your training img_size
    CONF_THRESHOLD = 0.25 # Lower to capture more detections, even weaker ones
    IOU_THRESHOLD = 0.7   # Standard NMS threshold

    # Run the extraction
    extract_annotated_frames_and_labels(
        model_path=TRAINED_MODEL_PATH,
        video_path=INPUT_VIDEO_PATH,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR,
        imgsz=IMG_SIZE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )