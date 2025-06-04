import os
from ultralytics import YOLO

def perform_inference(model_path, source_path, output_dir='runs/predict', imgsz=640, conf_threshold=0.25, iou_threshold=0.7, show_results=False, save_results=True, save_txt_labels=False):
    """
    Loads a trained YOLOv8 model and performs inference on images/videos.

    Args:
        model_path (str): Path to your trained model weights (e.g., 'my_trained_model/weights/best.pt').
        source_path (str): Path to the image, folder of images, video, or webcam ID (0).
        output_dir (str): Directory where prediction results will be saved.
        imgsz (int): Image size to resize input images to for inference.
        conf_threshold (float): Confidence threshold for object detection.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
        show_results (bool): Whether to display the results in a pop-up window (useful for live webcam).
        save_results (bool): Whether to save the annotated images/videos to disk.
        save_txt_labels (bool): Whether to save detected bounding box coordinates as .txt files.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"\n--- Starting Inference ---")
    print(f"Loading model from: {model_path}")
    print(f"Source for inference: {source_path}")

    # Load the trained YOLO model
    model = YOLO(model_path)

    # Perform inference
    # The 'predict' method returns a list of Results objects
    results = model.predict(
        source=source_path,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        show=show_results,      # Show results in a window (useful for webcam or live video)
        save=save_results,      # Save results to 'runs/predict/expX' (will be a video file)
        save_txt=save_txt_labels, # Save labels in YOLO format
        save_conf=save_txt_labels, # Include confidence in saved labels
        # save_crop=True, # Uncomment to save cropped detected objects
        name='my_video_predictions', # Name for this prediction run (changed for clarity)
        project='runs/predict', # Root directory for prediction runs
        exist_ok=True, # Allow overwriting if directory exists
        device=0 # Use GPU (0 for the first GPU). Use 'cpu' for CPU.
    )

    # You can also iterate through the results to access detected objects programmatically
    # For video, 'results' will contain one 'Result' object for each frame.
    print("\n--- Detected Objects Summary (for video, this will log per frame) ---")
    frame_count = 0
    for r in results: # 'r' represents a Result object for a single frame
        # r.path: The path to the source image/video (will be the video path for all frames)
        # r.boxes: Bounding boxes for this specific frame
        # r.masks: Segmentation masks (if applicable)
        # r.probs: Classification probabilities (if applicable)

        # print(f"Processing frame {frame_count} from {r.path}") # Uncomment for verbose frame-by-frame logging
        if r.boxes:
            # print(f"  Detected {len(r.boxes)} objects in frame {frame_count}.")
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # xyxy = box.xyxy[0].tolist() # Bounding box in [x1, y1, x2, y2] format
                # print(f"  - Class ID: {class_id}, Confidence: {confidence:.2f}, Box: {xyxy}")
                # class_name = model.names[class_id] # Requires loading model.names
                # print(f"  - Frame {frame_count}: Class: {class_name}, Conf: {confidence:.2f}")
                pass # You can add custom logic here, e.g., count objects over time
        # else:
            # print(f"  No objects detected in frame {frame_count}.")

        frame_count += 1
    print(f"Processed a total of {frame_count} frames from the video.")


    print(f"\nInference complete. Annotated video saved to: {output_dir}/my_video_predictions/")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your trained model weights
    TRAINED_MODEL_PATH = 'train4_results/content/runs/detect/train4/weights/best.pt'

    # --- Choose your inference source ---
    # OPTION 1: Single image (commented out)
    # INFERENCE_SOURCE = 'my_inference_images/new_image_1.jpg'

    # OPTION 2: Folder of images (commented out)
    # INFERENCE_SOURCE = 'my_inference_images/'

    # OPTION 3: Video file (USE THIS ONE!)
    INFERENCE_SOURCE = 'sample.mp4' # <--- CHANGE THIS TO YOUR VIDEO PATH

    # OPTION 4: Webcam (uncomment and set show_results=True for live view)
    # INFERENCE_SOURCE = 0 # 0 for default webcam, 1 for second webcam, etc.

    # Other inference parameters
    IMG_SIZE = 640       # Keep this consistent with your training image size
    CONF_THRESHOLD = 0.5 # Lower this to detect more objects (but potentially more false positives)
    IOU_THRESHOLD = 0.7  # Adjust for Non-Maximum Suppression (lower for more unique boxes)
    SHOW_RESULTS_POPUP = True # Set to True to see results in a pop-up window as it processes.
                             # For long videos, this might be slow, so you might prefer False.
    SAVE_RESULTS_TO_DISK = True # Absolutely keep this True to save the output video.
    SAVE_TXT_LABELS = False # Saves bounding box coordinates in .txt format per frame (can generate many files!)

    # Perform the inference
    perform_inference(
        model_path=TRAINED_MODEL_PATH,
        source_path=INFERENCE_SOURCE,
        imgsz=IMG_SIZE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        show_results=SHOW_RESULTS_POPUP,
        save_results=SAVE_RESULTS_TO_DISK,
        save_txt_labels=SAVE_TXT_LABELS
    )