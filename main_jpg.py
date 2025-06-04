import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, image_prefix="frame_", image_format=".jpg"):
    """
    Extracts every frame from a video and saves them as images.

    Args:
        video_path (str): Path to the input video file (e.g., 'sample.mp4').
        output_folder (str): Directory where the extracted images will be saved (e.g., 'images/').
        image_prefix (str): Prefix for the output image filenames (e.g., 'frame_00001.jpg').
        image_format (str): Format of the output images (e.g., '.jpg', '.png').
                             Use '.png' for lossless quality, '.jpg' for smaller size.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Output directory '{output_folder}' already exists. Frames will be added/overwritten.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n--- Video Information ---")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Resolution: {video_width}x{video_height}")
    print(f"Estimated duration: {total_frames / fps:.2f} seconds")
    print(f"Saving frames to: {output_folder}")
    print(f"-------------------------")

    frame_count = 0
    # Use tqdm for a progress bar
    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # If ret is False, it means we have reached the end of the video or an error occurred
            if not ret:
                break

            # Construct the output filename (e.g., 'frame_00000.jpg', 'frame_00001.jpg')
            # The :05d ensures zero-padding to 5 digits (e.g., 1 becomes 00001)
            frame_filename = os.path.join(output_folder, f"{image_prefix}{frame_count:05d}{image_format}")

            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
            pbar.update(1) # Update the progress bar

    # Release the video capture object
    cap.release()

    print(f"\nExtraction complete. Saved {frame_count} frames to '{output_folder}'.")
    if frame_count == 0 and total_frames > 0:
        print("Warning: No frames were extracted. The video might be corrupted or empty.")

if __name__ == "__main__":
    # --- Configuration ---
    video_file = 'sample.mp4'  # Make sure this video file is in the same directory as the script,
                               # or provide its full path.
    output_dir = 'images/'     # This folder will be created if it doesn't exist.
    # ---------------------

    extract_frames(video_file, output_dir)