import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, image_prefix="frame_", image_format=".png"): # <--- Changed here!
    """
    Extracts every frame from a video and saves them as images.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    else:
        print(f"Output directory '{output_folder}' already exists. Frames will be added/overwritten.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

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
    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Construct the output filename
            frame_filename = os.path.join(output_folder, f"{image_prefix}{frame_count:05d}{image_format}")

            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"\nExtraction complete. Saved {frame_count} frames to '{output_folder}'.")
    if frame_count == 0 and total_frames > 0:
        print("Warning: No frames were extracted. The video might be corrupted or empty.")

if __name__ == "__main__":
    video_file = 'sample.mp4'
    output_dir = 'images/'

    # Call the function, explicitly setting image_format to '.png'
    extract_frames(video_file, output_dir, image_format=".png")