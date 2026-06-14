"""
Extract frames from video for processing
"""

import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video at regular intervals
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every N frames (default: 30 = ~1 frame per second at 30fps)
    
    Returns:
        List of saved frame paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps")
    print(f"Extracting every {frame_interval} frames")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"✅ Extracted {saved_count} frames to {output_dir}")
    
    return frame_paths

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    
    # Calculate duration
    if info['fps'] > 0:
        info['duration_seconds'] = info['total_frames'] / info['fps']
        info['duration_minutes'] = info['duration_seconds'] / 60
    
    return info

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output_dir", default="./extracted_frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Extract every N frames")
    
    args = parser.parse_args()
    
    # Show video info
    info = get_video_info(args.video_path)
    print(f"\n📹 Video Information:")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   FPS: {info['fps']:.2f}")
    print(f"   Duration: {info['duration_minutes']:.2f} minutes")
    print(f"   Total frames: {info['total_frames']}\n")
    
    # Extract frames
    extract_frames(args.video_path, args.output_dir, args.interval)
