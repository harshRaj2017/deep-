import cv2
import numpy as np
from collections import defaultdict
import os
import argparse

def get_color(idx):
    """Get unique color for visualization"""
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, frame_id):
    """Plot tracking boxes"""
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = [int(float(x)) for x in tlwh]
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        color = get_color(abs(obj_id))
        
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        label = f'ID: {obj_id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, text_scale, text_thickness)[0]
        cv2.putText(im, label, (intbox[0], intbox[1] - t_size[1]), 
                    cv2.FONT_HERSHEY_PLAIN, text_scale, color, 
                    thickness=text_thickness)

    # Draw frame id
    cv2.putText(im, f'Frame: {frame_id}', (0, int(30 * text_scale)), 
                cv2.FONT_HERSHEY_PLAIN, text_scale*2, (0, 255, 0), 
                thickness=text_thickness*2)
    
    return im

def read_results(filename):
    """Read MOT format tracking results file"""
    results = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '': continue
            fields = line.split(',')
            frame_id = int(fields[0])
            track_id = int(fields[1])
            x, y, w, h = map(float, fields[2:6])
            results[frame_id].append((track_id, x, y, w, h))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='path to input video')
    parser.add_argument('--txt', required=True, help='path to tracking results txt file')
    parser.add_argument('--output', default='output.mp4', help='path to output video')
    parser.add_argument('--fps', type=int, default=30, help='output video FPS')
    args = parser.parse_args()

    print(f"Reading tracking results from: {args.txt}")
    tracking_results = read_results(args.txt)

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Creating output video: {args.output}")
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (width, height)
    )

    frame_id = 1  # MOT format usually starts from frame 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw tracking results if available for this frame
        if frame_id in tracking_results:
            # Get bounding boxes and IDs
            results = tracking_results[frame_id]
            tlwhs = []
            track_ids = []
            
            for track_id, x, y, w, h in results:
                tlwhs.append([x, y, w, h])
                track_ids.append(track_id)
            
            # Draw boxes
            frame = plot_tracking(frame, tlwhs, track_ids, frame_id)

        # Write frame
        out.write(frame)
        
        # Print progress
        if frame_id % 100 == 0:
            print(f"Processing frame {frame_id}/{total_frames}")
            
        frame_id += 1

    # Clean up
    cap.release()
    out.release()
    print(f"Done! Output saved to: {args.output}")

if __name__ == "__main__":
    main()