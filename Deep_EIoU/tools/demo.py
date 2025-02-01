import argparse
import os
import os.path as osp
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.basetrack import TrackState
from tracker.Deep_EIoU import Deep_EIoU
from tracker.tracking_utils.timer import Timer
from reid.torchreid.utils import FeatureExtractor

def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo!")
    parser.add_argument("--path", default="./videos/demo.mp4", help="Path to video file")
    parser.add_argument("--yolo_model", default="checkpoints/person_model.pt", help="Path to YOLOv8 model")
    parser.add_argument("--save_result", action="store_true", help="Save results")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--device", default="gpu", help="Device: cpu or gpu")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    # Tracking parameters
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="Tracking high confidence threshold")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="Tracking low confidence threshold")
    parser.add_argument("--new_track_thresh", type=float, default=0.7, help="New track threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="Frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold")
    parser.add_argument("--min_box_area", type=float, default=10, help="Min box area to track")
    
    # ReID parameters
    parser.add_argument("--with-reid", action="store_true", default=True, help="Use ReID")
    parser.add_argument("--proximity_thresh", type=float, default=0.5, help="Proximity threshold")
    parser.add_argument("--appearance_thresh", type=float, default=0.25, help="Appearance threshold")
    return parser

def debug_detections(dets, frame_id):
    """Debug YOLOv8 detections"""
    logger.info(f"\n[Frame {frame_id}] YOLOv8 Detections:")
    logger.info(f"Number of detections: {len(dets)}")
    if len(dets) > 0:
        logger.info(f"Detection format: [x1, y1, x2, y2, conf]")
        for i, det in enumerate(dets):
            logger.info(f"Detection {i+1}: box={det[:4].round(2)}, conf={det[4]:.3f}")
def get_color(idx):
    """Get unique color for visualization"""
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=None):
    """Plot tracking boxes with proper scaling"""
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    
    # Scale text and line size based on image dimensions
    text_scale = max(1, min(im_w, im_h) / 1000.)
    text_thickness = max(1, int(min(im_w, im_h) / 500.))
    line_thickness = max(1, int(min(im_w, im_h) / 300.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = [float(x) for x in tlwh]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(im_w-1, x1))
        y1 = max(0, min(im_h-1, y1))
        x2 = max(0, min(im_w-1, x1 + w))
        y2 = max(0, min(im_h-1, y1 + h))
        
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        obj_id = int(obj_ids[i])
        
        # Get unique color for ID
        color = get_color(abs(obj_id))
        
        # Draw box
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        
        # Draw label with score if available
        label = f'ID:{obj_id}'
        if scores is not None:
            label += f' {scores[i]:.2f}'
            
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
        txt_bk_color = tuple([min(c + 100, 255) for c in color])
        
        # Draw label background
        cv2.rectangle(im, (intbox[0], intbox[1] - label_size[1] - 5),
                     (intbox[0] + label_size[0], intbox[1]), txt_bk_color, -1)
        
        # Draw label text
        cv2.putText(im, label, (intbox[0], intbox[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 
                    thickness=text_thickness)

    # Draw frame counter and FPS if available
    status_text = f'Frame: {frame_id}'
    if fps is not None:
        status_text += f' FPS: {fps:.1f}'
    
    cv2.putText(im, status_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                text_scale * 2, (0, 255, 0), thickness=text_thickness * 2)

    return im

def debug_embeddings(embs, frame_id):
    """Debug ReID embeddings"""
    logger.info(f"\n[Frame {frame_id}] ReID Features:")
    if embs is not None:
        logger.info(f"Number of embeddings: {len(embs)}")
        logger.info(f"Embedding dimension: {embs.shape[1]}")
        logger.info(f"Embedding norms: {np.linalg.norm(embs, axis=1).round(3)}")
        if len(embs) > 1:
            similarity = embs @ embs.T
            logger.info(f"Pairwise similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")

def debug_tracking(online_targets, frame_id):
    """Debug tracking results"""
    logger.info(f"\n[Frame {frame_id}] Tracking Results:")
    logger.info(f"Number of tracked objects: {len(online_targets)}")
    for t in online_targets:
        logger.info(f"Track ID {t.track_id}: state={t.state}, age={frame_id - t.start_frame}")

def process_yolo_detections(results, frame, width, height, min_score=0.1, debug=False):
    """Process YOLOv8 detections with proper scaling"""
    dets = []
    crops = []
    
    if debug:
        logger.info(f"\nProcessing YOLOv8 detections:")
        logger.info(f"Original image size: {width}x{height}")
        logger.info(f"YOLOv8 size: {results.orig_shape}")
        
    # Calculate scale from YOLOv8 inference size to original image size
    yolo_height, yolo_width = results.orig_shape
    w_scale = width / yolo_width
    h_scale = height / yolo_height
    
    if debug:
        logger.info(f"Width scale: {w_scale:.3f}, Height scale: {h_scale:.3f}")
    
    for box in results.boxes:
        if int(box.cls[0]) == 2:  # person class
            conf = float(box.conf[0])
            if conf < min_score:
                continue
                
            # Get coordinates in YOLOv8 input size
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Scale to original image size
            x1, x2 = x1 * w_scale, x2 * w_scale
            y1, y2 = y1 * h_scale, y2 * h_scale
            
            if debug:
                logger.info(f"Detection - Original box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], conf: {conf:.3f}")
            
            # Skip detections at edges
            if x1 < 1 or y1 < 1 or x2 > width-1 or y2 > height-1:
                if debug:
                    logger.info("Skipping edge detection")
                continue
            
            # Add detection
            dets.append(np.array([x1, y1, x2, y2, conf]))
            
            # Get crop for ReID
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                if debug:
                    logger.info(f"Extracted crop shape: {crop.shape}")
    
    if not dets:
        return np.empty((0, 5)), []
        
    dets = np.array(dets)
    
    if debug:
        logger.info(f"Final valid person detections: {len(dets)}")
        
    return dets, crops

def imageflow_demo(model, extractor, args):

    cap = cv2.VideoCapture(args.path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if args.debug:
        logger.info(f"\nVideo properties:")
        logger.info(f"Size: {width}x{height}")
        logger.info(f"FPS: {fps}")
    
    # Setup output

    save_folder = osp.join(args.output_dir, 'seq1')
    os.makedirs(save_folder, exist_ok=True)
    
    if args.save_result:
        save_path = osp.join(save_folder, osp.basename(args.path))
        logger.info(f"Video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
    
    tracker = Deep_EIoU(args, frame_rate=fps)
    timer = Timer()
    frame_id = 0
    results = []
    
    while True:
        if frame_id % 20 == 0:
            logger.info(f'Processing frame {frame_id} ({1. / max(1e-5, timer.average_time):.2f} fps)')
        
        ret_val, frame = cap.read()
        if not ret_val:
            break
            
        timer.tic()
        
        if args.debug:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing frame {frame_id}")
            
       
        yolo_results = model(frame)[0]
        
        
        det, crops = process_yolo_detections(
            yolo_results, frame, width, height, 
            min_score=args.track_low_thresh,
            debug=args.debug
        )
        
        if args.debug:
            logger.info("\nRunning YOLOv8 detection:")
        
        yolo_results = model(frame)[0]
        
        if args.debug:
            logger.info("YOLOv8 model classes:")
            for cls_id, cls_name in yolo_results.names.items():
                logger.info(f"  {cls_id}: {cls_name}")
            logger.info("\nDetections by class:")
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                logger.info(f"  {yolo_results.names[cls_id]}: {conf:.3f}")
        
        
        det, crops = process_yolo_detections(
            yolo_results, frame, width, height,
            min_score=args.track_low_thresh,
            debug=args.debug
        )
        
        if len(det) > 0:
            if crops:
                
                embs = extractor(crops).cpu().numpy()
                
                if args.debug:
                    debug_embeddings(embs, frame_id)
                
                
                online_targets = tracker.update(det, embs)
                
                if args.debug:
                    debug_tracking(online_targets, frame_id)
                
               
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for t in online_targets:
                    tlwh = t.last_tlwh
                    tid = t.track_id
                    
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                            f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                
                timer.toc()
                
                # Visualize
                online_im = plot_tracking(
                    frame, 
                    online_tlwhs, 
                    online_ids, 
                    scores=online_scores,
                    frame_id=frame_id,
                    fps=1. / timer.average_time if timer.average_time > 0 else 0
                )
            else:
                timer.toc()
                online_im = frame
        else:
            timer.toc()
            online_im = frame
            
        if args.save_result:
            try:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                
                if 'vid_writer' not in locals():
                    save_path = osp.join(save_folder, osp.basename(args.path))
                    logger.info(f"Creating video writer: {save_path}")
                    vid_writer = cv2.VideoWriter(
                        save_path, 
                        cv2.VideoWriter_fourcc(*"mp4v"), 
                        fps, 
                        (width, height)
                    )
                    if not vid_writer.isOpened():
                        logger.error("Failed to create video writer")
                        args.save_result = False
                
                # Save frame
                vid_writer.write(online_im)
            except Exception as e:
                logger.error(f"Error saving video frame: {e}")
                args.save_result = False 
            
        frame_id += 1
    
    cap.release()
    if args.save_result:
        vid_writer.release()
    
   
    if results:
        res_file = osp.join(save_folder, "seq1.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"Save results to {res_file}")

def get_device():
    """Get available device with priority: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')



def deep_eiou_main(video_path):
    class Args:
        def __init__(self):
            self.path = video_path
            self.yolo_model = "Deep_EIoU/checkpoints/person_model.pt"
            self.save_result = True 
            self.output_dir = "results"
            self.debug = False
            self.track_high_thresh = 0.6
            self.track_low_thresh = 0.1
            self.new_track_thresh = 0.7
            self.track_buffer = 60
            self.match_thresh = 0.8
            self.min_box_area = 10
            self.with_reid = True
            self.proximity_thresh = 0.5
            self.appearance_thresh = 0.25

    args = Args()
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model = YOLO(args.yolo_model)
    model.to(device)
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path="Deep_EIoU/checkpoints/sports_model.pth.tar-60",
        device=str(device)
    )
    
    imageflow_demo(model, extractor, args)
    return  args.output_dir

if __name__ == "__main__":
    deep_eiou_main()