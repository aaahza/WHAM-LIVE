import os
import argparse
import os.path as osp
from collections import defaultdict
import time

import cv2
import torch
import joblib
import numpy as np
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates!')
    _run_global = False

def run_wham_webcam(cfg,
                   output_pth,
                   network,
                   calib=None,
                   run_global=False,
                   fps_target=30):
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    assert cap.isOpened(), 'Failed to open webcam'
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f'Webcam initialized: {width}x{height} at {fps} FPS')
    
    # Create output window
    cv2.namedWindow('WHAM Webcam Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('WHAM Webcam Demo', width*2, height)
    
    # Initialize processing models
    detector = DetectionModel(cfg.DEVICE.lower())
    extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    
    # Buffer for storing frames and results
    frame_buffer = []
    tracking_results = defaultdict(dict)
    slam_results = None
    
    # Frame counter and timing
    frame_count = 0
    buffer_size = 30  # Process in chunks of frames
    
    # If using global coordinates, initialize SLAM
    if run_global and _run_global:
        # For SLAM, we need to create a temporary video file
        temp_video_path = osp.join(output_pth, 'temp_stream.mp4')
        temp_video = cv2.VideoWriter(
            temp_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        slam = SLAMModel(temp_video_path, output_pth, width, height, calib)
    else:
        slam = None
        temp_video = None
    
    logger.info('Starting webcam processing loop')
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break
            
            # Store frame in buffer
            frame_buffer.append(frame.copy())
            
            # Run detection and tracking
            person_detections = detector.track(frame, fps, buffer_size)
            
            # If using SLAM, write frame to temporary video
            if temp_video is not None:
                temp_video.write(frame)
                slam.track()
            
            # Process when buffer is full
            if len(frame_buffer) >= buffer_size:
                logger.info(f"Processing buffer of {len(frame_buffer)} frames")
                
                # Process tracking results
                tracking_results_chunk = detector.process(fps)
                
                # Create a temporary video file for feature extraction
                temp_feat_video_path = osp.join(output_pth, 'temp_feat_stream.mp4')
                temp_feat_video = cv2.VideoWriter(
                    temp_feat_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
                
                # Write frames to temporary video
                for frame in frame_buffer:
                    temp_feat_video.write(frame)
                temp_feat_video.release()
                
                # Extract features using the original method
                tracking_results_chunk = extractor.run(temp_feat_video_path, tracking_results_chunk)
                
                # Remove temporary feature video
                if osp.exists(temp_feat_video_path):
                    os.remove(temp_feat_video_path)
                
                # If using SLAM, process SLAM results
                if slam is not None:
                    slam_results_chunk = slam.process()
                else:
                    slam_results_chunk = np.zeros((len(frame_buffer), 7))
                    slam_results_chunk[:, 3] = 1.0  # Unit quaternion
                
                # Process the chunk through WHAM
                results = process_wham_chunk(
                    cfg, 
                    network, 
                    tracking_results_chunk, 
                    slam_results_chunk, 
                    width, 
                    height, 
                    fps
                )
                
                # Visualize the results
                visualize_results(frame_buffer, results, network.smpl, run_global)
                
                # Clear buffer for next batch
                frame_buffer = []
            
            # Display the current frame (without WHAM processing)
            # This provides immediate feedback while the buffer fills
            display_frame = np.zeros((height, width*2, 3), dtype=np.uint8)
            display_frame[:, :width] = frame
            display_frame[:, width:] = frame  # Placeholder until processing is complete
            
            # Add text indicating processing status
            cv2.putText(
                display_frame, 
                f"Buffering: {len(frame_buffer)}/{buffer_size}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display the frame
            cv2.imshow('WHAM Webcam Demo', display_frame)
            
            # Control frame rate
            elapsed = time.time() - start_time
            wait_time = max(1, int((1.0/fps_target - elapsed) * 1000))
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in webcam processing: {str(e)}")
    finally:
        # Clean up
        if temp_video is not None:
            temp_video.release()
            if osp.exists(temp_video_path):
                os.remove(temp_video_path)
                
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam session ended")

def process_wham_chunk(cfg, network, tracking_results, slam_results, width, height, fps):
    """Process a chunk of frames through WHAM."""
    # Build dataset for the chunk
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    results = defaultdict(dict)
    n_subjs = len(dataset)
    
    # Process each subject in the dataset
    for subj in range(n_subjs):
        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # Extract results
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
    
    return results

# First, modify the visualize_results function to fix the Renderer initialization
def visualize_results(frame_buffer, results, smpl, vis_global=False):
    """Visualize WHAM results alongside original frames."""
    try:
        from lib.vis.renderer import Renderer
        
        # Initialize renderer - Fix the initialization parameters
        img_width = frame_buffer[0].shape[1]
        img_height = frame_buffer[0].shape[0]
        focal_length = (img_width ** 2 + img_height ** 2) ** 0.5
        renderer = Renderer(width=img_width, height=img_height, focal_length=focal_length, device=cfg.DEVICE, faces=smpl.faces)
        
        for i, frame in enumerate(frame_buffer):
            # Find which subject and frame this corresponds to
            rendered = False
            for subj_id, subj_data in results.items():
                frame_indices = subj_data['frame_ids']
                if i in frame_indices:
                    idx = np.where(frame_indices == i)[0][0]
                    
                    # Get the data for rendering
                    vertices = subj_data['verts'][idx]
                    
                    # Render the mesh
                    rendered_img = renderer.render(
                        vertices,
                        smpl.faces,
                        frame.copy(),
                        mesh_color=[0.8, 0.3, 0.3]
                    )
                    
                    # Create side-by-side display
                    display_frame = np.zeros((frame.shape[0], frame.shape[1]*2, 3), dtype=np.uint8)
                    display_frame[:, :frame.shape[1]] = frame
                    display_frame[:, frame.shape[1]:] = rendered_img
                    
                    # Display the frame
                    cv2.imshow('WHAM Webcam Demo', display_frame)
                    cv2.waitKey(1)
                    rendered = True
                    break
            
            # If no corresponding result was found, just show the original frame
            if not rendered:
                display_frame = np.zeros((frame.shape[0], frame.shape[1]*2, 3), dtype=np.uint8)
                display_frame[:, :frame.shape[1]] = frame
                display_frame[:, frame.shape[1]:] = frame
                cv2.imshow('WHAM Webcam Demo', display_frame)
                cv2.waitKey(1)
    
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        # Fallback to simple display
        for frame in frame_buffer:
            display_frame = np.zeros((frame.shape[0], frame.shape[1]*2, 3), dtype=np.uint8)
            display_frame[:, :frame.shape[1]] = frame
            display_frame[:, frame.shape[1]:] = frame
            cv2.imshow('WHAM Webcam Demo', display_frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_pth', type=str, default='output/webcam', 
                        help='output folder to write temporary results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--fps_target', type=int, default=30,
                        help='Target frames per second for processing')
    
    parser.add_argument('--buffer_size', type=int, default=30,
                        help='Number of frames to buffer before processing')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # Create output directory
    output_pth = args.output_pth
    os.makedirs(output_pth, exist_ok=True)
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Run webcam demo
    run_wham_webcam(
        cfg, 
        output_pth, 
        network,
        args.calib, 
        run_global=not args.estimate_local_only,
        fps_target=args.fps_target
    )
    
    logger.info('Done!')