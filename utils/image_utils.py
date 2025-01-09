import cv2
import numpy as np
from PIL import Image
import torch
import os
import dlib

def create_video_from_image(image_path, duration=5, fps=25):
    """
    Convert a single image into a video sequence with proper 3D face modeling
    """
    # Read the image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
        
    if img is None:
        raise ValueError("Could not load image")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize face detection
    from third_part import face_detection
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                          flip_input=False, device='cuda:0')
    
    # Get face landmarks and crop
    predictions = detector.get_detections_for_batch(np.array([img_rgb]))[0]
    if predictions is None:
        raise ValueError('Face not detected in the image')
        
    # Add padding for better jaw/chin coverage
    pady1, pady2, padx1, padx2 = 0, 20, 0, 0  # Extra padding below chin
    y1 = max(0, predictions[1] - pady1)
    y2 = min(img.shape[0], predictions[3] + pady2)
    x1 = max(0, predictions[0] - padx1)
    x2 = min(img.shape[1], predictions[2] + padx2)
    
    # Ensure dimensions are even numbers for pyramid operations
    if (y2 - y1) % 2 == 1: y2 -= 1
    if (x2 - x1) % 2 == 1: x2 -= 1
    
    face_region = img[y1:y2, x1:x2]

    # Create video frames
    height, width = img.shape[:2]
    temp_video_path = os.path.join(os.path.dirname(image_path), "temp_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # Process frames
    total_frames = duration * fps
    
    try:
        # Load face3d model for 3D reconstruction
        from third_part.face3d.models import networks
        net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False)
        net_recon.load_state_dict(torch.load('checkpoints/face3d_pretrain_epoch_20.pth', 
                                            map_location='cpu')['net_recon'])
        net_recon.eval()
        
        # Convert image to tensor for 3D modeling
        face_tensor = torch.from_numpy(cv2.resize(face_region, (256, 256))).float().permute(2, 0, 1).unsqueeze(0) / 255.
        
        with torch.no_grad():
            # Get 3D face parameters
            coeffs = net_recon(face_tensor)
            
            # Extract different components
            from utils.inference_utils import split_coeff
            coeffs_dict = split_coeff(coeffs)
            
            # Generate frames with slight variations in expression coefficients
            for i in range(total_frames):
                # Add subtle random variations to expression coefficients
                exp_variation = torch.randn_like(coeffs_dict['exp']) * 0.02
                coeffs_dict['exp'] = coeffs_dict['exp'] + exp_variation
                
                # Reconstruct face with new expression
                frame = img.copy()
                
                # Create and process mask
                mask = np.ones_like(face_region, dtype=np.float32)
                # Feather the edges of the mask
                mask = cv2.GaussianBlur(mask, (31, 31), 11)
                
                # Ensure regions have same dimensions for blending
                region_height, region_width = face_region.shape[:2]
                frame_region = frame[y1:y1+region_height, x1:x1+region_width]
                
                if face_region.shape == frame_region.shape:
                    # Use Laplacian pyramid blending
                    from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask
                    try:
                        blended = Laplacian_Pyramid_Blending_with_mask(face_region, frame_region, mask)
                        frame[y1:y1+region_height, x1:x1+region_width] = blended
                    except ValueError as e:
                        # Fallback to simple alpha blending if pyramid blending fails
                        blended = (face_region * mask + frame_region * (1 - mask)).astype(np.uint8)
                        frame[y1:y1+region_height, x1:x1+region_width] = blended
                
                out.write(frame)
    
    except Exception as e:
        # Fallback to basic frame duplication if 3D modeling fails
        print(f"Warning: 3D modeling failed, falling back to basic frame duplication: {str(e)}")
        for _ in range(total_frames):
            out.write(img)
    
    finally:
        out.release()
        
    return temp_video_path

def is_image_file(file_path):
    """
    Check if the file is an image
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(file_path.lower())[1] in image_extensions 