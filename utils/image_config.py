"""
Configuration settings for image processing
"""

# Video creation settings
DEFAULT_VIDEO_DURATION = 5  # seconds
DEFAULT_VIDEO_FPS = 25

# Image preprocessing settings
MIN_FACE_SIZE = 256  # minimum size for face detection
DESIRED_ASPECT_RATIO = 1.0  # desired aspect ratio for processed images

# Model paths
FACE3D_MODEL_PATH = 'checkpoints/face3d_pretrain_epoch_20.pth'
SHAPE_PREDICTOR_PATH = 'checkpoints/shape_predictor_68_face_landmarks.dat'

# Processing settings
USE_FACE3D = True  # Use 3D face reconstruction for better quality
USE_GPU = True    # Use GPU if available 