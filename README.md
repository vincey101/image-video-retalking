## CMake
First install the following:
- [CMake](https://cmake.org/download/)
- [Visual Studio](https://visualstudio.microsoft.com/) (Make sure to select "Desktop Development with C++ workload)
- [MiniConda](https://docs.anaconda.com/miniconda/)
- [git for Windows](https://git-scm.com/downloads/win)

Then proceed with the install below:

## Windows Install
```
git clone https://github.com/vincey101/image-video-retalking
cd video-retalking
conda create -n video-retalking python=3.8
conda activate video-retalking

conda install ffmpeg

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Quick Inference

#### Pretrained Models
Please download our [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and put them in `./checkpoints`.

<!-- We also provide some [example videos and audio](https://drive.google.com/drive/folders/14OwbNGDCAMPPdY-l_xO1axpUjkPxI9Dv?usp=share_link). Please put them in `./examples`. -->

#### Inference

Modify the `run.bat` to point to your anaconda activate script path and environment name and then run the batch file. The WebUI can also  be launched with `python webUI.py`.
Or you can call the original inference script with:

```
python inference.py --face examples/face/1.mp4 --audio examples/audio/1.wav --outfile results/1_1.mp4
```
This script includes data preprocessing steps. You can test any talking face videos without manual alignment. But it is worth noting that DNet cannot handle extreme poses.

You can also control the expression by adding the following parameters:

```--exp_img```: Pre-defined expression template. The default is "neutral". You can choose "smile" or an image path.

```--up_face```: You can choose "surprise" or "angry" to modify the expression of upper face with [GANimation](https://github.com/donydchen/ganimation_replicate).

## Running the Server

### Setup
First ensure you have all the required dependencies installed (see Windows Install section above).

### Start the Server
```bash
python app.py
```
The server will start on `http://localhost:5000`. An API key will be generated and displayed in the console - save this for making API requests.

### Available Endpoints

All endpoints require an `X-API-Key` header with your API key.

#### Main Endpoints
- `POST /api/generate` - Generate a talking video from face and audio inputs
  - Accepts both file uploads and URLs
  - Supports images or videos as face input
  - Supports WAV, MP3, OGG audio files

- `POST /api/store/face` - Store face image/video for processing
  - Accepts: JPG, JPEG, PNG images or MP4, WEBM videos
  - Returns: File ID and path for later use

- `POST /api/store/audio` - Store audio for processing
  - Accepts: WAV, MP3, OGG files
  - Automatically converts to WAV format if needed

#### Utility Endpoints
- `GET /api/health` - Check server status
- `GET /download/{filename}` - Download or stream generated videos
  - Add `?view=true` parameter for video streaming

### Image and Audio Features

The server can process both static images and videos:
- Upload any portrait image to create an animated talking version
- Generate character animations with synchronized lip movements
- Combine text-to-speech or custom audio with image animation
- Real-time video processing and streaming support
- Supports both local file uploads and remote URLs

## Acknowledgement
Thanks to
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip),
[PIRenderer](https://github.com/RenYurui/PIRender), 
[GFP-GAN](https://github.com/TencentARC/GFPGAN), 
[GPEN](https://github.com/yangxy/GPEN),
[ganimation_replicate](https://github.com/donydchen/ganimation_replicate),
[STIT](https://github.com/rotemtzaban/STIT)
for sharing their code.


## Related Work
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)
