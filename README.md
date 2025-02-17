<b>Phase contrast video processor for immune cell transmigration studies.</b>\
Developed for TraCe-bMPS, University of Rochester.


The following code will take a .tif or .nd2 image stack consisting of phase contrast images and contrast match/balance the entire stack to remove common microscopy artifacts.\
\
Image stacks are matched to a 'keyframe.tif', with a default example provided in the repository.\
Users can use either an 'argparse' via command line(RunIPSArg.py) or simple GUI via tkinter (RunIPS.py) to run the code.

For the tkinter based approach, the following variables need to filled out:

```python
img_processor = ip.Raw_Image_Processor( selected_path,
                                        desired_dim=None, # Set interpolation size for final tif stack. Set to None if you want to use the original image dimensions.
                                        keyframe='/Path_to_KeyFrameImage.tif', # Path to the keyframe image that you want to contrast match against. This should be a .tif file.
                                        save_path='/Path_to_SaveFolder/',
                                        need_video = True, # Set to true if you want a compressed (<10MB) video output.
                                        vid_fps = 24 # Frames per second for the video. Default is 24 fps.
                                        )
```
And for the argparse approach, the command line usage is as follows. Note, only -f (folder path) and -s (save path) are mandatory.\
You can set an image axis interpolation size with -d, different keyframe with -k (not recommended), and generate a compressed .mp4 video with -v (True/False) and -r (target framerate).

```bash
usage: RunIPSArg.py [-h] -f FOLDER [-d DIMENSION] [-k KEYFRAME] -s SAVEPATH
              [-v VIDEO] [-r FRAMERATE]

Run on a folder to contrast flatten phase contrast microscopy videos.

options:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        Path to the folder to be analyzed
  -d DIMENSION, --dimension DIMENSION
                        Final .tif dimensions
  -k KEYFRAME, --keyframe KEYFRAME
                        Path to ideal frame
  -s SAVEPATH, --savepath SAVEPATH
                        Path to save location for final output
  -v VIDEO, --video VIDEO
                        True/False for a compressed .mp4 video after analysis
  -r FRAMERATE, --framerate FRAMERATE
                        Frame rate for compressed video
```
