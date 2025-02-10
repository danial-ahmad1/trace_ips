import ImgProcStandalone as ip
import tkinter as tk
from tkinter import filedialog

# Example Usage with tkinter for local systems.

def select_file():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        filetypes=[("TIF files", "*.tif"), ("ND2 files", "*.nd2")]  # Only allow .tif and .nd2 files
    )  
    return file_path


# Fill in the parameters as needed. Keyframe should always be the location of a source image that you want to contrast match to.
selected_path = select_file()
img_processor = ip.Raw_Image_Processor( selected_path,
                                        desired_dim=None, # Set interpolation size for final tif stack. Set to None if you want to use the original image dimensions.
                                        keyframe='/Path_to_KeyFrameImage.tif', # Path to the keyframe image that you want to contrast match against. This should be a .tif file.
                                        save_path='/Path_to_SaveFolder/',
                                        need_video = True, # Set to true if you want a compressed (<10MB) video output.
                                        vid_fps = 24 # Frames per second for the video. Default is 24 fps.
                                        )
img_processor.run_beq()

