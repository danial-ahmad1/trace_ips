# import sys
# sys.path.insert(0, '/Users/moose/Desktop/trace_lseg-local')
import ImgProcStandalone as ip
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        filetypes=[("TIF files", "*.tif"), ("ND2 files", "*.nd2")]  # Only allow .tif and .nd2 files
    )  
    return file_path

selected_path = select_file()
img_processor = ip.Raw_Image_Processor( selected_path,
                                        desired_dim=1400,
                                        keyframe='/Users/moose/Desktop/Brightness EQ Testing/ideal_frame.tif',
                                        save_path='/Users/moose/Desktop/Newmodel_Img/',
                                        need_video = True,
                                        vid_fps = 24
                                        )
img_processor.run_beq()

