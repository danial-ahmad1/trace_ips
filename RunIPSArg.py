import ImgProcStandalone as ip
import argparse
import os

# Example Usage with argparse.

parser = argparse.ArgumentParser(prog='RunIPS', description='Run on a folder to contrast flatten phase contrast microscopy videos.')
parser.add_argument('-f', '--folder', type=str, required=True, help='Path to the folder to be analyzed')
parser.add_argument('-d', '--dimension', type=int, required=False, help='Final .tif dimensions')
parser.add_argument('-k', '--keyframe', type =str, required=False, help='Path to ideal frame')
parser.add_argument('-s', '--savepath', type=str,  required=True, help='Path to save location for final output')
parser.add_argument('-v', '--video', type=bool, required=False, help='True/False for a compressed .mp4 video after analysis')
parser.add_argument('-r', '--framerate', type=int, required=False, help='Frame rate for compressed video')

args = parser.parse_args()

if args.folder.endswith('/'):
    pass
else:
    args.folder += '/'

if args.savepath.endswith('/'):
    pass
else:
    args.savepath += '/'

selected_path = args.folder
files = os.listdir(selected_path)
files_im = [f for f in files if f.endswith('.tif') or f.endswith('.nd2')]
save_pth = args.savepath
bool_video = args.video

if args.dimension is None:
    des_dim = None
else:
    des_dim = args.dimension

if args.keyframe is None:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    kf = os.path.join(script_dir + '/keyframe.tif')
else:
    kf = args.keyframe


if bool_video is True:
    pass
else:
    bool_video = False
    
if args.framerate is None:
    sel_fps = 24
else:
    sel_fps = args.framerate

print(f'Folder: {args.folder}')
print(f'Files: {files_im}')
print(f'Dimension: {args.dimension}')
print(f'Keyframe: {args.keyframe}')
print(f'Save Path: {args.savepath}')
print(f'Video: {args.video}')
print(f'Framerate: {args.framerate}')

for i in range(len(files_im)):
    print(f'Processing file: {files_im[i]}')
    img_processor = ip.Raw_Image_Processor( os.path.join(selected_path + files_im[i]), 
                                            desired_dim=des_dim, # Set interpolation size for final tif stack. Set to None if you want to use the original image dimensions.
                                            keyframe=kf, # Path to the keyframe image that you want to contrast match against. This should be a .tif file.
                                            save_path=save_pth,
                                            need_video = bool_video, # Set to true if you want a compressed (<10MB) video output.
                                            vid_fps = sel_fps # Frames per second for the video. Default is 24 fps.
                                            )
    img_processor.run_beq()