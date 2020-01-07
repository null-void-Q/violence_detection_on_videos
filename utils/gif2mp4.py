import moviepy.editor as mp
import sys
import os
import time
directory = sys.argv[1]



gifs = []
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in [f for f in filenames if f.endswith(".gif")]:
        gifs.append(os.path.join(dirpath, filename))


for i,gif in enumerate(gifs):        
    clip = mp.VideoFileClip(gif)
    clip.write_videofile(gif[:-4]+".mp4")
    time.sleep(0.1)
    os.remove(gif)

    