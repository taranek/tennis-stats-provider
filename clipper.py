from moviepy import VideoFileClip

clip = VideoFileClip("evans.mp4")
starting_point = 420  # start at second minute
end_point = 3420   # record for 300 seconds (120+300)
subclip = clip.subclip(starting_point, end_point)
subclip.write_videofile("evans_clip.mp4")