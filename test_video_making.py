import os
import ffmpeg


main_direc = 'C:\\Users\\Ben\\Desktop\\labelled_frames\\sample_frames'
framesFile = 'C:\\Users\\Ben\\Desktop\\labelled_frames\\sample_frames\\*png'

print(framesFile)


(
    ffmpeg
    .input(framesFile, pattern_type='glob', framerate=25)
    .output('movie.mp4')
    .run()
)



# outputFile = os.path.join(main_direc, 'video_output.mp4')
# stream = ffmpeg.input(framesFile, pattern_type='glob', framerate=30)
# stream = ffmpeg.output(stream, os.path.join(main_direc, 'video_output.mp4'))
# ffmpeg.run(stream)