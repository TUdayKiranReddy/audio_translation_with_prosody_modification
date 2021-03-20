import os
import sys
import subprocess

def install(pkg):
	subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

def separater(file):
	os.system('ffmpeg -i '+file+' -ac 1 -f wav '+file[:-4]+ '_audio' + '.wav')
	# video = mp.VideoFileClip(file)
	# audio = video.audio
	# audio.write_audiofile(file + '_audio' + '.wav')

def replace_audio(video_file, target_audio_file, output_file='video.mp4'):
	os.system('ffmpeg -i '+ video_file +' -i ' + target_audio_file +
	 ' -shortest -c:v copy -c:a aac -b:a 256k ' + output_file)

def extract_subtitles(video_file, opfile='subs.srt'):
	os.system('ffmpeg -i '+video_file+' -map 0:s:0 '+ opfile) 