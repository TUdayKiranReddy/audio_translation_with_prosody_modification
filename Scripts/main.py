import sys
from utills import *
from speech2text import *
from translator import *


def main(file_name, src_lang, target):
	# print('###################')
	# print('Installing Dependencies')
	# print('###################')

	# install('SpeechRecognition')
	# install('google-cloud-speech')
	# install('googletrans==3.1.0.0a0')

	print('###################')
	print('Extracting audio from ' + file_name)
	print('###################')
	
	separater(file_name)
	audiofile = file_name[:-4]+ '_audio' + '.wav'

	print('###################')
	src_subs = audiofile[:-4]+'.txt'
	print('Speech to text of ' + audiofile + ' and writing to ' + src_subs)
	print('###################')

	transcrib(audiofile, lang=src_lang)

	print('###################')
	dest_subs = src_subs[:-4]+'_'+str(target)+'.txt'
	print('Traslation of ' + src_subs + 'to ' + target+ ' and writing it to '+ dest_subs)
	print('###################')

	translator(src_subs, target=target)

if __name__ == '__main__':
	file_name = sys.argv[1]
	src_lang = sys.argv[2]
	dest_lang = sys.argv[3]
	main(file_name, src_lang, dest_lang)
	