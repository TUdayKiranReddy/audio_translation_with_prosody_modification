from googletrans import Translator

def translator(ip_file, target=None):
	translator = Translator()

	with open(ip_file, 'r') as src_file:
		src_txt = src_file.read()
		tar_txt = translator.translate(src_txt, dest=target)
		print('Translated from '+tar_txt.src+' to '+target)
		print(tar_txt.text)
		with open(ip_file[:-4]+'_'+str(target)+'.txt', 'w') as tar_file:
			tar_file.write(tar_txt.text)
	return tar_txt