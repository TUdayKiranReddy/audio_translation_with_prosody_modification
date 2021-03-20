import speech_recognition as sr


def transcrib(audiofile, lang='en-US'):
    r = sr.Recognizer()
    with sr.AudioFile(audiofile) as source:
        audio = r.record(source)


    # Google
    print('Google Speech Recognition')
    try:
        transcrib = r.recognize_google(audio, language=lang)
        with open(audiofile[:-4]+'.txt', 'w') as subs_file:
            subs_file.write(transcrib)
        print(transcrib)
        return transcrib
    except sr.UnknownValueError:
        print('Not recongized by Google Speech API')
    except sr.RequestError as e:
        print('Google Speech error; {0}'.format(e))

        # Sphinx
    # print('Sphinx Recognition')
    # try:
    #     print(r.recognize_sphinx(audio))
    # except sr.UnknownValueError:
    #     print('Not recongized by Sphinx')
    # except sr.RequestError as e:
    #     print('Sphinx error; {0}'.format(e))
    return None