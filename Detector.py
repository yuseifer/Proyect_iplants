import wikipedia as w
from gtts import gTTS
import PIL
from ultralytics import YOLO

w.set_lang("es")
language = "es-es"
absolute_path ="C:/Users/Renzo/PycharmProjects/Herramientas_programacion_python/"

class detector:
  def __init__(self,model_path):
    self.model_path = model_path



  def detectar(self,imagen, confianza= 0.5):
      resultado_nombres=[]
      model = YOLO(self.model_path)
      results = model.predict(imagen,conf=confianza)
      clases = model.names
      for r in results:
          for c in r.boxes.cls:
              resultado_nombres.append(clases[int(c)])
      clases = set(resultado_nombres)

      return results, clases


  def hablar(self,keywords):
    sound_file=[]
    for k in keywords:
        if k == 'Rosas':
            k='Rosa'
        resultado = w.summary(k)
        tts = gTTS(resultado, lang=language)  # Provide the string to convert to speech
        tts.save(absolute_path+f'{k}.wav')  # save the string converted to speech as a .wav file
        sound_file.append(absolute_path+f'{k}.wav')
    return sound_file

