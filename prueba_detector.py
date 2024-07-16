import Detector
import PIL
from ultralytics import YOLO
import pyaudio
import wave
import pyttsx3

engine= pyttsx3.init()

model_path = "C:/Users/Renzo/PycharmProjects/Herramientas_programacion_python/best.pt"
#Creamos el detector
detector = Detector.detector(model_path)
modelo=YOLO(model_path)
#Obtenemos los archivos de audio de cada clase registrada
lista_plantas = [modelo.names[(i)] for i in range(len(modelo.names.keys()))]
print(lista_plantas)
file_path = detector.hablar(lista_plantas)






#print(results)
#print(type(clase))

#streamlit run C:\Users\Renzo\PycharmProjects\Herramientas_programacion_python\identiflora.py