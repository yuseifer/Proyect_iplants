import streamlit as st
import cv2
import Detector
import wikipedia as w
from gtts import gTTS        #Import Google Text to Speech
import PIL

absolute_path = "audio/"
model_path = "weights/best (1).pt"

#Creamos el detector
detector = Detector.detector(model_path)

#Estableciendo plantilla de la página
st.set_page_config(
    page_title="Identiflora",               #Nombre de la página
    page_icon="🌱",                         #Ícono para la página
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creando la sección de subir imágen

with st.sidebar:
    st.header("Parámetros de detección")
    #Añdimos para subir la imágen
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    #Añadimos la barra de confianza
    confidence = float(st.slider(
        "Umbral de confianza para la detección", 25, 100, 40)) / 100

st.title("Identiflora")
st.caption('Suba la foto de la :blue[planta] a identificar :seedling:')
st.caption('Luego de click en el botón :blue[Detectar plantas] para obtener el resultado.')

#Creamos las columnas de la página
col1,col2,col3 = st.columns(3)

with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption="Imágen subida",
                 use_column_width=True)

if st.sidebar.button('Detectar plantas'):
    res, clases = detector.detectar(uploaded_image,
                            confianza=confidence)

    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]

    with col2:
        st.image(res_plotted,
                 caption='Imagen detectada',
                 use_column_width=True)
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")

    planta = st.sidebar.selectbox('Planta detectada',tuple(clases))
    #if st.sidebar.button('Hablar'):
    with col3:
        for c in clases:
            if c == 'Rosas':
                c='Rosa'
            #st.write(c)
            st.markdown(f"<h1 style='font-size: 50px; color: red;'>{c}</h1>", unsafe_allow_html=True)
            p = absolute_path+f'{c}.mp3'
            st.audio(p, format="audio/mp3")
