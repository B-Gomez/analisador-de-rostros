import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

#crear base de datos
ruta='Empleados'
mis_imagenes=[]
nombres_empleados=[]
lista_empleados=os.listdir(ruta)

def registrar_ingresos(persona):
    f=open('registro.csv','r+')
    lista_datos=f.readlines()
    nombre_registro=[]
    for linea in lista_datos:
        ingreso=linea.split(',')
        nombre_registro.append(ingreso[0])

    if persona not in nombre_registro:
        ahora=datetime.now()
        string_ahora=ahora.strftime('%H:%M:S')
        f.writelines(f'\n{persona},{string_ahora}')

for nombre in lista_empleados:
    imagen_actual=cv2.imread(f'{ruta}\\{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

#codificar imagenes
def codificar(imagenes):
    #crear un lista nueva
    lista_codificada=[]

    #pasar a RGB
    for imagen in imagenes:
        imagen= cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

        #codificar
        codificado= fr.face_encodings(imagen)[0]

        #agregar a la lista
        lista_codificada.append(codificado)
        print('1')
    return lista_codificada




lista_empleados_codificada = codificar(mis_imagenes)

print('tomando foto')
# tomar una imagen de camara web
captura = cv2.VideoCapture(0,cv2.CAP_DSHOW)
print('comparando')
#LEER IMAGEN DE LA CAMARA
img=fr.load_image_file('bri')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

exito=True
imagen=img

if not exito:
    print('No se ha podido tomar la caotura')
else:
    #reconoser cara en captura
    cara_captura=fr.face_locations(imagen)


    #codificar cara capturada
    cara_captura_codificada=fr.face_encodings(imagen,cara_captura)

    #buscar coincidencias
    for caracodif, caraubic in zip(cara_captura_codificada,cara_captura):
        coincidencias=fr.compare_faces(lista_empleados_codificada,caracodif)

        distancia=fr.face_distance(lista_empleados_codificada,caracodif)

        print(distancia)

        indice_coincidencia=numpy.argmin(distancia)

        #mostrar coincidencias
        if distancia[indice_coincidencia]>0.6:
            print('No coincide en la base de datos')
        else:
            #buscar nombre del empleado encontrado
            nombre=nombres_empleados[indice_coincidencia]

            y1,x2,y2,x1=caraubic

            cv2.rectangle(imagen,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(imagen,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(imagen,nombre,(x1+6,y2-6),cv2.FONT_ITALIC,2,(255,255,255),2)
            registrar_ingresos(nombre)
            #mostrar la imagen obtenida
            cv2.imshow('Imagen WEB',imagen)

            #mantener ventana abierta
            cv2.waitKey(0)