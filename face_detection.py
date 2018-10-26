#! /usr/bin/env python3

import cv2, os, time, sys
import numpy as np 
import pickle
import shutil 

#cv2 pour l'utilisation de la webcam
#os pour pouvoir save les images tests
#numpy pour faire des calculs matriciels
#time pour gerer le temps et l'ajouter dans le nom pour le trainning
#shutil pour deplacer les photos dans un repertoire

#Utilsation des patterns pour les haars
face_cascade =  cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#On recupere le temps courant 
localtime = time.asctime((time.localtime(time.time())))
origin_path = "/home/julien/python/OpenCV/Script/MagicMirror/"
final_path = "/home/julien/python/OpenCV/Script/MagicMirror/images/"

#On lit le modèle entrainé
recognizer.read("trainner.yml")
labels = {"person_name": 1}
i = 1 

with open('label.pickle', "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#initialisation de la vidéo
cap = cv2.VideoCapture(0)
#Ouverture de la vidéo 

while (cap.isOpened()):
    #decoupage frame par frame
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        #Affiche les postion x,y,w,h du visage en fonction du frontalface pattern    
        #print(x,y,w,h)
        rectangle_color = (255,0,0)
        stroke = 2
        cv2.rectangle(img, (x,y), (x+w, y+h), rectangle_color, stroke)
        roi_gray =  gray[y:y+h, x:x+y]
        #cv2.imwrite("hello.jpg",roi_gray)
        roi_color = img[y:y+h, x:x+y]
        id_, conf = recognizer.predict(roi_gray)
        
        if conf>=70 :#and conf <=85:
            #print(conf)
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 0) 
            strole = 2
            counter = 1
            #Affiche le texte au dessus de l'haar
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            for i in range (0,2):
                print(name)
                img_item = name+".png"
                print(img_item)
                cv2.imwrite(img_item, img)
                cv2.imwrite(img_item, roi_color)
                origin_path = origin_path+img_item
                print(origin_path)
                folder_list = os.listdir(final_path)
                #Pour les noms de fichier dans le dossier "python/OpenCV/Script/Mirroir_Connecté/images/"
                for folder_name in folder_list:
                    #On affiche la liste des dossiers
                    print(folder_name)
                    #Si l'un des dossiers se nomme "michel" alors on dit bonjour, on se deplace dans le dossier 
                    
                    if folder_name == "michel" and folder_name ==  name:
                        shutil.move(origin_path, "/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                        count_the_number_of_file_inside_the_directory = os.listdir("/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                        for filename in len(count_the_number_of_file_inside_the_directory):
                            print(count_the_number_of_file_inside_the_directory)
                        
                    elif folder_name == "julien_hivert" and folder_name == name: 
                        #on depace la photo dans le dossier de la personne

#TODO CREER UN DOSSIER TAMPON 

                        #origin_path =  racine du projet
                        shutil.move(origin_path, "/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                        count_the_number_of_file_inside_the_directory = len(os.listdir("/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name))
                        #Pour les noms de fichiers dans la taille du tableau de fichier 
                        for filename in range (count_the_number_of_file_inside_the_directory):
                            #On incremente le compteur 
                            i = count_the_number_of_file_inside_the_directory + 1 
                            #On rename le fichier julien_hivert.png par le nombre /i/.png
                            final_name = str(i)+".png"
                            print(final_name)
                            os.rename("/home/julien/python/OpenCV/Script/MagicMirror/images/julien_hivert/"+img_item, "/home/julien/python/OpenCV/Script/MagicMirror/images/julien_hivert/"+final_name) 
                    # elif folder_name == "Eva_Green" and folder_name == name:
                    #     shutil.move(origin_path, "/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                    #     count_the_number_of_file_inside_the_directory = os.listdir("/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                    #     for filename  in range (len(count_the_number_of_file_inside_the_directory)):
                    #         i = i+1
                    #         print(i)
                    #         os.rename("/home/julien/python/OpenCV/Script/MagicMirror/"+img_item, "/home/julien/python/OpenCV/Script/MagicMirror/Eva_Green/"+str(i))
                    #     sys.exit()
                    elif folder_name == "emilia_clarcke" and folder_name == name:
                        shutil.move("/home/julien/python/OpenCV/Script/MagicMirror/"+img_item, "/home/julien/python/OpenCV/Script/MagicMirror/images/"+folder_name)
                        sys.exit()
                    else :
                        print("Personne inconnu au bataillon")
            #Affichage du resultat
            cv2.imwrite(img_item, roi_color)

    cv2.imshow("frame", img)
    if cv2.waitKey(30) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()