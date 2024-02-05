from PIL import Image
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
import numpy as np
import os

drive_path = 'C:\\Users\\jpedr\\OneDrive\\Documentos\\TCC\\Codigos\\CancerDePele\\cancer\\' 

entrada = drive_path + 'data.txt' 

arq = open(entrada,'r')
conteudo_entrada = arq.readlines()
arq.close() 

dir_dataset = drive_path + 'data\\' 

dir_destino = drive_path + 'libsvm\\' 

if not os.path.exists(dir_destino):
  os.makedirs(dir_destino)

img_rows, img_cols = 299, 299 

##CASO UTILIZE O VGG COMO EXTRATOR
""" arq_svm_VGG = dir_destino + 'data_VGG.txt' 
file_svm_VGG = open(arq_svm_VGG, 'w')
model_VGG = VGG19(weights='imagenet', include_top=False)
for i in conteudo_entrada:
  nome, classe = i.split()
  img_path = dir_dataset + nome
  print (img_path) 
  img = image.load_img(img_path, target_size=(img_rows,img_cols))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  inception_features = model_VGG.predict(img_data)
  features_np = np.array(inception_features)
  features_np = features_np.flatten()
  file_svm_VGG.write(classe+' ')
  for j in range (features_np.size):
    file_svm_VGG.write(str(j+1)+':'+str(features_np[j])+' ')
  file_svm_VGG.write('\n')
file_svm_VGG.close() """


##CASO USE O XCEPTION COMO EXTRATOR
""" arq_svm_Xception = dir_destino + 'data_Xception.txt' ##Cria o arquivo onde ficarão os atributos
file_svm_Xception = open(arq_svm_Xception, 'w') ##Abre o arquivo no modo leitura
model_Xception = Xception(weights='imagenet', include_top=False) ##PARA USAR O EXTRATOR DO XCEPTION
for i in conteudo_entrada:
  nome, classe = i.split()
  img_path = dir_dataset + nome 
  print (img_path)
  img = image.load_img(img_path, target_size=(img_rows,img_cols))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  inception_features = model_Xception.predict(img_data)
  features_np = np.array(inception_features)
  features_np = features_np.flatten()
  file_svm_Xception.write(classe+' ')
  for j in range (features_np.size):
    file_svm_Xception.write(str(j+1)+':'+str(features_np[j])+' ')
  file_svm_Xception.write('\n')
file_svm_Xception.close() """

##CASO USE O INCEPTION COMO EXTRATOR
arq_svm_Inception = dir_destino + 'data_Inception.txt' ##Cria o arquivo onde ficarão os atributos
file_svm_Inception = open(arq_svm_Inception, 'w') ##Abre o arquivo no modo leitura
model_Inception = InceptionV3(weights="imagenet", include_top=False) ##PARA USAR O EXTRATOR DO INCEPTIONv3
for i in conteudo_entrada:
  nome, classe = i.split()
  img_path = dir_dataset + nome
  print (img_path)
  img = image.load_img(img_path, target_size=(img_rows,img_cols))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  inception_features = model_Inception.predict(img_data)
  features_np = np.array(inception_features)
  features_np = features_np.flatten()
  file_svm_Inception.write(classe+' ')
  for j in range (features_np.size):
    file_svm_Inception.write(str(j+1)+':'+str(features_np[j])+' ')
  file_svm_Inception.write('\n')
file_svm_Inception.close()
##-------------------------------------------------------------------------------------------------


