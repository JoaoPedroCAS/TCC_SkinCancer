# Introdução

Este estudo tem como objetivo propor duas metodologias diferentes para diagnosticar lesões cancerígenas da pele a partir das imagens obtidas pelo *dataset*  Homem Contra Máquina com 10000 imagens para treino (Human Against Machine with 10000 training images - [HAM10000](https://arxiv.org/abs/1803.10417)). Os resultados serão avaliados através de diferentes métricas, como a acurácia, o *F1-Score* e a precisão.

A primeira abordagem envolve a utilização de redes neurais convolucionais (*Convolutional Neural Networks - CNN's*) como a VGG 19, Inception e Xception estritamente como métodos de classificação.

A segunda metodologia irá utilizar as mesmas *CNN's* como extratores de atributos. Os atributos extraídos servem de entrada para diversos algoritmos clássicos de inteligência artificial, como a análise discriminante linear (*Linear Discriminant Analysis - LDA*), o Perceptron, a máquina de vetores de suporte linear (*Linear Support Vector Machine (Linear SVM)*), que irão classificar as imagens.

Estas duas formas diferentes de classificação são avaliadas com a finalidade de encontrar quais são os pontos fortes e fracos de cada uma dessas abordagens e qual destas gera o melhor resultado. 

Futuramente, uma terceira estratégia irá ser implementada, onde mais uma vez as CNN’s irão ser utilizadas apenas como extratoras de atributos, que servirão como entrada para algoritmos de seleção dinâmica, como o Seleção dinâmica de conjunto com recurso META (*META-feature Dynamic Ensemble Selection - META-DES*) e o K-vizinhos mais próximos com rejeição otimizada e seleção dinâmica (*K-nearest neighbors with Optimized Rejection and dynamic Selection -* *KNORA-U*), utilizados para a classificação.

# Materiais e Métodos

Esta seção trata do dataset utilizado e das metodologias abordadas para a exploração do HAM10000.

Os códigos explicitados nas subseções seguintes podem ser encontrados no [Github](https://github.com/JoaoPedroCAS/TCC_SkinCancer).

## Dataset HAM10000

O dataset HAM10000 contém 10015 imagens em alta resolução de lesões pigmentadas na pele. Estas imagens representam um conjunto amplo de condições dermatológicas que podem ser utilizadas para treinamento de algoritmos de aprendizagem profunda e outros métodos de aprendizado de máquina.

O dataset possui sete classes diferentes de lesões pigmentadas. Dentre as lesões estão os carcinomas basocelulares, os melanomas, as queratoses actínicas, lesões benignas do tipo queratose, dermatofibromas, nevos melanocíticos e lesões vasculares. As imagens deste banco de dados são de vários territórios da Europa, gerando uma variabilidade no dataset, o que representa cenários mais realistas. 

Apesar deste dataset ser comumente utilizado para classificação entre diferentes diferentes classes em conjunto com a utilização de *CNN’s* para a predição das classes, outras metodologias podem ser utilizadas, como serão demonstradas em seções seguintes desta seção.

## Redes Neurais como Classificadores

Esta seção aborda o passo a passo da construção do código de classificação utilizando CNN's.

```python
import tensorflow as tf
```

- Plataforma de código aberto que é utilizada para códigos de inteligência artificial.

```python
import cv2
```

- Essa biblioteca será utilizada para ler e alterar o tamanho das imagens.

```python
import numpy as np
```

- A biblioteca NumPy provê manipulações matemáticas e aritméticas sobre vetores, além de incluir operações de álgebra linear, transformadas de Fourier e gera números aleatórios.

```python
import matplotlib.pyplot as plt
```

- Utilizado para fazer os gráficos e abrir figuras na tela

```python
from sklearn.model_selection import train_test_split
```

- A função train_test_split irá dividir os vetores ou matrizes em conjuntos aleatórios de treino e teste.

```python
from sklearn import preprocessing
```

- Inclui os principais módulos de escalonamento, centralização, normalização e métodos binários.

```python
from keras.models import Model
```

- Agrupa camadas em um objeto com recursos para treinamento e inferência.

```py
from keras.layers import Dense, GlobalAveragePooling2D
```

- Dense: implementa a rede neural densamente conectada em N camadas.
- GlobalAveragePooling2D: Operação de agrupamento de média global.

```python
from keras.applications import VGG19
```

- Implementa a arquitetura da rede VGG19, 

```python
drive_path = 'insira aqui o seu caminho'
data_file_path = os.path.join(drive_path, 'data.txt')
```

- O arquivo data.txt é composto do nome de cada arquivo .jpeg a ser analisado, além da classe a que este artigo pertence. 
- A variável drive_path contém o caminho para o diretório que possui as imagens e o arquivo data.txt
- data_file_path é a variável que leva até o arquivo data.txt

```python
with open(data_file_path, 'r') as file:
    lines = file.readlines()
```

- Abre o arquivo data.txt no modo leitura
- Le individualmente cada linha, salvando na variável lines

```python
X = []
y = []
for line in lines:
    img_name, label = line.split()
    img_path = drive_path + 'data\\' + name
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append(label)
```

- Inicia as variáveis X e y, a variável X contém as imagens em si, já a variável y contém a classe que a imagem pertence
- Para cada objeto na variável lines o programa irá:
  - Separar o nome da imagem (o .jpeg) da classe da imagem
  - Colocar na variável img_path o caminho completo até a imagem
  - Salvar a imagem utilizando a biblioteca cv2 na variável img
  - Transformar a imagem para o tamanho padrão da rede VGG16 (224, 224)
  - Adicionar a imagem à variável X
  - Adicionar a classe à variável y

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

- Separa o conjunto de treino e teste, nesse caso o tamanho do conjunto de teste é de 33% do conjunto total, o parâmetro random_state é um inteiro que garante a aleatoriedade no processo.

```python
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

- A primeira linha cria uma instância do LabelEncoder() e a salva na variável le
- o conjunto de treino e teste recebem as classes respostas de forma numérica

```
y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)
```

- Essa linha altera as classes de números para vetores:
  - A classe 0 pode, por exemplo, ser representada pelo vetor [1,0,0,0,0,0,0]
  - A classe 1 pode ser representada pelo vetor [0,1,0,0,0,0,0]
  - E assim sucessivamente

```python
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test) 
```

- Transformando os vetores em vetores numpy, que são mais facilmente manipuláveis.

```python
img_rows, img_cols = 224, 224
vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
```

- A primeira linha mostra o tamanho em linhas e colunas da imagem
- A segunda linha instância a rede VGG16, utilizando os pesos da imagenet (base de dados que pode ser utilizada gratuitamente), não incluindo  as 3 camadas totalmente conectadas, já o parâmetro input-shape mostra a configuração da imagem. 

```python
for layer in vgg.layers:
    layer.trainable = False
```

- Congela a base da da rede neural, impossibilitando que os pesos das camadas da VGG16 se atualizem enquanto o treino acontece.

```python
def add_custom_head(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model
```

- Define a função add_custom_head(), que usa como parâmetros o modelo (a rede vgg16) e o número de classes (nesse caso, sete).
  - A saída da rede VGG16 é utilizada como entrada para a variável top_model
  - A função GlobalAveragePooling2D é utilizada para computar o valor médio de cada mapa de atributos em todas as dimensões, reduzindo a dimensão espacial para um único valor
  - Duas camadas totalmente conectadas de 1024 neurônios são adicionadas ao modelo utilizando a ReLU
  - Mais uma camada totalmente conectada de 512 neurônios é adicionada
  - A última camada possui 7 neurônios e utiliza a ativação do softmax

```python
num_classes = 7
custom_head = add_custom_head(vgg, num_classes)
model = Model(inputs=vgg.input, outputs=custom_head)
```

- Parâmetros utilizados na função add_custom_head
- O modelo é composto das entradas da VGG e das saídas da cabeça retornada pela função add_custo_head.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

- Configura o modelo para o treinamento
- O otimizador adam é escolhido por ser computacionalmente eficiente, exigir pouca memória e serve para problemas com muitos dados. A função de perda  categorical_crossentropy mede a diferença entre a distribuição de probabilidade prevista e os rótulos verdadeiros.

```python
history = model.fit(np.array(X_train), np.array(y_train), epochs=10, validation_data=(np.array(X_test), np.array(y_test)), verbose=1)
```

- Treina o modelo por um número fixo de épocas.

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
```

- Define os valores de perda e acurácia para o conjunto de treino e para a validação

```python
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```

- Gera um gráfico com a acurácia de treino e de validação, a fim de facilitar a visualização dos resultados.

## Extraindo atributos com diferentes CNN's

```python
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg
from keras.applications.xception import Xception, preprocess_input as preprocess_xception
```

- Algumas dessas bibliotecas já foram explicadas na subseção acima.
  - A biblioteca keras.preprocessing trata do pré processamento de imagens.
  - As linhas seguintes tratam da importação das CNN's que serão utilizadas no projeto, três dessas foram escolhidas com o intuito de comparar resultados.

```python
drive_path = 'C:\\Users\\jpedr\\OneDrive\\Documentos\\TCC\\Codigos\\CancerDePele\\cancer\\'
entrada = drive_path + 'data.txt'
dir_dataset = drive_path + 'data\\'
dir_destino = drive_path + 'libsvm\\'
```

- Definindo os caminhos para o .txt com o nome das imagens e as classes, para a pasta com as imagens e para a pasta onde ficarão os atributos extraídos.

```python
if not os.path.exists(dir_destino):
    os.makedirs(dir_destino)
```

- Garante que a pasta destino dos atributos extraídos exista

```python
img_rows, img_cols = 224, 224
with open(entrada, 'r') as arq:
    conteudo_entrada = arq.readlines()
```

- Definindo o tamanho das linhas e colunas das imagens, após isso abre o arquivo com o nome das imagens e suas respectivas classes no modo leitura, por fim, salva cada linha desse arquivo na variável conteudo_entrada.

```python
def process_images(model, preprocess_input, output_file):
    with open(output_file, 'w') as file:
        for i in conteudo_entrada:
            nome, classe = i.split()
            img_path = dir_dataset + nome
            img = image.load_img(img_path, target_size=(img_rows, img_cols))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            features = model.predict(img_data)
            features_np = np.array(features).flatten()
            file.write(classe + ' ')
            for j in range(features_np.size):
                file.write(str(j+1) + ':' + str(features_np[j]) + ' ')
            file.write('\n')
```

- Cria a função que irá processar as imagens, o modelo pode ser o Inception, o Xception ou o VGG19, mas qualquer outra CNN pode ser implementada.
  - Abre o arquivo que armazenará as saídas.
    - Separa o conteúdo na variável conteudo_linha entre nome e classe
    - O caminho para a imagem é escrito na variável img_path
    - Carrega a imagem na variável img
    - Converte a imagem em um vetor
    - Expande as dimensões do vetor
    - Pré processa as imagens
    - Prediz as características
    - Converte os atributos para um vetor de uma única dimensão 
    - Escreve no arquivo de saída: a classe resposta e todas as características extraídas.

```
# model_VGG = VGG19(weights='imagenet', include_top=False)
# process_images(model_VGG, preprocess_vgg, dir_destino + 'data_VGG.txt')

# model_Xception = Xception(weights='imagenet', include_top=False)
# process_images(model_Xception, preprocess_xception, dir_destino + 'data_Xception.txt')

model_Inception = InceptionV3(weights="imagenet", include_top=False)
process_images(model_Inception, preprocess_inception, dir_destino + 'data_Inception.txt')
```

- Define qual dos modelos de CNN será utilizado.

## Predição por métodos clássicos de IA

```python
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score
import random

```

- Traz as bibliotecas utilizadas para a predição ou para o auxílio desta.
  - load_svmlight_file é utilizado para ler os atributos extraídos
  - RepeatedStraitifiedKFold é utilizado no processo de cross-validation
  - Os métodos KneighborsClassifier, LinearDiscriminantAnalysis, LogisticRegression, Perceptron e LinearSVC são os métodos de predição utilizados.
  - As métricas de avaliação são realizadas pelos métodos accuracy_score, f1_score, precision_score
  - A biblioteca random também é utilizada no processo de cross-validation

```python
def initialize_models(max_iter=5000, random_state=None):
    lda = LinearDiscriminantAnalysis()
    perceptron = Perceptron(max_iter=max_iter)
    linearSVM = LinearSVC(dual=True, max_iter=max_iter, random_state=random_state)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(max_iter=max_iter)
    return lda, perceptron, linearSVM, knn, lr
```

- Função que inicializa todos os métodos de aprendizado de máquina, alguns com seus parâmetros adicionais para garantir uma melhor execução.

```python
def load_data(data):
    X_data, y_data = load_svmlight_file(data)
    X_data = X_data.toarray()  # Convert to dense array
    return X_data, y_data
```

- Função responsável por ler os atributos que estão salvos no formato svmlight, cada linha deste arquivo é formado pela classe resposta e todos os atributos de cada uma das imagens.

```python
def print_metrics(y_test, model_pred, model_name):
    print(f'Accuracy {model_name}: ', accuracy_score(y_test, model_pred))
    print(f'F1-Score {model_name}: ', f1_score(y_test, model_pred))
    print(f'Precision {model_name}: ', precision_score(y_test, model_pred))
```

- Função responsável por gerar as métricas que servem como resultado para este experimento.

```python
def main(data):
    # Initialize models
    lda, perceptron, linearSVM, knn, lr = initialize_models()

    # Load data
    X_data, y_data = load_data(data)

    # Define Repeated Stratified K-Fold cross-validator
    rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random.randint(0, 368512346))

    # Iterate over the folds
    for i, (train, test) in enumerate(rsfk.split(X_data, y_data)):
        # Split data
        X_train, y_train = X_data[train], y_data[train]
        X_test, y_test = X_data[test], y_data[test]

        # Fit models
        print('Fitting...')
        lda.fit(X_train, y_train)
        perceptron.fit(X_train, y_train)
        linearSVM.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        # Predict
        print('Predicting...')
        lda_pred = lda.predict(X_test)
        perceptron_pred = perceptron.predict(X_test)
        linearSVM_pred = linearSVM.predict(X_test)
        knn_pred = knn.predict(X_test)
        lr_pred = lr.predict(X_test)

        # Print metrics
        print_metrics(y_test, lda_pred, 'LDA')
        print_metrics(y_test, perceptron_pred, 'Perceptron')
        print_metrics(y_test, linearSVM_pred, 'Linear SVM')
        print_metrics(y_test, knn_pred, 'KNN')
        print_metrics(y_test, lr_pred, 'LR')

# Execute main function if the script is run directly
if __name__ == "__main__":
    main("cancer/libsvm/data_Inception.txt")
```

- Execução da função principal do código.
  - Na primeira linha chama a função que inicializa todos os métodos de aprendizado de máquina
  - Em seguida, chama a função que separa os dados em X e y, ou seja, atributo e resposta.
  - Depois faz os ajustes dos dados para cada algoritmo utilizado pelas funções de *fit*
  - A partir deste momento já podemos fazer as predições, estes métodos já possuem funções que as fazem de forma automática, chamadas *predict*
  - Por fim, cada uma das métricas é impressa.

## Resultados

### CNN como classificadora

Os resultados gerados pela CNN estão apresentados abaixo.

```
        Classes |  Precisão |  F1-Score |
           0    |   0.57    |  0.39     |
           1    |   0.80    |  0.88     |
           2    |   0.37    |  0.30     |
           3    |   0.43    |  0.39     |
           4    |   0.43    |  0.33     |
           5    |   0.64    |  0.34     |
           6    |   0.47    |  0.56     |

Acurácia média: 0.74
```

### CNN como extratora e métodos de aprendizagem de máquina como classificadores

A rede neural Xception gerou um erro ocasionado pela falta de RAM no momento de extrair , então os resultados apresentados aqui não possuem as

Os resultados obtidos pelos métodos de aprendizagem de máquina e CNN extratora estão apresentados abaixo.

```
LDA - Inception:
Classes| Precisão | F1-Score |
   0   | 0.48     | 0.43     |
   1   | 0.83     | 0.88     |
   2   | 0.47     | 0.32     |
   3   | 0.62     | 0.54     |
   4   | 0.72     | 0.34     |
   5   | 0.45     | 0.37     |
   6   | 0.88     | 0.58     |

Acurácia média: 0.75
```

```
Perceptron - Inception:
Classes| Precisão | F1-Score |
   0   | 0.47     | 0.49     |
   1   | 0.84     | 0.88     |
   2   | 0.51     | 0.33     |
   3   | 0.64     | 0.56     |
   4   | 0.38     | 0.38     |
   5   | 0.45     | 0.36     |
   6   | 0.62     | 0.51     |

Acurácia média: 0.75
```

```
Linear SVM - Inception:
Classes| Precisão | F1-Score |
   0   | 0.49     | 0.48     |
   1   | 0.84     | 0.88     |
   2   | 0.50     | 0.34     |
   3   | 0.58     | 0.55     |
   4   | 0.72     | 0.43     |
   5   | 0.44     | 0.39     |
   6   | 0.78     | 0.53     |

Acurácia média: 0.75
```

```
KNN - Inception:
Classes| Precisão | F1-Score |
   0   | 0.39     | 0.40     |
   1   | 0.79     | 0.85     |
   2   | 0.24     | 0.17     |
   3   | 0.36     | 0.28     |
   4   | 0.19     | 0.13     |
   5   | 0.41     | 0.25     |
   6   | 0.66     | 0.15     |

Acurácia média: 0.70
```

```
LR - Inception:
Classes| Precisão | F1-Score |
   0   | 0.48     | 0.48     |
   1   | 0.84     | 0.88     |
   2   | 0.50     | 0.36     |
   3   | 0.58     | 0.55     |
   4   | 0.64     | 0.41     |
   5   | 0.45     | 0.39     |
   6   | 0.84     | 0.52     |

Acurácia média: 0.75
```

