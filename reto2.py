#Maria Camila Saldarriaga Ortega

import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy.stats import mode
from colormath.color_objects import XYZColor, sRGBColor


# Configuracion

n_clusters = 6
seed_radius = 5

# Carga de la imagen original
original = cv2.cvtColor(cv2.imread('mym.png'), cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(original)

# Remover el fondo seprando los canales y aplicando thresholding
img = cv2.split(original)
for idx, channel in enumerate(img):
    channel = cv2.medianBlur(channel, 5)
    _, channel = cv2.threshold(channel, 0 ,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img[idx] = channel

# Suma y normalizacion de la imagen
img = np.sum(cv2.merge(img), axis=-1)
img[img>=255] = 255

plt.figure()
plt.imshow(img)

# Remover ruido
kernel = np.ones((3,3), np.uint8)
img = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2)

# Fondo
bg = cv2.dilate(img, kernel)

# Escena
img = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_5)
peaks = peak_local_max(img, min_distance=10, threshold_rel=0.6)

plt.figure()
plt.imshow(img)

# Localizacion de los maximos locales
img = np.zeros(img.shape)
for x, y in peaks:
    img[x, y] = 255

# Labeling de de los maximos locales
n, markers = cv2.connectedComponents(img.astype(np.uint8))

# Semillas para el watershed
img = np.zeros(img.shape, np.int32)
for idx in range(n):
    (x, y) = np.where(markers == idx)
    img = cv2.circle(img, (int(np.mean(y)), int(np.mean(x))), seed_radius, idx+1, -1)

plt.figure()
plt.imshow(img)

# Region de incertidumbre
unknown = bg - img
img[unknown==255] = 0

plt.figure()
plt.imshow(unknown)

# Watershed /contornos identificadosd
segments = cv2.watershed(original, img)

plt.figure()
plt.imshow(segments)

plt.show()
# Transformacion a hsv
hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)

# Reshape para el training
segments_flat = segments.reshape((original.shape[0]*original.shape[1]))
rgb_flat = original.reshape((original.shape[0]*original.shape[1], original.shape[2]))
hsv_flat = hsv.reshape((original.shape[0]*original.shape[1], original.shape[2]))

# Seleccion de muestras relevantes
img = np.concatenate((rgb_flat,hsv_flat), axis=-1)
training_samples = np.stack([img[:,idx][segments_flat>1] for idx in range(img.shape[1])], axis=-1) 

# Entrenamiento del KMeans
clt = KMeans(n_clusters=n_clusters)
clt.fit(training_samples)

# Labeling de los pixeles de las regiones de interes
img = clt.predict(img)

img[segments_flat<=1] = -1
labels = []
for idx in range(2,n+1):
    labels.append(mode(img[segments_flat==idx], axis=None)[0])
    img[segments_flat==idx] = labels[-1]
   

# Labels de las areas de interes por color
clusters = img.reshape(original.shape[0:2])


# Hallar, contar y dibujar los bounding boxes de cada area
count = np.zeros((n_clusters,), int)
for idx in range(2, n+1):
    drawing = segments==idx
    (_, contours, _) = cv2.findContours(drawing.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original = cv2.drawContours(original, contours, 0, [0,0,0], 2)
    center, radius = cv2.minEnclosingCircle(contours[0])
    center = (int(center[0]-radius*0.5), int(center[1]+radius*0.6))
    count[labels[idx-2]] = count[labels[idx-2]]+1
    original = cv2.putText(original, str(count[labels[idx-2]][0]), center, cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 2)

# Extraer y normalizar los centroides de los clusters en RGB
colors = clt.cluster_centers_[:, 0:3].astype(int)

explode = (0.1, 0, 0, 0,0,0)

plt.pie(count, explode=explode, labels=count )



c=colors[0][0]
print(colors) 
print(c)

plt.figure()
plt.imshow(original)
plt.show()