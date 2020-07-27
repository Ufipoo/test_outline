import matplotlib.pyplot as plt
import numpy as np
import cv2

sigma = 1
filter = 31

path = 'Materials/cameraman.tif'

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(img)
plt.show()
print("Image shape", img.shape)

def set_noise(img):
    x = img.shape[0]
    y = img.shape[1]
    z = img.shape[2]

    mu, sigma = 5, 10
    noise = np.random.normal(mu, sigma, ((x, y, z)))
    image = img + noise

    max = np.max(image)
    min = np.min(image)
    image = ((image - min) / (max - min))
    return image

def to_gray(img):
    new_image = np.ones((img.shape[0], img.shape[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            r = img[x, y][0]
            g = img[x, y][1]
            b = img[x, y][2]
            sr = (r + g + b) / 3
            new_image[x, y] = sr
    return new_image

def set_padding(img, filter):
    padding = int(np.floor(filter / 2))
    x = img.shape[0]
    z = img.shape[2]
    pad_y = np.zeros((x, padding, z))
    img = np.concatenate((pad_y, img), axis=1)
    img = np.concatenate((img, pad_y), axis=1)

    y = img.shape[1]
    z = img.shape[2]
    pad_x = np.zeros((padding, y, z))
    img = np.concatenate((pad_x, img))
    img = np.concatenate((img, pad_x))
    return img

def gauss(sigma, filter):
    coord = np.linspace(-3 * sigma + 1, 3 * sigma - 1, filter)
    G = np.zeros((filter, filter))
    for x in range(filter):
        for y in range(filter):
            G[x, y] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(coord[x] ** 2 + coord[y] ** 2) / (2 * sigma ** 2))
    return G

def bluring(img, sigma, filter):
    pad_image = set_padding(img, filter)
    new_image = np.ones((img.shape[0], img.shape[1], img.shape[2]))
    G = gauss(sigma, filter)

    for z in range(pad_image.shape[2]):
        for x in range(pad_image.shape[0] - filter):
            for y in range(pad_image.shape[1] - filter):
                t = pad_image[x : x + filter, y : y + filter, z]
                pixel = np.sum(t * G)
                new_image[x, y, z] = pixel
    max = np.max(new_image)
    min = np.min(new_image)
    new_image =((new_image - min) / (max - min))
    return new_image

#добавить шум
image_noise = set_noise(img)
plt.figure(2)
plt.imshow(image_noise)
plt.show()

# перевести изображение в серое и построить гистограмму
image_gray = to_gray(image_noise)
plt.figure(3)
plt.imshow(image_gray, cmap='gray')
plt.show()

plt.figure(4)
vals = image_gray.flatten()
vals = ((vals - 0) / (1 - 0)) * (255 - 0) + 0 # нормировка к (0, 255)
bins = range(255)
plt.hist(vals, bins)
plt.xlim([0, 255])
plt.show()

# размыть изображение (цветное)
image_blur = bluring(img, sigma, filter)
plt.figure(5)
plt.imshow(image_blur)
plt.show()
