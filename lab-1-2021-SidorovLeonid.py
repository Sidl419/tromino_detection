import numpy as np
import cv2
import os, os.path
import sys
from PIL import Image, ImageEnhance
from skimage.morphology import area_closing, area_opening
from scipy.signal import convolve2d
from tqdm import tqdm

def rotate_image(image, angle, center=None):
    """
    Поворот картинки на angle градусов вокруг center

    image - тензор формата BGR
    """
    if center is not None:
        image_center = center
    else:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def triangle_mask(filename, convs):
    """
    На основе картинки получаем сегментированное изображение
    с выделенными треугольниками

    filename - имя изображения в рабочей директории
    convs - список ядер свёртки
    """
    img = cv2.imread(filename)

    # Преобразуем BGR в HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Применяем бинаризацию по порогу, чтобы выделить цвета фишек
    mask = cv2.inRange(hsv, np.array([0,70,0]), np.array([15,255,150]))
    
    # Применяем пространственные преобразования, чтобы избавиться от шума
    mask = area_opening(mask, 2300)
    mask = area_closing(mask, 2000)

    # Применяем свёртку с треугольным ядром к полученному изображению
    conv_res = []
    for conv_num in tqdm(range(len(convs))):
        conv_res.append(convolve2d(mask.astype('int64'), convs[conv_num].astype('int64'), mode='valid'))

    conv_res = sum(conv_res)

    # Отбрасываем недостаточно большие значения свёртки
    conv_res = (conv_res > 600000000)
    conv_res = conv_res.astype('uint8')

    # Используем полученные компоненты связности для оценки числа треугольников
    num_triags, _ = cv2.connectedComponents(conv_res)

    # На основании этой информации оцениваем масштаб изображения
    scale = (mask.sum() / (num_triags - 1))

    # При помощи масштаба отбрасываем лишние объекты
    mask = area_opening(mask, scale / 380)

    return mask

def get_info(filename, mask, output):
    """
    Основная функция, выводящая количество треугольников, 
    их координаты и маркировку в текстовый файл

    filename - имя изображения в рабочей директории
    mask - сегментационная макска (вывод функции triangle_mask)
    output - файл для вывода результатов
    """
    # Перенаправляем вывод в файл
    original_stdout = sys.stdout
    sys.stdout = output

    img = Image.open(filename)

    # Повышаем яркость изображения
    enhancer = ImageEnhance.Brightness(img)
    img = np.array(enhancer.enhance(1.5))[:, :, ::-1]

    # Получаем маски для каждого треугольника на картинке
    num_triags, labels = cv2.connectedComponents(mask)
    print(num_triags - 1)

    size = np.prod(labels.shape)

    # Отдельно для каждого треугольника вычисляем координаты центра и код фишки
    for t in range(1, num_triags):
        # Координаты центра это средние значения координат маски
        cords = np.where(labels == t)
        print(int(cords[1].mean()), int(cords[0].mean()), sep=', ', end='; ')

        # При помощи маски "вырезаем" треугольник из исходного изображения
        triag_mask = (labels == t).astype('uint8')
        triag = cv2.bitwise_and(img, img, mask = triag_mask)

        # Заводим массив для записи кода фишки
        numbers = []
        
        # Вычисляем присутствие цвета каждой фишки на картинке
        hsv = cv2.cvtColor(triag, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (20,0,120), (30, 120, 255)).sum()
        green = cv2.inRange(hsv, (36,0,0), (70, 255, 255)).sum()
        yellow = cv2.inRange(hsv, (15,200,200), (36, 255, 255)).sum()
        blue = cv2.inRange(hsv, (110,0,0), (130, 255, 255)).sum()
        red = cv2.inRange(hsv, (0,190,200), (10, 255, 255)).sum() + cv2.inRange(hsv, (350,190,200), (360, 255, 255)).sum()
        
        # При достижении значения цвета определённого порога причисляем фишке некоторое число
        if red > 1000:
            numbers.append(5)

        colors = [blue, yellow, green, white]
        th1 = np.array([1100, 1000, 200, 1000]) / 623808 * size
        th2 = np.array([60000, 25000, 9000, 4000]) / 623808 * size

        for idx, val in enumerate(colors):
            if val > th1[idx]:
                numbers.append(4 - idx)
                if val > th2[idx]:
                    numbers.append(4 - idx)

        # Как показала практика, фишки только с пятёрками плохо распознаются
        if len(numbers) == 0:
            numbers.append(5)
            numbers.append(5)

        # Обрежем массив, если в него попали лишние значения
        if len(numbers) > 3:
            numbers = numbers[:3]
        
        # Дополним массив нулями (отсутсвие цвета на одной стороне) до длины три
        numbers += [0] * (3 - len(numbers))

        for i in range(len(numbers) - 1):
            print(numbers[i], end=', ')
        print(numbers[-1])
    
    # Возвращаем стандартный поток вывода
    print()
    sys.stdout = original_stdout

#------------Основная часть программы-------------------

DIR = './data'
full_dataset = []
output = open('seg_output.txt', 'w')

# На основе шаблона создаём массив треугольных ядер свёртки
conv = cv2.imread('./resources/conv_2.jpeg', 0)

_, mask = cv2.threshold(conv, 53, 255, cv2.THRESH_BINARY_INV)

conv1 = area_closing(mask, 1000)
conv2 = rotate_image(conv1, 30, (30, 70))
conv3 = rotate_image(conv1, 60, (40, 50))
conv4 = rotate_image(conv1, 90, (60, 55))

convs = [conv1, conv2, conv3, conv4]

# Основной цикл обработки изображений
for filename in sorted(os.listdir(DIR)):
    print('reading ', filename)

    # Получаем сегментационную маску для треугольных фишек
    print('processing image...')
    img = triangle_mask(os.path.join(DIR, filename), convs)
    print('done!', end='\n\n')

    get_info(os.path.join(DIR, filename), img, output)

output.close()