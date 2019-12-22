import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

original = cv.imread('image/2_chicks.jpg')

f, axes = plt.subplots(2, 3)

thresh_value = 165
block_size = 21
dil_ero_value_x = 15
dil_ero_value_y = 15
ksize_1 = 1
ksize_2 = 7


def on_press(event):
    global thresh_value, dil_ero_value_x, dil_ero_value_y, block_size, ksize_1, ksize_2
    if event.key == 'j':
        thresh_value -= 5
    if event.key == 'k':
        thresh_value += 5
    if event.key == 'n':
        dil_ero_value_x -= 2
    if event.key == 'm':
        dil_ero_value_x += 2
    if event.key == ',':
        dil_ero_value_y -= 2
    if event.key == '.':
        dil_ero_value_y += 2
    if event.key == '[':
        block_size -= 2
    if event.key == ']':
        block_size += 2
    if event.key == 'z':
        ksize_1 -= 2
    if event.key == 'x':
        ksize_1 += 2
    if event.key == 'c':
        ksize_2 -= 2
    if event.key == 'v':
        ksize_2 += 2

    show()
    plt.show()


def show():
    axes[0][0].set_title('Origin')
    axes[0][0].imshow(original)

    # Convert image in grayscale
    img = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    axes[0][1].set_title('Grayscale image')
    axes[0][1].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with histogramm equalization
    img = cv.equalizeHist(img)
    axes[0][2].set_title('Histogram equalization')
    axes[0][2].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Local adaptative threshold
    # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size,
    #                               thresh_value)
    t_val, img = cv.threshold(img, thresh_value, 255, cv.THRESH_BINARY)
    img = cv.medianBlur(img, ksize=ksize_1)
    axes[1][0].set_title('blocksize([]) = ' + str(block_size) + '. ' + 'C(jk) = ' + str(thresh_value) + '\nksize(zx): ' + str(ksize_1))
    axes[1][0].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Dilatation et erosion
    kernel = np.ones((dil_ero_value_x, dil_ero_value_y), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    img = cv.erode(img, kernel, iterations=1)
    # clean all noise after dilatation and erosion
    img = cv.medianBlur(img, ksize=ksize_2)
    axes[1][1].set_title('D&E (nm,.): ' + str(dil_ero_value_x) + '-' + str(dil_ero_value_y) + '\nksize(cv): ' + str(ksize_2))
    axes[1][1].imshow(img, cmap="gray", vmin=0, vmax=255)

    # Labeling
    ret, labels = cv.connectedComponents(img)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    img = cv.merge([label_hue, blank_ch, blank_ch])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    img[label_hue == 0] = 0
    axes[1][2].imshow(img)
    axes[1][2].set_title('Objects counted:' + str(ret - 1))


cid = f.canvas.mpl_connect('key_press_event', on_press)
show()
plt.show()
