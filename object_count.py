import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

original = cv.imread('image/image4.jpeg')

f, axes = plt.subplots(2, 3)

thresh_value = 100
dil_ero_value_x = 15
dil_ero_value_y = 15


def on_press(event):
    global thresh_value, dil_ero_value_x, dil_ero_value_y
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

    show()
    plt.show()


def show():
    # Convert image in grayscale
    gray_im = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

    # Contrast adjusting with histogramm equalization
    # gray_equ = cv.equalizeHist(gray_im)

    # Local adaptative threshold
    # thresh = cv.adaptiveThreshold(gray_im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 255, thresh_value)
    t_val, thresh = cv.threshold(gray_im, thresh_value, 255, cv.THRESH_BINARY_INV)

    # Dilatation et erosion
    kernel = np.ones((dil_ero_value_x, dil_ero_value_y), np.uint8)
    img_dilation = cv.dilate(thresh, kernel, iterations=1)
    img_erode = cv.erode(img_dilation, kernel, iterations=1)
    # clean all noise after dilatation and erosion
    img_erode = cv.medianBlur(img_erode, ksize=7)

    # Labeling
    ret, labels = cv.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    axes[0][0].set_title('Origin')
    axes[0][0].imshow(original)
    axes[0][1].set_title('Grayscale image')
    axes[0][1].imshow(gray_im, cmap="gray", vmin=0, vmax=255)
    # axes[0][2].set_title('Histogram equalization')
    # axes[0][2].imshow(gray_equ, cmap="gray", vmin=0, vmax=255)
    axes[1][0].set_title('Threshold: ' + str(thresh_value))
    # axes[1][0].set_title('Local adaptive threshold: ' + str(thresh_value))
    axes[1][0].imshow(thresh, cmap="gray", vmin=0, vmax=255)
    # axes[1][0].set_title('Dilatation')
    # axes[1][0].imshow(img_dilation, cmap="gray", vmin=0, vmax=255)
    axes[1][1].set_title('Erosion: ' + str(dil_ero_value_x) + '-' + str(dil_ero_value_y))
    axes[1][1].imshow(img_erode, cmap="gray", vmin=0, vmax=255)
    axes[1][2].imshow(labeled_img)
    axes[1][2].set_title('Objects counted:' + str(ret - 1))


cid = f.canvas.mpl_connect('key_press_event', on_press)
show()
plt.show()
