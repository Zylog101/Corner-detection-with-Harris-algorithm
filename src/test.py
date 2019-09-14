import cv2
import numpy as np
from operator import itemgetter
import math

WIN_1 = 'window1'
WIN_2 = 'window2'
WIN_3 = 'window3'


def detect_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corner_list = []
    scale = cv2.getTrackbarPos('scale', WIN_1)
    neighbour_size = cv2.getTrackbarPos('neighbour_size', WIN_1)
    trace_weight = cv2.getTrackbarPos('trace_weight', WIN_1)
    threshold = cv2.getTrackbarPos('threshold', WIN_1)
    trace_weight = 0.01 * trace_weight
    row, column = img.shape[0], img.shape[1]
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=scale)
    dy = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=scale)
    cornerness = 0

    # computing squares of derivatives and dx*dy
    dxx = dx * dx
    dyy = dy * dy
    dxy = dx * dy
    half_window = neighbour_size // 2
    for row_x in range(half_window, row - half_window, 1):
        for column_y in range(half_window, column - half_window, 1):
            window_ixx = dxx[row_x:row_x + neighbour_size + 1, column_y:column_y + neighbour_size + 1]
            window_iyy = dyy[row_x:row_x + neighbour_size + 1, column_y:column_y + neighbour_size + 1]
            window_ixy = dxy[row_x:row_x + neighbour_size + 1, column_y:column_y + neighbour_size + 1]
            ixx_Sum = window_ixx.sum()
            iyy_Sum = window_iyy.sum()
            ixy_Sum = window_ixy.sum()

            # Find determinant and trace, use to get corner response
            det = (ixx_Sum * iyy_Sum) - (ixy_Sum ** 2)
            trace = ixx_Sum + iyy_Sum
            cornerness = det - trace_weight * (trace ** 2)
            if cornerness > threshold:
                corner_list.append([row_x, column_y, cornerness])
    return corner_list


def perform_non_maximum_suppression(img, corner_list):
    localized_corners = []
    corner_list = sorted(corner_list, key=itemgetter(2),  reverse=True)
    print(len(corner_list))
    for val in corner_list:
        if len(localized_corners) == 0:
            localized_corners.append(val)
            continue
        for l_val in localized_corners:
            if (val[0] >= l_val[0]+16) or (val[0] <= l_val[0]-16):
                if (val[1] >= l_val[1] + 16) or (val[1] <= l_val[1] - 16):
                    # print(val[0], l_val[0], val[1], l_val[1])
                    localized_corners.append(val)
                    break
    print(len(localized_corners))
    return localized_corners


def reprocess_corners(scale, neighbour_size, trace_weight, threshold):
    first_image = cv2.imread('11.jpg')
    second_image = cv2.imread('22.jpg')
    cv2.imshow(WIN_1, first_image)
    cv2.imshow(WIN_2, second_image)

    first_image_corners = detect_corners(first_image)
    second_image_corners = detect_corners(second_image)

    localized_img1_corners = perform_non_maximum_suppression(first_image, first_image_corners)
    localized_img2_corners = perform_non_maximum_suppression(second_image_corners, second_image_corners)

    display_corners(localized_img1_corners, 100, first_image, localized_img2_corners, second_image)

    print(scale, neighbour_size, trace_weight, threshold)


def nothing(param):
    scale = cv2.getTrackbarPos('scale', WIN_1)
    neighbour_size = cv2.getTrackbarPos('neighbour_size', WIN_1)
    trace_weight = cv2.getTrackbarPos('trace_weight', WIN_1)
    threshold = cv2.getTrackbarPos('threshold', WIN_1)

    reprocess_corners(scale, neighbour_size, trace_weight, threshold)


def initialize():
    first_img = cv2.imread('11.jpg')
    second_img = cv2.imread('22.jpg')
    cv2.imshow(WIN_1, first_img)
    cv2.imshow(WIN_2, second_img)
    cv2.createTrackbar('scale', WIN_1, 5, 5, nothing)
    cv2.createTrackbar('neighbour_size', WIN_1, 3, 5, nothing)
    cv2.createTrackbar('trace_weight', WIN_1, 4, 15, nothing)
    cv2.createTrackbar('threshold', WIN_1, 10, 20, nothing)

    return first_img, second_img


def display_corners(localized_img2_corners, perc_pixel, second_image, localized_img1_corners, first_image):
    i = 0
    perc_pixel = ((second_image.shape[0] + second_image.shape[1]) * perc_pixel)//100
    for value in localized_img2_corners:
        if i > perc_pixel:
            break
        second_image.itemset((value[0], value[1], 0), 0)
        second_image.itemset((value[0], value[1], 1), 0)
        second_image.itemset((value[0], value[1], 2), 255)
        i += 1
    cv2.imshow(WIN_2, second_image)

    i = 0
    for value in localized_img1_corners:
        if i > perc_pixel:
            break
        first_image.itemset((value[0], value[1], 0), 0)
        first_image.itemset((value[0], value[1], 1), 0)
        first_image.itemset((value[0], value[1], 2), 255)
        i += 1
    cv2.imshow(WIN_1, first_image)
    cv2.waitKey()


def compute_feature_vector(corner_list, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    neighbour_size = cv2.getTrackbarPos('neighbour_size', WIN_1)
    win_w = neighbour_size * 2 + 1
    win_h = neighbour_size * 2 + 1

    degree_histogram = np.zeros((len(corner_list), 9, 1), dtype=np.uint8)

    for i in range(0, len(corner_list)):
        im = cv2.getRectSubPix(gray, (win_w, win_h), (corner_list[i][0], corner_list[i][1]))
        dx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)

        angle = np.arctan2(dy, dx)
        for j in range(0, 9):
            degree_histogram[i][j] = (((math.pi / 4 * (j - 4)) <= angle) & (angle < (math.pi / 4 * (j - 3)))).sum()

    return degree_histogram


def plot_corners(img1, img2, corner_list1, corner_list2, histogram1, histogram2):
    corner_list1 = np.int0(corner_list1)
    corner_list2 = np.int0(corner_list2)

    for i in range(0, corner_list1.shape[0]):
        cv2.rectangle(img1, (corner_list1[i, 0] - 18, corner_list1[i, 1] - 18), (corner_list1[i, 0] + 18, corner_list1[i, 1] + 18),
                      [0, 0, 0])
    for j in range(0, corner_list2.shape[0]):
        cv2.rectangle(img2, (corner_list2[j, 0] - 18, corner_list2[j, 1] - 18), (corner_list2[j, 0] + 18, corner_list2[j, 1] + 18),
                      [0, 0, 0])

    cno = 1
    min_distance = 0
    min_index = 0

    for i in range(0, len(corner_list1)):
        for j in range(0, len(corner_list1)):
            distance = np.sum((histogram1[i] - histogram2[j]) * (histogram1[i] - histogram2[j]), None)
            if j == 0:
                min_distance = distance
                min_index = j
            else:
                if distance < min_distance:
                    min_distance = distance
                    min_index = j

        cv2.putText(img1, str(cno), (corner_list1[i, 0], corner_list1[i, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img2, str(cno), (corner_list1[min_index, 0], corner_list1[min_index, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cno = cno + 1

    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)


def main():
    first_image, second_image = initialize()
    cv2.waitKey()
    first_image_corners = detect_corners(first_image)
    second_image_corners = detect_corners(second_image)

    localized_img1_corners = perform_non_maximum_suppression(first_image, first_image_corners)
    localized_img2_corners = perform_non_maximum_suppression(second_image_corners, second_image_corners)

    display_corners(localized_img1_corners, 100, first_image, localized_img2_corners, second_image)


if __name__ == '__main__':
    main()
