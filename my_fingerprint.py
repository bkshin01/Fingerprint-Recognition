import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os
import math
import random

def filtering(src, kernel, ks):
    filtered = cv2.filter2D(src, -1, kernel)
    return filtered.astype(np.uint8)

def detect_minutiae(skel):
    kernel = np.ones((3, 3), dtype = np.float32)
    row, col = skel.shape
    mask = (skel == 255)
    
    skel_float = skel.astype(np.float32)
    filtered = cv2.filter2D(skel_float, -1, kernel)
    
    endpoints_indices = np.where((filtered == 510) & mask)
    endpoints = np.column_stack(endpoints_indices)

    bifurcations_indices = np.where((filtered == 1020) & mask)
    bifurcations = np.column_stack(bifurcations_indices)

    minutiae = np.concatenate(
        [np.column_stack([endpoints, np.zeros(len(endpoints))]),
         np.column_stack([bifurcations, np.full(len(bifurcations), 255)])]
    )

    return endpoints, bifurcations, minutiae

def excludeBorderPoints(points, image):
    image = np.array(image)
    
    def is_zero_section(start, end, h):
        return np.all(image[h, start+1:end] == 0)
    
    def is_zero_section_for_y(start, end, w):
        return np.all(image[start+1:end, w] == 0)
    
    points = np.array(points)
    
    mask1 = (points[:, 1] <= 127) & (np.vectorize(is_zero_section)(0, points[:, 1], points[:, 0]))
    mask2 = (points[:, 1] > 127) & (np.vectorize(is_zero_section)(points[:, 1], 256, points[:, 0]))
    
    result = points[~(mask1 | mask2)]
    
    mask3 = (result[:, 0] <= 127) & (np.vectorize(is_zero_section_for_y)(0, result[:, 0], result[:, 1]))
    mask4 = (result[:, 0] > 127) & (np.vectorize(is_zero_section_for_y)(result[:, 0], 256, result[:, 1]))
    
    result2 = result[~(mask3 | mask4)]
    
    return result2.tolist()

def mark_minutiae(src, endpoints, bifurcations):
    color_src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    
    for (x, y) in endpoints:
        cv2.circle(color_src, (y, x), 1, (0, 0, 255), 1)  # endpoint 파랑

    for (x, y) in bifurcations:
        cv2.circle(color_src, (y, x), 1, (255, 0, 0), 1)  # intersections 빨강

    return color_src

def get_fp_feature(image, flg_show=True):
    ks = 3
    kernel = np.ones((ks, ks)) / (ks*ks)
    blurred1 = filtering(image, kernel, ks)
    blurred2 = filtering(blurred1, kernel, ks)

    block_size = 9
    C = 5
    blur_bin = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    blur_bin = np.max(blur_bin) - blur_bin
    
    morph_blur_bin = cv2.morphologyEx(blur_bin, cv2.MORPH_CLOSE, None)
    
    skeleton0 = skeletonize(morph_blur_bin // 255)
    skeleton = skeleton0.astype(np.uint8) * 255

    end0, bif0, minutiae = detect_minutiae(skeleton)
    end = excludeBorderPoints(end0, skeleton)
    bif = excludeBorderPoints(bif0, skeleton)
    
    if flg_show:
        marked_image = mark_minutiae(skeleton, end, bif)
        
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(morph_blur_bin, cmap='gray')
        
        plt.subplot(2, 2, 3)
        plt.imshow(skeleton, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(marked_image, cmap='gray')

    return end, bif, minutiae

def match_finger(A, B, threshold, flg_show=False, img_A=None, img_B=None):
    matches = []
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    dist = 0

    for xA, yA in A:
        distances = np.sqrt(np.sum((B - np.array([xA, yA]))**2, axis=1))
        closest_point_index = np.argmin(distances)
        min_distance = distances[closest_point_index]
        
        if min_distance <= threshold:
            matches.append((xA, yA, B[closest_point_index][0], B[closest_point_index][1]))
            dist += min_distance
        
    match_score = 100 * (len(matches)) / ((len(A) + len(B)) / 2)
    matches = np.array(matches)

    if flg_show:
        img_query = cv2.cvtColor(img_A, cv2.COLOR_GRAY2BGR)
        img_train = cv2.cvtColor(img_B, cv2.COLOR_GRAY2BGR)
        match_img = cv2.hconcat([img_query, img_train])

        for i in range(len(matches)):
            xA = matches[i][0]
            yA = matches[i][1]
            xB = matches[i][2]
            yB = matches[i][3]
            
            pointA = (int(xA), int(yA))
            pointB = (int(xB), int(yB))
            rand_R = random.randint(50, 200)
            rand_G = random.randint(50, 200)
            rand_B = random.randint(50, 200)
            match_color = (rand_R, rand_G, rand_B)

            converted_pointA = (pointA[1], pointA[0])
            converted_pointB = (pointB[1]+256, pointB[0])
            
            cv2.circle(match_img, converted_pointA, 1, match_color, 3)
            cv2.circle(match_img, converted_pointB, 1, match_color, 3)
            cv2.line(match_img, converted_pointA, converted_pointB, match_color, 1)
    
            plt.imshow(match_img)
    
    return dist, match_score

def find_match(query_image, list_train, match_threshold):
    test_end, test_bif, _ = get_fp_feature(query_image, flg_show=False)
    feat_query = np.concatenate([test_end, test_bif]).astype(np.uint8)

    best_score = 0
    best_dist = 0
    best_img = 0
    
    for t in list_train:
        basename = os.path.basename(t)
        img_train2 = cv2.imread(t, 0)
        
        db_x_end, db_x_bif, _ = get_fp_feature(img_train2, False)
        feat_db_x = np.concatenate([db_x_end, db_x_bif]).astype(np.uint8)
        
        dist, match_score = match_finger(feat_query, feat_db_x, match_threshold, False, img_A=None, img_B=None)
        
        if match_score > best_score:
            best_score = match_score
            best_dist = dist
            best_img = basename
            
    return best_img, best_score, best_dist

def calculateAccuracy(pred, label):
    length = len(pred)
    success = []
    fail = []
    for i in range(length):
        if pred[i] == label[i]:
            success.append(i)
        else:
            fail.append(u)
        
    accuracy = 100 * len(success) / (len(success) + len(fail))
    
    print(f"Success: {len(success)} out of {len(pred)}\nAccuracy: {accuracy}")
    