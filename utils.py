from cv2 import bitwise_or
import numpy as np
import matplotlib.pyplot as plt
from approxes import Dot
import cv2

def marker_mean(markers):
    return sum(markers.get_points())/len(markers.get_points())

def sort_markers(markers,axis=0,reverse=False):
    markers = sorted(markers, key=lambda marker: marker.C[axis], reverse=reverse)
    return markers

def sort_markers_by_side(markers,side):
    markers = [marker for marker in markers if marker.side == side]
    return sort_markers(markers,1,True)

def dot_mean(dots):
    return sum(dots)/len(dots) 

def rotate(vector, theta) -> np.ndarray:
    """
    reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    :param vector: list of length 2 OR
                   list of list where inner list has size 2 OR
                   1D numpy array of length 2 OR
                   2D numpy array of size (number of points, 2)
    :param theta: rotation angle in degree (+ve value of anti-clockwise rotation)
    :param rotation_around: "vector" will be rotated around this point, 
                    otherwise [0, 0] will be considered as rotation axis
    :return: rotated "vector" about "theta" degree around rotation
             axis "rotation_around" numpy array
    """
    vector = np.array(vector)

    if vector.ndim == 1:
        vector = vector[np.newaxis, :]

    vector = vector.T

    theta = np.radians(theta)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta) , 0],
        [0            , 0             , 1]
    ])

    output: np.ndarray = (rotation_matrix @ vector).T

    return output.squeeze()


def hsv_mask(img):

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # convert to hsv
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # threshold using inRange
    range1 = (44,97,0)
    range2 = (148,255,255)
    mask1 = cv2.inRange(hsv,range1,range2)

    range3 = (0,100,0)
    range4 = (179,255,255)
    mask2 = cv2.inRange(hsv,range3, range4)
    mask2 = 255 - mask2
    mask1 = 255 - mask1

    # cv2.imshow("mask1",mask1)
    # cv2.imshow("mask2",mask2)

    mask = bitwise_or(mask1,mask2)

    # apply morphology closing and opening to mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # make mask 3 channel
    # mask = cv2.merge([mask,mask,mask])

    # invert mask
    # mask_inv = 255 - mask


    # cv2.namedWindow('img_masked',cv2.WINDOW_NORMAL)
    # # cv2.namedWindow('img_new_masked',cv2.WINDOW_NORMAL)
    # cv2.imshow("img_masked", mask_inv)
    # cv2.imshow("img_new_masked", paddedMask)
    return mask

def fill_holes(mask):
    for i in range(mask.shape[1]):
        if mask[0,i] == 0 :
            cv2.floodFill(mask, (i,0), 255, 0, 10, 10);
        
        if mask[mask.shape[0] -1, i] == 0 :
            cv2.floodFill(mask, (i, mask.shape[0]-1), 255, 0, 10, 10);
        
    
    for i in range(mask.shape[0]):
        if (mask[i, 0] == 0) :
            cv2.floodFill(mask,(0, i), 255, 0, 10, 10);
        
        if (mask[i, mask.shape[1]-1] == 0) :
            cv2.floodFill(mask, (mask.shape[1]-1, i), 255, 0, 10, 10);
        
    return mask

def edge_mask(frame):
    BLUR = 5
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 120
    MASK_DILATE_ITER = 20
    MASK_ERODE_ITER = 20

    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(frame, (11, 11), 0)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[2:]
    
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    mask = np.zeros(edges.shape)
    for c in contour_info:
        
        cv2.fillConvexPoly(mask, c[0], (255))

            # cv2.drawContours(frame,c[0],-1,(255,255,0),3)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    return mask
    
    



def remove_background(frame,fill_holes_flag=0):
    
    if fill_holes_flag:

        des = hsv_mask(frame)
        contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des,[cnt],0,255,-1)
        mask = des


    else:
        mask = hsv_mask(frame)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    return mask


def check_straight(aprox,width,tolerance_coeff = 1./25):
    i = 0

    for point in aprox:
        point = point.flatten()
        center = width /2
        tolerance = width * tolerance_coeff
        # print(center, tolerance, "centol")
        if point[0] < center + tolerance and point[0] > center - tolerance:
            i += 1 
    if i == len(aprox):
        return True
    else:
        return False

def check_sides(aprox, height, width, side, tolerance_coeff= 1./7):
    i = 0

    for point in aprox:
        point = point.flatten()
        center = height /2
        tolerance = height * tolerance_coeff
        # print(center, tolerance, "centol")
        if point[1] < center + tolerance and point[1] > center - tolerance:
            if side == "left":
                if point[0] < width / 2:
                    i += 1 
            else:
                if point[0] > width / 2:
                    i += 1
    if i == len(aprox):
        return True
    else:
        return False

def check_rectangle(cnt, width, tol=4):
    rect = cv2.minAreaRect(cnt)
    if abs(rect[2]) > 88:
        print(rect[2])
        # print(rect)

        # box = cv2.boxPoints(rect)

        # box = np.int0(box)
        # print(box)

        # if rect[0][0] < width / 2:
        #     return ("left" , box)

        # elif rect[0][0] > width / 2:
        #     return ("right" , box)
        # else:
        #     return ("none", box)
        return True
    else:
        return False
        # print(type(box[0][0]))
        # cv2.drawContours(img,[box],0,(255,0,0),2)
    # below1 = aprox[0].flatten()
    # below2 = aprox[1].flatten()
    # up1 = aprox[3].flatten()
    # up2 = aprox[4].flatten()
    # tol = 15
    # i = 0
    # if abs(below1[1] - below2[1]) < tol:
    #     i += 1
    # if abs(up1[1] - up2[1]) < tol:
    #     i += 1
    # if i == 2:
    #     return True
    # else:
    #     return False

def check_bottom_half(aprox, height, margin):
    i = 0
    for point in aprox:
        point = point.flatten()
        center = height /2
        # print(center + margin * height, "centol")
        if point[1] > center + margin * height:
            i += 1 
    if i == len(aprox):
        return True
    else:
        return False

def choose_binary_cadidate(binary_candiates):
    if len(binary_candiates) == 0:
        return [1,1,1,1,1]
    sums = np.array([sum(b) for b in binary_candiates])
    # print("sums", sums)
    best = np.argmin(sums)
    
    return binary_candiates[best]
    

def compute_angle(ellipses,ellipse_ranges):
    binary_angle = [1,1,1,1,1]
    dots = []
    for ellipse in ellipses:
        for i in range(5):
            if int(ellipse[0][1]) in range(int(ellipse_ranges[i+1]),int(ellipse_ranges[i])):
                # print("Im here")
                # print("debug",ellipse[0][1], ellipse_ranges[i+1], ellipse_ranges[i])
                binary_angle[i] = 0
                dot = Dot(ellipse,4-i)
                dots.append(dot)
    return binary_angle, dots

def show_planer_points(points):
    # plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(points[:,0], points[:,1])
    for i,point in enumerate(points):
        ax.annotate(str(i),(point[0], point[1]))
    plt.show()


# import cv2
# cap = cv2.VideoCapture("data/obj01.mp4") 

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = remove_background(frame)
#     cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     # else:
#     #     cv2.waitKey(0)

# cap.release()
# # result.release()
# cv2.destroyAllWindows()

def voxel_position_to_index(point, initial):
    initial_x, initial_y, initial_z = initial
    x = int(point[0] - initial_x)
    y = int(point[1] - initial_y)
    z = int(point[2] - initial_z)
    return x,y,z
