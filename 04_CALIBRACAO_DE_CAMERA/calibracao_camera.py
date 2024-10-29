import cv2 as cv
import numpy as np
import os
import glob

PATH = "/home/andre/Documents/Git/VISAO_ROBOTICA_EGM0008/04_CALIBRACAO_DE_CAMERA/"
PATH_IMG = "/home/andre/Documents/Git/VISAO_ROBOTICA_EGM0008/IMAGENS/"

def cm_to_inch(value):
    return value/2.54

def rad_to_angle(value):
    return value*180/np.pi

def angle_to_rad(value):
    return value*np.pi/180

#=================================================
#  MAIN
#=================================================

def main():

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    posX = posY = 100

    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera nao encontrada: {cap}")
        exit()
    
    cnt = 0
    while True:
        # Capture frame-by-frame
        ret_frame_read, frame = cap.read()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        # Find the chess board corners
        ret_findChess, corners = cv.findChessboardCorners(gray, (7,6), None)
 
        # if frame is read correctly ret is True
        if ret_findChess:
            print(f'Chess')

            cv.imwrite(f'{PATH}chessPattern_{cnt}.jpg', gray)
            cnt += 1

            if cnt == 9:
                break

        # Display the resulting frame
        cv.imshow('frame', gray)
        cv.moveWindow('frame',posX, posY)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    images = glob.glob(PATH+"*.jpg")
    # print(np.sort(images))

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
 
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
 
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            
            cv.imshow('img', img)
            cv.moveWindow('img', posX, posY)

            cv.waitKey(500)
        
    cv.destroyAllWindows()

    # So para pegar o tamanho da imagem
    img = cv.imread(PATH+"chessPattern_0.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # So para pegar o tamanho da imagem

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera( objpoints, 
                                                                     imgpoints, 
                                                                     gray.shape[::-1],
                                                                     None, None)
    
    print(f'Coefs. Intrisecos:\n{cameraMatrix}\n')

    undistorted_img = cv.undistort( img, 
                                   cameraMatrix,
                                    distCoeffs)
    
    cv.imshow("undistorted_img", undistorted_img)
    cv.moveWindow("undistorted_img", posX, posY)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

#=================================================
#  MAIN
#=================================================

if __name__ == "__main__":
    main()