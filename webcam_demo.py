import tensorflow as tf
import json
import math
import cv2
import time
import argparse
import concurrent.futures
import posenet
import keyboard
import sys
import numpy as np
from threading import Thread
from slugify import slugify

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    # tf.config.threading.set_inter_op_parallelism_threads(0)
    # tf.config.threading.set_intra_op_parallelism_threads(0)
    # print(tf.config.threading.get_inter_op_parallelism_threads())
    # print(tf.config.threading.get_intra_op_parallelism_threads())
    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)



        start = time.time()
        frame_count = 0
        recording = True
        # ret,frame1 = cap.read()
        # ret,frame2 = cap.read()
        file_content = []
        while True:
            # diff = cv2.absdiff(frame1,frame2)
            # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray,(15,15),0)
            # _, thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
            # dilated = cv2.dilate(thresh,None, iterations=3)
            # contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # # if(len(contours)>0):
            # #     print("One:")
            # #     print(dir(contours[0]))
            # #     print("One it is.")
            # for contour in contours:
            #     (x,y,w,h) = cv2.boundingRect(contour)
            #     if(cv2.contourArea(contour)>400):
            #         continue
            #     cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            # # cv2.drawContours(frame1,contours, -1,(0,255,0),2)
            # cv2.imshow("feed",frame1)
            # frame1 = frame2
            # ret, frame2 = cap.read()                 
            input_image, display_image, output_scale = posenet.read_cap(cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )                
                    
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)                
            keypoint_coords *= output_scale

                    # TODO this isn't particularly fast, use GL for drawing and display someday...
            # print("\n ===================================== \n")
            
               
            img = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.15)
        
                

            cv2.imshow('posenet', img)

            frame_count += 1

            if(recording):
                normalize_poses(keypoint_coords)
                results = json.dumps({
                "timestamp":time.time() - start,
                "pose_scores":pose_scores.tolist(),
                "keypoint_scores":keypoint_scores.tolist(),
                "scores": keypoint_scores.size,
                "keypoint_coords":normalize_poses(keypoint_coords),
                "coords": keypoint_coords.size
                })
                file_content.append(results)
                file_content = file_content[-30:]
            
            if cv2.waitKey(1) & keyboard.is_pressed('w'):
                print('you pressed w - service it was!')
                time.sleep(0.5)
                path = "collected/serves/"
                filename = str(slugify("s-"+str(time.time()))+".txt")
                x = Thread(target=save_to_file, args=(str(path+filename),str(file_content)))
                x.start()         
                x.join()
                file_content = []

            if cv2.waitKey(1) & keyboard.is_pressed('d'):
                print('you pressed d - forehand it was!')
                time.sleep(0.5)
                path = "collected/forehand/"
                filename = str(slugify("f-"+str(time.time()))+".txt")
                x = Thread(target=save_to_file, args=(str(path+filename),str(file_content)))
                x.start()         
                x.join()
                file_content = []
            
            if cv2.waitKey(1) & keyboard.is_pressed('a'):
                print('you pressed a - backhand it was!')
                time.sleep(0.5)
                path = "collected/backhand/"
                filename = str(slugify("b-"+str(time.time()))+".txt")
                x = Thread(target=save_to_file, args=(str(path+filename),str(file_content)))
                x.start()         
                x.join()
                file_content = []
            

            if cv2.waitKey(1) & keyboard.is_pressed('q'):
                print('you pressed q - quitting!')
                cv2.destroyAllWindows()
                break
                
    print('Average FPS: ', frame_count / (time.time() - start))
    return 0

def my_function(toPrint):
    print(toPrint)

def save_to_file(filename,data):
    file = open(filename,'w')
    file.write(data) 
    file.close() 


def find_middle(left,right):
    x = (left[0]+right[0])/2.0
    y = (left[1]+right[1])/2.0
    return [x,y]
def find_distance(pointA,pointB):
    dist = math.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)  
    return dist  

def normalize_poses(poses):
    leftShoulderCords = poses[0][5]
    rightShoulderCords = poses[0][6]
    middleShoulderPoint = find_middle(leftShoulderCords,rightShoulderCords)
    leftHipCords = poses[0][11]
    rightHipCords = poses[0][12]
    middleHipPoint = find_middle(leftHipCords,rightHipCords)

    armHipDistance = find_distance(middleHipPoint,middleShoulderPoint);

    normalized = []
    for pose in poses[0]:
        normalized.append(
            [(pose[0]-middleHipPoint[0])/armHipDistance,
             (pose[1]-middleHipPoint[1])/armHipDistance]
            )
    return normalized

if __name__ == "__main__":
    main()