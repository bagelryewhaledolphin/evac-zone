
import cv2
import numpy as np
from time import sleep
from collections import namedtuple
from picamera2 import Picamera2
from ultralytics import YOLO
from newMotors import stopRobot, moveRobotFwdOrBwd, moveRobotRight, moveRobotLeft
from clawFunctions import liftClaw, moveClawDown, openClawHands, closeClawHands, openDoor, closeDoor





identifiedVictim = namedtuple('identifiedVictim', ['x', 'y', 'perimeter', 'color'])




def get_average_green_position(frame):
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define green range and create mask
    # These values cover most typical green shades
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 3. Find coordinates of all green pixels
    # np.where returns (rows, cols)
    y_coords, x_coords = np.where(mask > 0)

    # 4. Calculate average position
    if len(x_coords) > 0:
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        return (int(avg_x), int(avg_y))
    
    return None # No green pixels found



def get_average_red_position(frame):
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Define green range and create mask
    # These values cover most typical green shades
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([20, 255, 255])
    
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask1, mask2)

    # 3. Find coordinates of all green pixels
    # np.where returns (rows, cols)
    y_coords, x_coords = np.where(mask > 0)

    # 4. Calculate average position
    if len(x_coords) > 0:
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        return (int(avg_x), int(avg_y))
    
    return None # No green pixels found






# returns the closest object

def get_closest_object(frame, model, confidence_threshold=0.25, show_window=True):

    

    closest_target = None

    max_perimeter = -1

    

    results = model(frame, stream=True, conf=confidence_threshold)

    

    # deep copy for display

    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if show_window else None

    

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        clss = result.boxes.cls.cpu().numpy()

        

        for box, cls in zip(boxes, clss):

            x1, y1, x2, y2 = map(int, box)

            

            # Map numeric class IDs to your color strings

            class_id = int(cls)

            color_name = "black" if class_id == 0 else "silver" if class_id == 1 else "unknown"


            # Dimensions and spatial tracking math

            width = x2 - x1

            height = y2 - y1

            center_x = int(x1 + (width / 2))

            center_y = int(y1 + (height / 2))

            perimeter = 2 * (width + height)

            

            # Build current target namedtuple

            current_target = identifiedVictim(x=center_x, y=center_y, perimeter=perimeter, color=color_name)

            

            # Tracking check: Is this object closer than previous ones?

            if perimeter > max_perimeter:

                max_perimeter = perimeter

                closest_target = current_target


            # Draw everything on the frame if window output is turned on

            if show_window:

                box_color = (0, 0, 255) if color_name == "black" else (0, 255, 0)

                # Outer rectangle box

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                # Center point dot

                cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)

                # Text specs label

                label = f"{color_name.upper()} (P:{perimeter}px)"

                cv2.putText(display_frame, label, (x1, y1 - 10), 

                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


    # Render window frame externally if activated

    if show_window:

        # Highlight the tracked target on frame if one exists

        # if closest_target:

        #     cv2.putText(display_frame, "► TRACKING CLOSEST ◄", (20, 40), 

        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            

        cv2.imshow("Robot Tracking Stream", display_frame)

        cv2.waitKey(1)  # Must call waitKey to refresh the window system UI thread


    return closest_target







def pickup():

    openClawHands()
    sleep(1.0)
    moveClawDown()
    sleep(1.0)
    liftClaw()
    sleep(1.0)
    closeClawHands()
    sleep(1.0)
    liftClaw()
    sleep(1.0)
    openClawHands()
    sleep(1.0)






##########################################

# Loop

##########################################


### Camera ###

picam2 = Picamera2(1)

config = picam2.create_preview_configuration(main={"format":"BGR888", "size":(640,640)}) # ,transform=Transform(hflip=1, vflip=1)

picam2.start(config)





adjustError = 40




### State ###


state = "looking"

lookingfor = "silver1" # silver2 and black


'''

Looking --> *Found* --> Adjust --> *In position* --> Pick up --> Repeat for Silver2 -->


Find green area --> Adjust --> Drop off victims --> Looking


'''








model = YOLO("/home/firstenergyrobot/CRC/best_ncnn_model")


try:

    while True:

        frame = picam2.capture_array()

        frame = np.flipud(frame)


        if state == "looking":
            
            target = get_closest_object(frame, model, confidence_threshold=0.30, show_window=True)

            if target is not None:

                state = "adjust"

            # move 
        

        elif state == "adjust":

            target = get_closest_object(frame, model, confidence_threshold=0.30, show_window=True)

            if target is None:
                state = "looking"
                continue

            targetx,targety,targetp = target.x, target.y, target.perimeter

            if target.perimeter > 1400 and target.color == "silver" and lookingfor != "black" and target.x > 320-adjustError and target.x < 320+adjustError:
                state = "pickup"
                
            elif target.color == "silver" and lookingfor != "black" and target.x > 320-adjustError and target.x < 320+adjustError:
                moveRobotFwdOrBwd("fwd")

            elif target.color == "silver" and lookingfor != "black" and (target.x <= 320-adjustError or target.x >= 320+adjustError):
                if target.x <= 320-adjustError:
                    moveRobotLeft()
                else:
                    moveRobotRight()

            elif target.perimeter > 1400 and target.color == "black" and lookingfor == "black" and target.x > 320-adjustError and target.x < 320+adjustError:
                state = "pickup"

            elif target.color == "Black" and lookingfor == "black" and target.x > 320-adjustError and target.x < 320+adjustError:
                moveRobotFwdOrBwd("fwd")

            elif target.color == "Black" and lookingfor == "black" and (target.x <= 320-adjustError or target.x >= 320+adjustError):
                if target.x <= 320-adjustError:
                    moveRobotLeft()
                else:
                    moveRobotRight()

            else:
                print("error in adjust state")

            
        elif state == "pickup":

            pickup()
            if lookingfor == "silver1":
                lookingfor = "silver2"
                state = "looking"
            elif lookingfor == "silver2":
                lookingfor = "black"
                state = "adjustgreen"
            elif lookingfor == "black":
                state = "adjustred"
            else:
                print("error in pickup state")

        
        elif state == "adjustgreen":

            greenPos = get_average_green_position(frame)

            if greenPos is None:
                moveRobotRight()
            else:
                if greenPos[0] > 320-adjustError and greenPos[0] < 320+adjustError:
                    state = "findgreen"
                elif greenPos[0] <= 320-adjustError:
                    moveRobotLeft()
                else:
                    moveRobotRight()


        elif state == "findgreen":

            greenPos = get_average_green_position(frame)

            if greenPos is None:
                state = "adjustgreen"
            else:
                moveRobotFwdOrBwd("fwd")
                sleep(3)
                state = "dropoff"

        elif state == "dropoff":
            moveRobotFwdOrBwd("bwd")
            sleep(0.5)
            moveRobotRight()
            sleep(1.5)
            moveRobotFwdOrBwd("bwd")
            sleep(0.5)
            stopRobot()
            openDoor()
            moveRobotFwdOrBwd("fwd")
            sleep(0.3)
            moveRobotFwdOrBwd("bwd")
            sleep(2.0)
            closeDoor()

            if lookingfor == "silver2":
                state = "looking"
                lookingfor = "black"
            else:
                state = "exit"

        else:

            print("invalid state")

            








except KeyboardInterrupt:

    print("\nShutting down system...")

finally:

    picam2.stop()

    cv2.destroyAllWindows()
