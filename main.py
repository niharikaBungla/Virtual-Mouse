import cv2
import numpy as np
import mediapipe as mp
import autopy
import pyautogui
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER

wCam, hCam = 640, 480 
frameR = 100 
smoothening = 5

plocX, plocY = 0, 0 #prev cursor loc
clocX, clocY = 0, 0 #current
scroll_threshold = 40 
cursor_stopped = False 

# webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam) 
cap.set(4, hCam)

# MediaPipe Hands
mpHands = mp.solutions.hands # hand tracking module
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7) #hand detection
mpDraw = mp.solutions.drawing_utils #draw landmark

# Get screen size to map webcam cordinates with screen coordinate
wScr, hScr = autopy.screen.size() 

# Define functions
def changesystemvolume(pinchlv):
    devices = AudioUtilities.GetSpeakers() #an object that represents the audio device
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None) #activates the audio endpoint interface for controlling volume
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    currentVolumeLv = volume.GetMasterVolumeLevelScalar()
    currentVolumeLv += pinchlv / 50.0
    currentVolumeLv = max(0.0, min(1.0, currentVolumeLv))
    volume.SetMasterVolumeLevelScalar(currentVolumeLv, None)

def scrollVertical(pinchlv):
    if pinchlv < scroll_threshold:
        pyautogui.scroll(-120)  # Scroll down
    else:
        pyautogui.scroll(120)  # Scroll up

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
                x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip
                x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger tip

                # Calculate pinch level (distance between index and thumb tips)
                pinchlv = np.hypot(x2 - x1, y2 - y1)

                # Check which fingers are up
                fingers = []
                tipIds = [4, 8, 12, 16, 20]
                for i in range(5):
                    if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Stop cursor movement if full palm is shown
                if fingers == [1, 1, 1, 1, 1]:
                    cursor_stopped = True
                else:
                    cursor_stopped = False

                if not cursor_stopped:
                    # Move mode: Only index finger is up
                    if fingers[1] == 1 and fingers[2] == 0:
                        x_screen = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y_screen = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                        clocX = plocX + (x_screen - plocX) / smoothening
                        clocY = plocY + (y_screen - plocY) / smoothening

                        autopy.mouse.move(clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    # Scroll mode: Index and thumb are pinching
                    if fingers[1] == 1 and fingers[0] == 1:
                        scrollVertical(pinchlv)

                    # Pinch click: Index and middle fingers are pinching
                    if fingers[1] == 1 and fingers[2] == 1:
                        pinch_click_distance = np.hypot(x3 - x1, y3 - y1)
                        if pinch_click_distance < scroll_threshold:
                            autopy.mouse.click()

    cv2.putText(
        img,
        f"Cursor Stopped: {'Yes' if cursor_stopped else 'No'}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
