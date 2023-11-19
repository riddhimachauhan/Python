#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


import cv2


# In[1]:


import cv2
import numpy as np

def track_green_ball():
    video_path = r'C:\Users\ridds\Desktop\rugved video.mp4'
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print("Error: Couldn't open video file.")
        return

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 0, 0), 4)

        cv2.imshow('Tracking video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

track_green_ball()


# In[ ]:




