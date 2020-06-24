# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = tf.keras.models.load_model('./weigts3.h5')

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue

	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		xscale = h//96 + 1
		yscale = w//96 + 1
		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(96,96))
		face_section = cv2.cvtColor(face_section,cv2.COLOR_BGR2GRAY)
		face_section = face_section.reshape((96,96,1))
		img = np.expand_dims(face_section,axis=0)
		img = img.astype('float64')
		predictions = model.predict(img)[0]
		# print(img.shape)
		# print(predictions)
		xa = []
		ya = []
		for i,p in enumerate(predictions):

			if i%2==0:
		 		xa.append(p)
			else:
		 		ya.append(p)

	print(h,w)



	for i in range(len(xa)):
		frame = cv2.circle(frame,(x + int(xa[i]*xscale),y+ int(ya[i]*yscale)),1,(0,0,255),4)
		face_section = cv2.circle(face_section,(int(xa[i]),int(ya[i])),1,(0,0,255),1)

	face_section = cv2.resize(face_section,(300,300))
	cv2.imshow("Frame",frame)

	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
# face_data = np.asarray(face_data)
# face_data = face_data.reshape((face_data.shape[0],-1))
# print(face_data.shape)

# Save this data into file system
# np.save(dataset_path+file_name+'.npy',face_data)
# print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()
