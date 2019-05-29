import cv2 as cv
import numpy as np
from facedetection import face_detection
from training import training 


def predict(testimagepath,face_recognizer,names):
	img=cv.imread(testimagepath)

	#if img is None:
		#print("wrong path")
	img=img.copy()

	cropped_faces,coordinates=face_detection(testimagepath) #detect faces in test image
	#print(str( len(cropped_faces))+ " faces detected")
	if cropped_faces is None:
		print("No face detected")
		return None



	for i in range(len(cropped_faces)): #for each detected face
		(x,y,width,height)=coordinates[i]

		cropped_face=cropped_faces[i]

		label,confidence=face_recognizer.predict(cropped_face)  #confidence is an integer returned by predict method which gives associated confidence (e.g. distance) for the predicted label.
		cv.rectangle(img,(x, y),(x+ width, y + height),(0, 255, 0),2)  #draw a rectangle around detected face
		cv.putText(img,str(names[label]),(x,y),cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1) #put the correct label above the detected face


	

	return img




#train the data using all the images in the dataset
faces,labels=training("training-data")

#LFBP Face Recognizer
face_recognizer=cv.face.LBPHFaceRecognizer_create()

#convert labels (list) to numpy array as LBPH Face Recognizer needs a numpy array as its second argument
labels=np.array(labels)
face_recognizer.train(faces,labels)


testimagepath="test-data/1.jpeg"

names={1:'Evans',2:'Hemsworth'} # a dictionary to store name associated with each label
predictedimg=predict(testimagepath,face_recognizer,names)

if predictedimg is not None:

	cv.imshow('Recognized Faces',predictedimg)
	cv.waitKey(0)
	cv.destroyAllWindows()