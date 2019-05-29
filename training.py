import cv2 as cv
import numpy as np
import os
from facedetection import face_detection

def training(parent): #parent-parent folder path
	directorynames=os.listdir(parent)#directory names inside parent
	faces=[]
	labels=[]

	for directoryname in directorynames:

		if directoryname.startswith("s"):

			x=directoryname.replace("s","")
			label=int(x)

			directorypath=parent+"/"+directoryname #directory path
			imagenames=os.listdir(directorypath) #image names inside current directory_name
			for imagename in imagenames:

				if not imagename.startswith('.'):

					imagepath=directorypath+"/"+imagename #image path
					cropped_faces,coordinates=face_detection(imagepath)
					if cropped_faces is not None: #if face(s) are detected in the image 
						faces.append(cropped_faces[0]) 
						#assuming each image in **training data set** has only one face
						labels.append(label)

	#TO SEE CROPPED FACES WITH LABELS			
	#for i in range(len(faces)):

		#cv.imshow(str(labels[i]),faces[i])
		#cv.waitKey(0)
		#cv.destroyAllWindows()
	
	return faces,labels



