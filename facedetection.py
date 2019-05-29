import cv2 as cv

def face_detection(imagepath):
	# Read image from your local file system
	original_image = cv.imread(imagepath)
	cropped_faces=[]  

	# Convert color image to grayscale for Viola-Jones
	grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
	face_cascade = cv.CascadeClassifier('./opencvfiles/haarcascade_frontalface_alt.xml')

	detected_faces_coordinates = face_cascade.detectMultiScale(grayscale_image,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))

	#if no face is detected
	if(len(detected_faces_coordinates)==0):
		return None,None

	for (x, y, width, height) in detected_faces_coordinates: #x,y,w,h
		cv.rectangle(original_image,(x, y),(x+ width, y + height),(0, 255, 0),2)
		#cv.imshow('Image', original_image)
		#cv.waitKey(0)
		#cv.destroyAllWindows()
	for i in detected_faces_coordinates:
		(x,y,w,h)=i
		cropped_faces.append(grayscale_image[y:y+h,x:x+w])


	
	'''
	for img in cropped_faces:
		cv.imshow('cropped face',img)
		cv.waitKey(0)
		cv.destroyAllWindows()
	'''	
	return cropped_faces,detected_faces_coordinates
	
