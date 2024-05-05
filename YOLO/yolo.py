import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3


classesFiles = 'coco.names.txt'
classNames = []
with open(classesFiles, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    #print (classNames)
    #print (len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for outputs in outputs:
        for det in outputs:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox)) 
    indices = cv.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)  
    print(indices)


    for i in indices:
        i = i
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3] 
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10), cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)        



while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1/255, (whT, whT),[0,0,0],1, crop = False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)


    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])

    findObjects(outputs, img)



    cv.imshow('Image', img)
    cv.waitKey(1)