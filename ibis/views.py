import cv2
from django.http import HttpResponse
from .models import birds_info
from django.shortcuts import redirect
from django.shortcuts import render
import nltk
import os.path
from django.core.files.storage import FileSystemStorage
import numpy as np
from scipy import fftpack as fftp

import argparse
from matplotlib import pyplot as plt   #for histogram
import csv
import pandas as pd

import re, io, base64
from PIL import Image
from keras.models import load_model
from django.http import *
import tensorflow as tf




def index(request):
    response = render(request,'ibis/ibis.html',{})
    return response

def show(request):
    data = birds_info.objects.all()
    html = ''
    for d in range(len(data)):
        html += data[d].name + "</br>"+ "size : "+ str(data[d].size) + " cm" + "</br>" + "Colour :" + data[d].colour + "</br>" + data[d].info + "</br></br></br>"
    return HttpResponse (html)

def getData(request):
    all = []
    obj = [] 
    
    size1 = request.POST.get('size')
    colour1 = request.POST.get('colour')
    bill = request.POST.get('bill')
    head = request.POST.get('head')
    upperparts = request.POST.get('upperparts')
    underparts = request.POST.get('underparts')
    uwc = request.POST.get('upperwingCoverts')
    unc = request.POST.get('underwingCoverts')
    utc = request.POST.get('undertailCoverts')
    leg = request.POST.get('legs')
    neck = request.POST.get('neck')
    breast = request.POST.get('breast')
   
    all_birds = birds_info.objects.all()
    size2 = int(size1)
    print (size2)
    for j in range(len(all_birds)):
        if(0 <= size2 <= 20):
            if(0 <= all_birds[j].size <= 20):
                if(all_birds[j].colour == colour1):
                    all.append(all_birds[j])
        elif(21 <= size2 <= 35):
            if(21 <= all_birds[j].size <= 35):
                if(all_birds[j].colour == colour1):
                    all.append(all_birds[j])
        elif(36 <= size2 <= 60):
            if(36 <= all_birds[j].size <= 60):
                if(all_birds[j].colour == colour1):
                    all.append(all_birds[j])
        elif(61 <= size2 <= 80):
            if(61 <= all_birds[j].size <= 80):
                if(all_birds[j].colour == colour1):
                    all.append(all_birds[j])
        elif(size2 > 80):
            if(all_birds[j].size > 80):
                if(all_birds[j].colour == colour1):
                    all.append(all_birds[j])
        else:
            return HttpResponse("<h1>Invalid size</h1>")
    
    s = len(all)
    
    html = ""
    attr = [bill,head,upperparts,underparts,uwc,unc,utc,leg,neck,breast]
    test = []
    tokens=[]
    
    num=0
    c = ["bill", "head", "upperparts","underparts","upperwingcoverts", "underwingcoverts", "undertailcoverts", "legs", "neck", "breast", "black", "white", "red", "yellow", "brown", "brownishgrey", "blackishgrey", "blackishbrown", "darkred", "reddish", "bluegrey", "palegrey", "whitishbrown", "buff", "brownishbuff", "grey","greybrown","buffwhite",
    "greyishwhite", "orangebuff","rufousbrown","darkbrown","darkgrey","orangeyellow", "limegreen","slategrey","goldenyellow"]
    for y in range(s):
        tokens = []
        string = all[y].info.lower()
        tokens = nltk.word_tokenize(string)
        for x in range(len(tokens)):
            try:
                n1 = c.index(tokens[x].lower())
                test.append(tokens[x])
            except ValueError:
                continue
       
        test.append("end")   
        
        n=0
        for l in range(10):
            try:
                t=test.index(c[l].lower())
                if (attr[l] == test[t+1] or attr[l] == test[t+2] or attr[l] == "null"):
                    n += 1
                else:
                    n=50
                    break
            except ValueError:
                continue
        if(n>0 and n<50):
            obj.append(all[y])

    response = render(request,'ibis/show.html',{'obj':obj})
    return response
 
def description(request):
    k = request.POST['name']
    obj = birds_info.objects.filter(id = k)

    response = render(request,"ibis/description.html",{'obj':obj})
    return response

def compression(request):
    if request.method == 'POST' and request.FILES['input_file']:
        myfile = request.FILES['input_file']
        
        fs = FileSystemStorage(location = 'C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images')
        filename = fs.save(myfile.name,myfile)
        fname = os.path.splitext(myfile.name)[0]
        
    firstq = 30
    secondq = 40

    thres = 0.5

    image = cv2.imread('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/'+myfile.name)


    dct_rows = 0;
    dct_cols = 0;

    shape = image.shape;

    if shape[0]%8 != 0:
        dct_rows = shape[0]+8-shape[0]%8;
    else:
        dct_rows = shape[0];
    if shape[1]%8 != 0:
        dct_cols = shape[1]+8-shape[1]%8;
    else:
        dct_cols = shape[1];	
    dct_image = np.zeros((dct_rows,dct_cols,3),np.uint8)
    dct_image[0:shape[0], 0:shape[1]] = image

    y = cv2.cvtColor(dct_image, cv2.COLOR_BGR2YCR_CB)[:,:,0]

    w=y.shape[1]
    h=y.shape[0]
    n = w*h/64

    Y = y.reshape(h//8,8,-1,8).swapaxes(1,2).reshape(-1, 8, 8)

    qDCT =[]
    for i in range(0,Y.shape[0]):
        qDCT.append(cv2.dct(np.float32(Y[i])))

    qDCT = np.asarray(qDCT, dtype=np.float32)

    qDCT = np.rint(qDCT - np.mean(qDCT, axis = 0)).astype(np.int32)
    f,a1 = plt.subplots(8,8)
    a1 = a1.ravel()
    k=0;
    flag = True
    for idx,ax in enumerate(a1):
        k+=1;
        data = qDCT[:,int(idx/8),int(idx%8)]
        val,key = np.histogram(data, bins=np.arange(data.min(), data.max()+1),normed = True)
        z = np.absolute(fftp.fft(val))
        z = np.reshape(z,(len(z),1))
        rotz = np.roll(z,int(len(z)/2))
        slope = rotz[1:] - rotz[:-1]
        indices = [i+1 for i in range(len(slope)-1) if slope[i] > 0 and slope[i+1] < 0]
        peak_count = 0

                    
        for j in indices:
            if rotz[j][0]>thres:
                peak_count+=1
        if(k==3):
            if peak_count>=20:
                print ("Double Compressed")
            else:
                print ("Single Compressed")
            flag = False       
        ax.plot(rotz)    
        
        for x in range(0,1):
	        for y in range(0,2):
		        if y == 0:
			        image = cv2.imread('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/'+myfile.name)
			        cv2.imwrite("C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/"+fname+" compressed.jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),firstq])
		        # if y==1:
			    #     image = cv2.imread("C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/comp"+str(firstq)+".jpg")
			    #     cv2.imwrite("C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/comp"+str(firstq)+str(secondq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),secondq])
    os.remove('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/'+ myfile.name)

    # graph = tf.get_default_graph()
    # network = tf.keras.models.load_model('model/project_ibis/model/history.h5')
    
    # # with graph.as_default():
    # # size = 150,150
    # image = cv2.imread('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/'+fname+' compressed.jpg')
    # image = np.resize(image,(150,150))
    # image = np.array(image)

    # with graph.as_default():
    #     value = network.predict(image)
    #     value = value.reshape((150,150))
    # #     value = value.tolist()

    # network.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    # img = cv2.imread('C:/Users/meet3/Desktop/project_ibis/media/project_ibis/input_images/'+fname+' compressed.jpg')
    # img = cv2.resize(img,(150,150))
    # img = np.reshape(img,[1,150,150,3])
    # classes = network.predict(img)
    # classes = classes.tolist()
    return HttpResponse ("success")