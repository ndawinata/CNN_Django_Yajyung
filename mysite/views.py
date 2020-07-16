from django.shortcuts import render
import os
import pickle
import numpy as np
import requests
from datetime import datetime
from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from keras.models import load_model

def index(request):
    context={
        'konten':'snippets/homeKonten.html',
    }
    
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        context['fileUrl'] = uploaded_file_url
        lokasi = os.path.dirname(__file__)
        module_path = os.path.join(lokasi, 'model.pkl')
        img_path = os.path.join(lokasi, "../media/"+filename)
        f = open(module_path, 'rb')
        model = pickle.load(f)
        f.close()

        raw_img = image.load_img(img_path, target_size=(64, 64))
        raw_img = image.img_to_array(raw_img)
        raw_img = np.expand_dims(raw_img, axis=0)
        raw_img = raw_img/255
        prediction = model.predict_classes(raw_img)[0][0]
        if (prediction):
            context['awan'] = "Cumulus"
        else:
            context['awan'] = "Cumulonimbus"

    return render(request, './index.html', context)


