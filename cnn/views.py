from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import json
from tensorflow import Graph
import tensorflow as tf
from keras.datasets import cifar10

from cnn.models import ImageFind

img_height, img_width=224,224
with open('./models/cifar10_classes.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/Cifar10.h5')


def index(request):
    myContext = {'a': 1}
    return render(request,'index.html',myContext)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    context = {}
    if request.FILES:
        fileObjectName = request.FILES['myPath']
        saveFile = FileSystemStorage()
        pathName = saveFile.save(fileObjectName.name, fileObjectName)
        pathName = saveFile.url(pathName)
        imageTesting = '.' + pathName
        imageLoad = image.load_img(imageTesting, target_size=(img_height, img_width))
        x = image.img_to_array(imageLoad)
        x = x / 255
        x = x.reshape(1, img_height, img_width, 3)
        with model_graph.as_default():
            with tf_session.as_default():
                predictClass = model.predict(x)

        import numpy as np
        classPrediction = labelInfo[str(np.argmax(predictClass[0]))]

        context = {'pathName': pathName, 'classPrediction': classPrediction[1]}
        image_find = ImageFind(image=context['pathName'], classifier=context['classPrediction'])
        image_find.save()
    return render(request, 'index.html', context)