from PIL import Image,ImageFilter
from clarifai.rest import Image as ClImage
#from clarifai.rest import ClarifaiApp
#app = ClarifaiApp(api_key = 'b9004e37bd6143c1b2d3d70a056bd80d')
import os
import random

def importfiles(dictionary):
    list=[]
    path = os.listdir(dictionary)
    for file in path:
        if file.endswith(".jpg"):
            list.append(file)
    return list

def random_show():
    dictionary = 'image'
    imagelist = importfiles(dictionary)
    # print imagelist
    path = random.choice(imagelist)
    path = dictionary + '/' + path
    # print path
    # image = ClImage(url = 'https://samples.clarifai.com/food.jpg')
    load_image(path)
    image = ClImage(file_obj=open(path, 'rb'))
    return image, path

def load_image(path):
    img = Image.open(path)
    img = img.filter(ImageFilter.BLUR)
    img = img.filter(ImageFilter.CONTOUR)
    img.show()

def load_origin(path):
    img = Image.open(path)
    img.show()

def loadmodule(app, module):
    if module == 'color':
        model = app.models.get('color')
    elif module == 'food':
        model = app.models.get('food-items-v1.0')
    else:
        model = app.public_models.general_model
        model.model_version = 'aa7f35c01e0642fda5cf400f543e7c40'
    return model

def getname(response):
    namelist = []
    input = response["outputs"]
    #print type(input[0])
    i = input[0]
    input2 = i["data"]
    #print input2
    #print type(input2)
    #print input2["concepts"]
    j = input2["concepts"]
    #print type(j)
    #print j[0]
    k = j[0]
    #print k["name"]
    #namelist = []
    for i in j:
        namelist.append(i["name"])
    return namelist

def getcolor(response):
    namelist = []
    input = response["outputs"]
    i = input[0]
    input2 = i["data"]
    j = input2["colors"]
    #print type(j[0])
    #print j[0]
    for i in j:
        i = i['w3c']
        i = i['name']
        namelist.append(i)
    #print namelist
    return namelist

def getcolorvalue(response):
    valuelist = []
    input = response["outputs"]
    i = input[0]
    input2 = i["data"]
    j = input2["colors"]
    # print type(j[0])
    # print j[0]
    for i in j:
        i = i['value']
        valuelist.append(i)
    return valuelist

def getvalue(response):
    valuelist = []
    input = response["outputs"]
    i = input[0]
    input2 = i["data"]
    j = input2["concepts"]
    k = j[0]
    for i in j:
        valuelist.append(i["value"])
    return valuelist

'''
if __name__ == '__main__':
    model = app.models.get('food-items-v1.0')
    #image = ClImage(url = 'https://samples.clarifai.com/food.jpg')
    image = ClImage(file_obj = open('beef.jpg','rb'))
    response = model.predict([image])
    name = getname(response)
    #print type(response)
    #input = Ingredent(outputs = response["outputs"])
    #print input.getdata().getdata().getdata().getname()
    print name
    value = getvalue(response)
    print value

    print "guess the ingredents!"
    input = sys.stdin.readline().strip('\n').lower()
    out = userinput.processinput(input)
    print out
    position = userinput.findingredent(out)
    #for i in position:
        #print namelist[i]
'''