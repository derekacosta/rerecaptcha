import imageclarify as im
import userinput as ui
import systempoutput as so
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import sys
import os
import random
app = ClarifaiApp(api_key = 'b9004e37bd6143c1b2d3d70a056bd80d')



def importfiles(dictionary):
    list=[]
    path = os.listdir(dictionary)
    for file in path:
        if file.endswith(".jpg"):
            list.append(file)
    return list

if __name__ == '__main__':

    print "Guess what is in the photo!"
    i = 0
    again = True
    while again == True:

        image, path = im.random_show()
        #print path
        state = True
        while state == True:
            name = []
            value = []
            #user input
            input = ui.getinput()
            module = ui.witprocess(input)
            model = im.loadmodule(app, module)

            #  get name and value
            response = model.predict([image])
            if module == 'color':
                name = im.getcolor(response)
                #print name
                value = im.getcolorvalue(response)
            else:
                name = im.getname(response)
                #print type(response)
                #print name
                value = im.getvalue(response)
                if input == "show":
                    print name
                    print value
                #print name
                #print value
                #print module
            out = ui.processinput(input)
            #print out
            position = ui.findingredent(out, name)
            #print name[i]
            so.channel(name, value, position, module)
            #state = False
            i = i + 1
            if i == 6:
                i = 0
                print "Do you want to see the original photo?"
                choice = sys.stdin.readline().strip('\n').lower()
                state = so.judge(choice, path)
                if state == False:
                    im.load_origin(path)
        print "Do you want to play one more time?"
        choice =  sys.stdin.readline().strip('\n').lower()
        again = so.judge2(choice)

