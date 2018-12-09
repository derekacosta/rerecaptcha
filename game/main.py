import imageclarify as im
import userinput as ui
import systempoutput as so
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import sys
app = ClarifaiApp(api_key = 'b9004e37bd6143c1b2d3d70a056bd80d')

name = []
value = []
if __name__ == '__main__':
    model = app.models.get('food-items-v1.0')
    path = 'beef.jpg'
    #image = ClImage(url = 'https://samples.clarifai.com/food.jpg')
    image = ClImage(file_obj = open(path,'rb'))
    #im.load_image(path)
    response = model.predict([image])
    name = im.getname(response)
    #print type(response)
    #input = Ingredent(outputs = response["outputs"])
    #print input.getdata().getdata().getdata().getname()
    #print name
    value = im.getvalue(response)
    #print value

    print "Guess the ingredents!"
    input = sys.stdin.readline().strip('\n').lower()
    out = ui.processinput(input)
    #print out

    position = ui.findingredent(out, name)
    #for i in position:
        #print name[i]

    so.output(name, value, position)