from PIL import Image

def load_image(path):
    img = Image.open(path)
    img.show()

def getname(response):
    namelist = []
    input = response["outputs"]
    #print type(input[0])
    i = input[0]
    input2 = i["data"]
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