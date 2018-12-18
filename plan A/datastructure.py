#from PIL import ImageGrab
from PIL import Image as im
import requests as req
from io import BytesIO
import json
import codecs
#from nltk.tokenize import WordPunctTokenizer
#from clarifai.rest import ClarifaiApp
#from clarifai.rest import Image as ClImage

#games=[]
class Image:
    def __init__(self, id, width, height, url):
        self.id = id
        self.width = width
        self.height = height
        self.url = url

    def getid(self):
        return self.id

    def geturl(self):
        return self.url

#https://blog.csdn.net/xingchenbingbuyu/article/details/71404664
    def load_image(self):
        response = req.get(self.url)
        image = im.open(BytesIO(response.content))
        image.show()

class Qa:
    def __init__(self, id, question, answer):
        self.id = id
        self.question = question
        self.answer = answer

    def getid(self):
        return self.id

    def getquestion(self):
        return self.question

    def getanswer(self):
        return self.answer

#https://blog.csdn.net/baidu_27438681/article/details/60468848
    def processquestion(self):
        self.question2 = []
        for sentence in self.question:
            words = WordPunctTokenizer().tokenize(sentence)
            self.question2.append(words)
        return self.question2

class Crop:
    def __init__(self, id, category_id, category, area, bbox, segment):
        self.id = id
        self.category_id = category_id
        self.category = category
        self.area = area
        self.bbox =bbox
        self.segment = segment

    def getid(self):
        return self.id

    def getcategory_id(self):
        return self.category_id

    def getcategory(self):
        return self.category

    def getarea(self):
        return self.area

    def getbbox(self):
        return self.bbox

    def getsegment(self):
        return self.segment

class initial:
    def __init__(self, id, image, qas, task, task_id):
        self.id = id
        self.task_id = task_id
        self.image = Image(id = image["id"],
                           width = image["width"],
                           height = image["height"],
                           #url = image["coco_url"])
                           url = image["flickr_url"])
        self.qas = Qa(id = [qa['id'] for qa in qas],
                     answer = [qa['answer'] for qa in qas],
                     question = [qa['question'] for qa in qas])

        self.task = Crop(id = [t['id'] for t in task],
                         category=[t['category'] for t in task],
                         category_id=[t['category_id'] for t in task],
                         bbox=[t['bbox'] for t in task],
                         area=[t['area'] for t in task],
                         segment=[t['segment'] for t in task])
    def getid(self):
        return self.id

    def getimage(self):
        return self.image

    def getqa(self):
        return self.qas

    def gettask(self):
        return self.task

def trans():
    games=[]
    jsonData = codecs.open('test.json', 'r', 'utf-8')
    i = 0
    for line in jsonData:
        dic = json.loads(line.strip('\n'))
        g = initial(
                id = dic['id'],
                task = dic['objects'],
                qas = dic['qas'],
                image = dic['image'],
                task_id = dic['object_id'])

        games.append(g)
    #print games[0].gettask()
    #print type(games[0].gettask)
    #print games[0].gettask().getbbox()
    #print games[0].getqa().getquestion()
    #output = games[0].getqa()
    #print output['answer']
    jsonData.close()
    return games
'''
def crop_image(image, bbox):
    response = req.get(image)
    img = im.open(BytesIO(response.content))
    crop_image = img.crop(bbox)
    crop_image.show()
''''''
def train():
    app = ClarifaiApp(api_key='b9004e37bd6143c1b2d3d70a056bd80d')
    img1 = ClImage(url = games[0].getimage().geturl(),concepts = ['boat'])
    img2 = ClImage(url = games[1].getimage().geturl(),concepts = ['skis'])
    app.inputs.bulk_create_images([img1,img2])
    model = app.models.create('seaside',concepts = ['boat'])
    model = app.models.get('{model_id}')
    model.train()
    image = ClImage(url = games[2].getimage().geturl())
    model.predict([image])
'''

if __name__ == '__main__':
    games = trans()
    #train()
    #print (games[3].getqa().getquestion())
    #print games[3].getqa().getanswer()
    #print games[0].gettask().getcategory()

    #games[3].getimage().load_image()
    #image = games[0].getimage().geturl()
    #area = games[0].gettask().getbbox()
    #print area[0]
    #crop_image(image, area[0]) #have problem