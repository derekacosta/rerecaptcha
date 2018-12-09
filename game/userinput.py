import sys
import re
import imageclarify as im

position = []

def processinput(input):
    input = re.sub('[^\w ]','',input)
    input = input.split()
    return input

def findingredent(out,name):
    namelist = name
    #print "print namelist"
    #print namelist
    #valuelist = im.getvaluelist()
    length_name = len(namelist)
    length_out = len(out)
    position = []
    j = 0
    while j < length_out:
        i = 0
        while i < length_name:
            if namelist[i] != out[j] :
                i = i + 1
            elif namelist[i] == out[j] :
                #print "find"
                position.append(i)
                break
        j = j + 1
    #print position
    return position

def getposition():
    return position
'''
if __name__ == '__main__':
    input = sys.stdin.readline().strip('\n')
    out = processinput(input)
    print out
    position = findingredent(out)
    print "print result"
    for i in position:
        print im.namelist[i]
'''