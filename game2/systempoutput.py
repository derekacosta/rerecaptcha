import random
import imageclarify as im

def channel(name, value, position, module):
    if len(position) == 0:
        print("sorry, it's not likely to have that.")
    else:
        if module == 'food':
            output(name, value, position)
            ing = random.choice(name)
            print "It is also made up with " + str(ing)
        elif module == 'color':
            output(name, value, position)
            ing = random.choice(name)
            print str(ing) + " is also in the photo"
        elif module == 'general':
            output(name, value, position)



def output(name, value, position):
    #print position
    #if len(position) == 0:
        #print("sorry, it's not likely to have that.")
    length = len(position)
    l = len(name)
    i = 0
    while i < length:
        pos = position[i]
        #print value[pos]
        ingredent = name[pos].capitalize()
        if pos < (l/3) :
            print ingredent+" is very likely!"
        elif pos > (l/3) :
            p = str(value[pos] * 100)
            print ingredent + " could be there, the possibility is " + p+ '% '
        else:
            p = str(value[pos] * 100)
            print "You are correct!, " + ingredent+ " has " + p+'% possibility to be there.'
        i = i + 1

def judge(choice, path):
    if 'yes' in choice:
        return False
    elif 'no' in choice:
        return True

def judge2(choice):
    if 'yes' in choice:
        return True
    elif 'no' in choice:
        return False
