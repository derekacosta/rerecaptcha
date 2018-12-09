

def output(name, value, position):
    #print position
    if len(position) == 0:
        print("sorry, it's not likely to have that.")
    else:
        length = len(position)
        i = 0
        while i < length:
            pos = position[i]
            #print value[pos]
            ingredent = name[pos].capitalize()
            if value[pos] > 0.8 :
                print ingredent+" is very likely!"
            elif value[pos] < 0.5 :
                print "It could be there"
            else:
                print "You are correct!"
            i = i + 1
