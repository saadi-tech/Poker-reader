import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import random
import math
import time
import string
non_numeric_chars = ''.join(set(string.printable) - set(string.digits))
import numpy as np


kernel = np.ones((3,3),np.uint8)
starting = time.time()
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
yellow = (0,255,255)
white = (255,255,255)

stack_options = [30,50,100]


positions = ['DEALER','SB','BB','UTG','MP','CO']


def get_closest_option(value,list):
    distance =[]
    for i in range(len(list)):
        distance.append(abs(list[i] - value))

    min_index = distance.index(min(distance))

    return list[min_index]

def remove_empty_string(test_list):
    while ("" in test_list):
        test_list.remove("")
    return test_list
def read_text(image):
    d = pytesseract.image_to_string(image, output_type=Output.DICT)
    return d

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


def in_the_boundary(player,box):
    (X,Y,W,H) = player
    (x,y,w,h) = box

    if ( (x >X and x<X+W) and (y > Y and y<Y+H) or (x+w >X and x+w<X+W) and (y+h > Y and y+h<Y+H) ):
        return True
    else:
        return False

class poker_player:

    def __init__(self,boundary):
        self.boundary = boundary
        self.deck = None
        self.dealer_sign = None
        self.account_money= None
        self.account_money_value = None
        self.position = None
        self.angle = None
        self.bb_money = None
        self.my_player = False
        self.my_hand = None
    def set_position(self,position):
        self.position = position
    def set_deck(self,deck):
        self.deck = deck
    def set_dealer_sign(self,sign):
        self.dealer_sign = sign
    def set_account(self,account):
        self.account_money = account




def load_coordinates(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    account_moneys = []
    players = []
    dealer_sign = []
    my_hand = []
    decks = []
    current_type = None

    for i in range(len(lines)):
        line = lines[i].strip()
        words = line.split(",")

        if (len(words) == 1):
            current_type = words[0]

        else:
            (x, y, w, h) = int(words[0]), int(words[1]), int(words[2]), int(words[3])

            if (current_type == 'account_moneys'):
                account_moneys.append((x, y, w, h))
            if (current_type == 'players'):
                players.append((x, y, w, h))
            if (current_type == 'dealer_sign'):
                dealer_sign.append((x, y, w, h))
            if (current_type == 'my_hand'):
                my_hand.append((x, y, w, h))
            if (current_type == 'decks'):
                decks.append((x, y, w, h))

    return (players,account_moneys,decks,dealer_sign,my_hand)

def show_all_players(image,all_players):

    for i in range(len(all_players)):
        current_player = all_players[i]
        color = (random.randint(100,255),random.randint(100,255),random.randint(100,255))

        (x,y,w,h) = current_player.deck
        cv2.rectangle(image,(x,y),(x+w,y+h),color,1)

        (x, y, w, h) = current_player.boundary
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

        (x, y, w, h) = current_player.dealer_sign
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

        (x, y, w, h) = current_player.account_money
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)


    cv2.imshow("All players:",image)
    cv2.waitKey(0)



def show_one_player(current_player,image,color,image_name):
    (x, y, w, h) = current_player.deck
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    (x, y, w, h) = current_player.boundary
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    (x, y, w, h) = current_player.dealer_sign
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    (x, y, w, h) = current_player.account_money
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    cv2.imshow(image_name,image)
    cv2.waitKey(0)

#this function matches the template by resizing template from 0.7x to 1.3x of the original size to cater different screen sizes..
def match_image(gray_image,template):
    (tW, tH) = template.shape[::-1]  # get the width and height
    # match the template using cv2.matchTemplate
    match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    position = np.where(match >= threshold)  # get the location of template in the image
    found = 0
    X = 0
    Y = 0
    W = 0
    H = 0
    R = 1

    scales = np.linspace(0.7, 1.3, 5)[::-1]
    scales = np.insert(scales, 0, 1.0, axis=0)
    for scale in scales:

        # Resize image to scale and keep track of ratio
        resized = maintain_aspect_ratio_resize(template, width=int(template.shape[1] * scale))
        r = template.shape[1] / float(resized.shape[1])

        match = cv2.matchTemplate(gray_image, resized, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        position = np.where(match >= threshold)  # get the location of template in the image

        for point in zip(*position[::-1]):  # draw the rectangle around the matched template

            #cv2.rectangle(main_image, point, (int(point[0] + tW / r), int(point[1] + tH / r)), (0, 204, 153), 3)
            #cv2.imshow("FOUND something..", main_image)

            found = 1
            (X,Y,W,H,R) = (int(point[0]),int(point[1]),int( tW / r),int(tH / r) , r)
            return True

    return False


def find_dealer(all_players,template,image):
    dealer_index = 0
    all_players.sort(key=lambda x: x.angle, reverse=True)

    for i in range(len(all_players)):
        current_player = all_players[i]
        (x,y,w,h) = current_player.dealer_sign

        cropped = image[y:y+h , x:x+w]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)


        if (match_image(cropped,template)):
            #print("DEALER FOUND.")
            #show_one_player(current_player,image,red,"DEALER")
            current_player.set_position("DEALER")
            dealer_angle = current_player.angle
            #print("Dealer angle:",dealer_angle)
            dealer_index = i
            break




    done = 0
    while(done<6):
        all_players[dealer_index].position = positions[done]
        done+=1
        dealer_index +=1
        if(dealer_index == 6):
            dealer_index = 0



    return all_players

def set_all_players(players,account_moneys,decks,dealer_sign,Width,Height,image):
    all_players = []

    for i in range(len(players)):
        current_player = poker_player(players[i])
        theta = get_angle(current_player,Width,Height,image.copy())
        current_player.angle = theta

        for j in range(len(account_moneys)):
            if ( in_the_boundary(players[i],account_moneys[j]) ):
                current_player.set_account(account_moneys[j])
                break

        for j in range(len(dealer_sign)):
            if ( in_the_boundary(players[i],dealer_sign[j]) ):
                current_player.set_dealer_sign(dealer_sign[j])
                break

        for j in range(len(decks)):
            if ( in_the_boundary(players[i],decks[j]) ):
                current_player.set_deck(decks[j])
                break

        all_players.append(current_player)

    return all_players

def get_angle(player,width,height,image):
    anchor_x = int(width/2)
    anchor_y = int(height/2)

    player_x = player.boundary[0]+int (player.boundary[2] / 2)
    player_y = player.boundary[1] + int(player.boundary[3] / 2)


    #cv2.line(image,(anchor_x,anchor_y),(player_x,player_y),red,2)


    if (player_x - anchor_x == 0):
        if (player_y > anchor_y):
            theta = -90
        if (player_y < anchor_y):
            theta = 90
    else:
        slope = abs(player_y - anchor_y)/abs(player_x - anchor_x)
        theta = math.atan(slope)
        theta = theta * 180 / 3.14

    if (player_x > anchor_x and player_y< anchor_y):
        #first quadrant.. do nothing

        pass

    elif (player_x <anchor_x and player_y < anchor_y):
        #2nd quadrant..............

        theta = 180 - theta
    elif (player_x < anchor_x and player_y > anchor_y):
        #3rd quadrant.

        theta = 180 + theta
    elif (player_x > anchor_x and player_y > anchor_y):
        #4th quad

        theta = 360 - theta
    #cv2.waitKey(0)
    return theta

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def read_my_hand(my_hand_cropped):
    my_hand_cropped = maintain_aspect_ratio_resize(my_hand_cropped, width=200)
    val = 85
    lower_red = np.array([val, val, val])
    upper_red = np.array([255, 255, 255])
    mask = cv2.inRange(my_hand_cropped, lower_red, upper_red)

    masked = cv2.bitwise_not(mask)
    res = cv2.Canny(masked, 30, 150)

    (cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    (cnts,_) = sort_contours(cnts)
    HEIGHT = 50
    img = np.zeros([HEIGHT, 200], dtype=np.uint8)
    img.fill(255)  # or img[:] = 255

    x_off = 35
    y_off = 35

    (height,width) = res.shape

    for i in range(len(cnts)):
        (x, y, w, h) = cv2.boundingRect(cnts[i])

        if (h>20 and x>2 and y>2 and x<(width-2) and y < (height-2)):

            cropped = masked[y:y+h , x:x+w]


            y_off = (int(HEIGHT/2) - int(h/2))


            img[y_off:y_off+h,x_off:x_off+w] = cropped

            x_off += int(1.2*w)
            #cv2.imwrite("cropped\\image_"+str(i)+".png",img)

    img = cv2.dilate(img, kernel, iterations=1)



    text = read_text(img)['text']

    text = text.replace('W','10')
    text = text.replace('l', '1')
    text = text.replace('O', '0')
    text = text.upper()
    return (text)

def set_players_account_moneys(all_players,image):
    img = np.zeros([500, 150], dtype=np.uint8)
    img.fill(255)  # or img[:] = 255

    x_off = 0
    y_off = 50
    for i in range(len(all_players)):
        current_player = all_players[i]
        (x,y,w,h) = current_player.account_money
        cropped = image[y:y+h , x:x+w]
        val = 60
        lower_red = np.array([val, val, val])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(cropped, lower_red, upper_red)
        mask = cv2.bitwise_not(mask)
        mask = maintain_aspect_ratio_resize(mask,width=150)
        (h,w) = (mask.shape)

        img[y_off:y_off+h ,x_off:x_off+w ] = mask
        y_off += int(1.5*h)

    text = read_text(img)['text'].split("\n")
    text = remove_empty_string(text)
    print("Accoutns:",text)
    for i in range(len(all_players)):
        if ('$' in text[i]):
            val = text[i].split('$')[-1].strip()
        else:
            val = 0
        all_players[i].account_money_value = int(val)

    return all_players


def read_bb(all_players,image):

    for i in range(len(all_players)):
        current_player = all_players[i]

        (x,y,w,h) = current_player.deck
        cropped = image[y:y+h , x:x+w]
        #cropped = maintain_aspect_ratio_resize(cropped, width=300)
        cropped = cv2.resize(cropped,(200,100))
        val = 130
        lower_red = np.array([val, val, val])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(cropped, lower_red, upper_red)
        mask = cv2.bitwise_not(mask)

        #text = (read_text(mask)['text'])



        res = cv2.Canny(mask, 30, 150)
        (cnts, _) = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if (len(cnts) > 0):
            (cnts, _) = sort_contours(cnts)
            img = np.zeros([120, 140], dtype=np.uint8)
            img.fill(255)  # or img[:] = 255

            x_off = 30
            y_off = 40

            for i in range(len(cnts)):
                (x, y, w, h) = cv2.boundingRect(cnts[i])

                if (x > 5 and y > 5 and w*h>60 and w<2*h):
                    cropped = mask[y:y + h, x:x + w]

                    img[y_off:y_off + h, x_off:x_off + w] = cropped

                    x_off += int(1.3* w)
                    # cv2.imwrite("cropped\\image_"+str(i)+".png",img)



            #cv2.imshow("Cropped chars",img)
            text = (read_text(img)['text'])
            #cv2.imshow("Mask",mask)
            #print("Read:", text)
            #cv2.waitKey(0)

            if ( '$' in text):
                #


                text = text.split("$")[-1].strip()

                #text = text.translate(non_numeric_chars)

                current_player.bb_money = int(text)
                #print("Amount: ",text)

            else:
                #print("Not raised anythin")
                current_player.bb_money = 0

                #show_one_player(current_player,image.copy(),green,current_player.position)
        else:
            # print("Not raised anythin")
            current_player.bb_money = 0
    return  all_players


def scan_image(path):
    all_players = []
    image = cv2.imread(path)

    image = maintain_aspect_ratio_resize(image, height=615)
    image = cv2.resize(image,(793,615))

    (Height, Width, ch) = image.shape

    (players, account_moneys, decks, dealer_sign, my_hand) = load_coordinates("poker_coords.txt")

    all_players = set_all_players(players, account_moneys, decks, dealer_sign, Width, Height, image.copy())
    #show_all_players(image.copy(), all_players)
    template = cv2.imread("dealer_template.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    all_players = find_dealer(all_players, template, image.copy())
    # for i in range(len(all_players)):
    # print("Position:", all_players[i].position)
    # show_one_player(all_players[i],image.copy(),red,"Sorted players.")

    (x, y, w, h) = my_hand[0]

    my_hand_cropped = image[y:y + h, x:x + w]

    my_hand_value = read_my_hand(my_hand_cropped)
    #print("My hand: ", my_hand_value)
    # ("My position: ",all_players[1].position)
    all_players[1].my_player = True
    all_players[1].my_hand = my_hand_value
    #show_one_player(all_players[1], image.copy(), red, "My player")

    all_players = set_players_account_moneys(all_players, image.copy())
    # for i in range(len(all_players)):
    # print(f"Amount: {all_players[i].position} :",all_players[i].account_money_value)
    # show_one_player(all_players[i],image.copy(),green,all_players[i].position)

    all_players = read_bb(all_players, image.copy())

    # final comparing:
    my_pos = all_players[1]
    my_bb = my_pos.bb_money

    players_raised = 0
    for i in range(len(all_players)):
        current_player = all_players[i]
        bb = current_player.bb_money
        if(my_bb == 0):
            print("NO money on player's deck...")
            return 0
        if (not current_player.my_player and bb >= 2 * my_bb):
            print("Player who rised: ", current_player.position)
            print(f"Deck Value: {current_player.bb_money}")
            print(f"Account value: {current_player.account_money_value}")
            if(my_bb != 0 ):
                players_raised+=1
                value = current_player.account_money_value / my_bb
                print("Value: (oppBB/myBB) ", value)
                closest_option = get_closest_option(value, stack_options)
                print(f"Closest stack option: {closest_option}")
                print("time taken: ", time.time() - starting, "secs..")

                print("------------- OUTPUT -----------------")
                print(
                    f"Input: {my_hand_value} {my_pos.position.lower()} {current_player.position.lower()}      Drop-Down: {closest_option}bb")
                show_one_player(current_player, image.copy(), green, "Player who rised")
            else:
                print("NO money on player's Deck..")


            print("-------------------------------------\n")
    if players_raised == 0:
        print("NO player raised..")

from os import listdir
from os.path import isfile, join

mypath = "Ranks and positions\\"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for i in range(len(onlyfiles)):
    print(onlyfiles[i])
    ind = i
    scan_image(mypath+onlyfiles[ind])
    img = cv2.imread(mypath + onlyfiles[ind])
    cv2.imshow("Image",img)
    cv2.waitKey(0)

    '''try:
        scan_image(mypath+onlyfiles[0])
    except Exception as e:
        print("Error =", e)'''