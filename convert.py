import cv2, random

def manual_count(vidcap):
    frames = 0
    while True:
        status, frame = vidcap.read()
        if not status:
            break
        frames += 1
    return frames

def convertToImage(data_list):
    hw = 128
    min = 10
    img_list = []
    img_label = []

    vidcap = cv2.VideoCapture(data_list)
    length = manual_count(vidcap)

    vidcap = cv2.VideoCapture(data_list)
    if(length != min):
        my_list = list(range(1, length-1))
        lines = random.sample(my_list, min - 2)
        lines.sort()

        success, image = vidcap.read()
        count = 0
        check = 0
        moo = 0
        while success:
            image = cv2.resize(image, (hw, hw))

            if(check == 0 or check == length - 1):
                img_list.append(image)
                label = data_list.split("/")[2]
                img_label.append(label)
                count += 1
                check += 1
            elif(check == lines[moo]):
                img_list.append(image)
                label = data_list.split("/")[2]
                img_label.append(label)
                count += 1
                check += 1
                moo += 1
            else:
                check += 1

            if(moo == min - 2):
                moo = 0
            success, image = vidcap.read()
        my_list.clear()
        lines.clear()
    else:
        success, image = vidcap.read()
        count = 0
        while success:
            image = cv2.resize(image, (hw, hw))
            img_list.append(image)
            label = data_list.split("/")[2]
            img_label.append(label)
            count += 1
            success, image = vidcap.read()

    return img_list, img_label