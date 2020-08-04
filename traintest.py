import convert
import cv2, os

original_path = os.getcwd() + '/Trainfix'
original_dir_list = os.listdir(original_path)
directory = "Trainframe"

print("Train")
for i in original_dir_list:
    path = original_path + "/" + i
    print("Dataset " + i)
    for j in os.listdir(path):
        count = 0
        filepath = path + "/" + j
        img, label = convert.convertToImage(filepath)
        nama = j.split(".")[0] #Z92
        le_dir = directory + "/" + i #Trainframe/A
        if not os.path.exists(le_dir):
            os.makedirs(le_dir)
        for x in img:
            cv2.imwrite(le_dir + "/" + nama + "-%d.jpg" % count, x)
            flip = cv2.flip(x, 1)
            temp = i + str(int(nama[1:]) + 100)
            cv2.imwrite(le_dir + "/" + temp + "-%d.jpg" % count, flip)
            count += 1

print("\nTest")
original_path = os.getcwd() + '/Testfix'
original_dir_list = os.listdir(original_path)
directory = "Testframe"

for i in original_dir_list:
    count = 0
    filepath = original_path + "/" + i
    img, label = convert.convertToImage(filepath)
    nama = i.split(".")[0] #Z92
    if not os.path.exists(directory):
        os.makedirs(directory)
    for x in img:
        cv2.imwrite(directory + "/" + nama + "-%d.jpg" % count, x)
        count += 1