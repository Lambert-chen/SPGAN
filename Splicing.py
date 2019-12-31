"""
@Author: ChenYao
--------------------
Splicing two pictures

"""
import os
from PIL import Image
UNIT_SIZE = 256 # the size of image
TARGET_WIDTH = 6 * UNIT_SIZE
save_path = 'E:\data/face_enhance/complex188'
path = "./pinjieImage"
Apath = "E:\data/face_enhance\photo"
images = []  # all pic name
def pinjie():
    for img in os.listdir(path):
        images.append(img)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(images)//6):# eyery 3 a group
        imagefile = []
        j = 0
        for j in range(6):
            aa = Image.open(path+'/'+images[i*6+j])
            bb = aa.resize((256,256))
            imagefile.append(bb)
        target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))  # width , height
        left = 0
        right = UNIT_SIZE
        for image in imagefile:
            target.paste(image, (left, 0, right, UNIT_SIZE))
            left += UNIT_SIZE
            right += UNIT_SIZE
            quality_value = 1000
        target.save('out_{}.jpg'.format(i), quality=quality_value)
if __name__ == '__main__':
    pinjie()
