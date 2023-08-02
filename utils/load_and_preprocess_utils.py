import pytesseract
from PIL import Image
import os
import re
import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageChops

# for de skew
from deskew import determine_skew  ## gray image
import numpy as np
from skimage.transform import rotate

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
# def cv2_imshow_rgb(img,resize=None):
#     if resize!=None:
#         img=cv2.resize(img,resize)


#     cv2_imshow(cv2.cvtColor(np.uint8(img),cv2.COLOR_RGB2BGR))

def cv2_imshow_rgb(img,resize=None, figsize=(15,15)):
    if resize!=None:
        img=cv2.resize(img,resize)

        #     cv2_imshow(cv2.cvtColor(np.uint8(img),cv2.COLOR_RGB2BGR))

    plt.figure(figsize=figsize)
    plt.imshow(np.uint8(img))


def display_multi(*images,resize=None, figsize=(15,15),bgr=False,axis=1):
    if resize!=None:
        res = np.array(cv2.resize(images[0],resize))
    else:
        res = np.array(images[0])

    for i in range(1,len(images)):

        if resize!=None:
            res_img = np.array(cv2.resize(images[i],resize))
        else:
            res_img = np.array(images[i])

        res = np.concatenate((res, res_img), axis=axis)

    if bgr==True:
        res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

    return cv2_imshow_rgb(res,resize=None, figsize=figsize)

def load_img(img_path, rgb=True, size=False):
    img = cv2.imread(img_path)
    if rgb==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)

    return img

def write_text_file(text_file_path,output_text,remove_empty_lines=True):
    if remove_empty_lines==True:
        output_text = output_text.split('\n')

    # Save the extracted text to a file
    with open(text_file_path, 'w') as f:
        if remove_empty_lines==True:
            for line in output_text:
                if line.strip():
                    f.write("%s\n" % line)
        else:
            f.write(output_text)


def deskew_image(image,cval=1.0):
    ## https://pypi.org/project/deskew/#:~:text=Deskewing%20is%20a%20process%20whereby,rather%20than%20at%20an%20angle.
    angle = determine_skew(image)
    image = rotate(image, angle, resize=True,cval=cval) * 255
    image = image.astype(np.uint8)
    return image


def add_image_border(img,border_width,border_color):
    img_border = cv2.copyMakeBorder(img,border_width,border_width,border_width,border_width,
                                    cv2.BORDER_CONSTANT,value=border_color)
    return img_border


def save_all_pages_raw(temp_path,all_pages,dpi=(300,300)):
    if all_pages:
        os.makedirs(temp_path, exist_ok=True)

        # Save images to the temporary directory
        for i in tqdm(range(0,len(all_pages))):
            image = all_pages[i]

            image_path = os.path.join(temp_path,"page_"+str(i+1)+".tiff")
            # image = Image.fromarray(image)
            image.save(image_path,dpi=dpi)
            image.save(image_path.split('.tiff')[0]+'.png',dpi=dpi)


def preprocess_image(image,deskew=True,img_thresh=False,add_border=False,border_width=10,border_color=(128,128,128),img_type='rgb'):
    assert img_type=='rgb' or img_type=='gray'

    # expects rgb image
    if img_type=='rgb':
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    if deskew==True:
        image = deskew_image(image,cval=1.0)

    if add_border==True:
        image = add_image_border(image,border_width,border_color)

    if img_thresh==True:
        ret,image = cv2.threshold(image,220,255,cv2.THRESH_BINARY)

    if img_type=='rgb':
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    return image


## rotate using tesseract
def check_orientation_tesseract(img):
    newdata = pytesseract.image_to_osd(np.array(img))
    orgi_orientation = re.search('(?<=Rotate: )\d+', newdata).group(0)
    # check image orientation
    if orgi_orientation != 0:
        return rotate_image_tesseract(img)
    else:
        return img

## pip install opevcv-python==4.1.0.25 (this functionality works only in v4.1.0.25 )
def rotate_image_tesseract(image, center=None, scale=1.0):
    angle = 360 - int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    mmm = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, mmm, (w, h))

    return rotated


def trim_image_borders(im,margin=10):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        x,y,x_max,y_max = bbox
        x = max(x-margin,0)
        y = max(y-margin,0)
        x_max = min(x_max+margin,im.size[0]-1)
        y_max = min(y_max+margin,im.size[1]-1)
        bbox = (x,y,x_max,y_max)
        return im.crop(bbox)
    else:
        return im
