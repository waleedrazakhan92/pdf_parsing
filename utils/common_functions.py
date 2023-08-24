import cv2
import numpy as np
import skimage
import pytesseract
import re
import math
from pytesseract import Output
from utils.load_and_preprocess_utils import *

def make_folder(in_path):
    if not os.path.isdir(in_path):
        os.mkdir(in_path)
    
def make_folders_multi(*in_list):
    for in_path in in_list:
        make_folder(in_path)

def remove_items(lst, value):
    return [item for item in lst if item != value]


def delete_characters(in_string,del_char_list):
    assert (type(del_char_list)==list or del_char_list==None) and (type(in_string)==str)
    # Filter multiple characters from string
    filtered_chars = filter(lambda item: item not in del_char_list, in_string)
    # Join remaining characters in the filtered list
    out_string = ''.join(filtered_chars)

    return out_string

# ## same as delete_characters
# def remove_multiple_characters(in_string,in_remove):
#     return in_string.translate({ord(i): None for i in in_remove})


def find_element_index(in_list,in_str,del_char_list=None,only_first=True):
    assert type(del_char_list)==str or del_char_list==None
    '''
    sees if a string lies in a line or not
    '''
    ## returns only the index of an input string(partial matching)
    ## for now [0]=Returning only the first occurance
    ## keep in mind s.lower()
    if del_char_list==None:
        indices = [i for i, s in enumerate(in_list) if in_str.lower() in s.lower()]
    else:
        indices = [i for i, s in enumerate(in_list) if delete_characters(in_str.lower(),del_char_list.lower()) in s.lower()]

    if indices!=[]:
        if only_first==True:
            indices = indices[0]

    return indices

def find_element_index_complete(in_list,in_str,only_first=True):
    '''
    used for document type as it matches the WHOLE line with a string
    '''
    ## returns only the index of an input string(partial matching)
    ## for now [0]=Returning only the first occurance
    ## keep in mind s.lower()
    indices = [i for i, s in enumerate(in_list) if in_str.lower() == s.lower()]

    if indices!=[]:
        if only_first==True:
            indices = indices[0]

    return indices


def find_key_by_val(in_dict,in_val):
    value = [i for i in in_dict if in_dict[i].lower()==in_val.lower()]
    return value[0]


def delete_empty_lines(extracted_text):
    extracted_text = extracted_text.split('\n')
    extracted_text = list(filter(lambda x: x not in ['',' ','\x0c'], extracted_text))
    return extracted_text

def clean_text(in_text):
    if in_text == None:
        return in_text
    '''
    trying to clean in_text by first spliting the string by ' '(space)
    then removing the entries which have length=1
    '''
    idx_to_del = []
    sp_in_text = in_text.split(' ')
    for i in range(0,len(sp_in_text)):
        if len(sp_in_text[i])==1:
            idx_to_del.append(i)

    new_in_text = del_list_indexes(sp_in_text, idx_to_del)

    clean_in_text = ''
    for i in range(0,len(new_in_text)):
        if new_in_text[i]!='':
            if i<len(new_in_text)-1:
                clean_in_text+=new_in_text[i]+' '
            else:
                clean_in_text+=new_in_text[i]

    return clean_in_text

def delete_list_indices(in_list,unwanted_indices):
    '''
    delete indices from a list
    '''
    for ele in sorted(unwanted_indices, reverse = True):
        del in_list[ele]

    return in_list

def del_list_indexes(re, id_to_del):
    somelist = [i for j, i in enumerate(re) if j not in set(id_to_del)]
    return somelist

def clean_the_data_dict(data_dict,del_list=['',' ']):
    for del_char in del_list:
        empty_indices = find_element_index_complete(data_dict['text'],del_char,only_first=False)
        for k in data_dict:
            data_dict[k] = delete_list_indices(data_dict[k],empty_indices)

    return data_dict

def erode_and_find_contours(img,erode_iters=10):
    ''' for further narrowing the cropped table cell '''
    ''' Specially filtered for name_co wala contour'''
    eroded_img = generate_contour_image(img,iterations=erode_iters)
    e_img_h,e_img_w = eroded_img.shape[:2]
    contours = find_contour_bboxes(eroded_img)
    if len(contours)>1:
        filtered_contours = filter_contour_indices(contours,e_img_h,e_img_w,y_limit=0.4,x_limit=0.4,custom_filters=filter_for_all_contours)
        all_contour_imgs = filter_contours(img,contours,filtered_contours=filtered_contours)
    elif len(contours)==1:
        all_contour_imgs = filter_contours(img,contours,filtered_contours=None)
    else:
        all_contour_imgs = []

    return all_contour_imgs,eroded_img

def filter_for_all_contours(x,y,w,h,y_limit,x_limit,img_h,img_w): ## 50 is in case of resize==2
    return (x<=(img_w*x_limit)) and (w>=h)  and (y<(img_h*y_limit)) and ((y+h)>40) and ((x+w)>50)


def generate_contour_image(img,iterations=10,kernel=np.ones((5,5),np.uint8)):
    ## erode since the image has black text and white bg
    eroded_img = cv2.erode(img,kernel,iterations=iterations)
    eroded_img = cv2.bitwise_not(eroded_img)
    ret, eroded_img = cv2.threshold(eroded_img, 150, 255, cv2.THRESH_BINARY)
    return eroded_img

def find_contour_bboxes(eroded_img):
    contours, hierarchy = cv2.findContours(eroded_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def crop_contour(img,contours,cont_id):
    x,y,w,h = cv2.boundingRect(contours[cont_id])
    cnt_img = img[y:y+h,x:x+w]
    return cnt_img

def filter_contours(img,contours,filtered_contours=None):
    all_contours = []
    for i in range(0,len(contours)):
        if filtered_contours!=None:
            if i in filtered_contours:
                all_contours.append(crop_contour(img,contours,i))
        else:
            all_contours.append(crop_contour(img,contours,i))

    return all_contours

def filter_contour_indices(contours,img_h,img_w,y_limit,x_limit,custom_filters):
    filtered_contours = []
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if custom_filters(x,y,w,h,y_limit,x_limit,img_h,img_w)==True:
            filtered_contours.append(i)

    return filtered_contours

def in_filters(x,y,w,h,y_limit,x_limit,img_h,img_w):
    return (x<=(img_w*x_limit)) and (w>=h)  and ((y+h)<(img_h*y_limit))


def calculate_distance(pt_1,pt_2):
    return round(math.dist(pt_1,pt_2))


def find_name_co_v2(all_lines_str,all_lines,all_bboxes,dist_limit=50):
    co_list = None
    name_co = None
    for in_str in ['C/O','c/o','C/o','c/O','clo','c1o']:
        indices = find_element_index(all_lines_str,in_str)
        if indices!=[]:
            ## removing everything except alphanueric and / because / comes in c/o
            co_line = re.sub(r'[^\w\s\/]', '', all_lines_str[indices])

            if co_line.strip().lower()[:3] != in_str.lower():
                ## meaning C/O: doesn't appear at the start of the line
                return None,None
            else:
                bad_dist_indices = check_coordinates_consecutive(all_bboxes,indices,dist_limit=dist_limit)
                co_list = delete_list_indices(all_lines[indices],bad_dist_indices)
                co_list = ' '.join(co_list)

        if co_list!=None:
            ## C/O found
            bad_dist_indices = check_coordinates_consecutive(all_bboxes,indices-1,dist_limit=dist_limit)
            name_co = delete_list_indices(all_lines[indices-1],bad_dist_indices)
            name_co = ' '.join(name_co)
            break

    return co_list,name_co

def text_to_old_format(data_dict):
    one_line = []
    all_lines = []

    bbox_line = []
    all_bboxes = []

    one_line_str = ''
    all_lines_str = []
    for i in range(0,len(data_dict['text'])):
    # for i in tqdm(range(15,40)):
        if data_dict['word_num'][i]==1:
            if i!=0:
                all_lines.append(one_line)
                all_lines_str.append(one_line_str)
                all_bboxes.append(bbox_line)

            one_line = [data_dict['text'][i]]
            one_line_str = data_dict['text'][i]
            bbox_line = [(data_dict['left'][i],data_dict['top'][i],data_dict['width'][i],data_dict['height'][i])]
        else:
            one_line_str = one_line_str+' '+data_dict['text'][i]
            one_line.append(data_dict['text'][i])
            bbox_line.append((data_dict['left'][i],data_dict['top'][i],data_dict['width'][i],data_dict['height'][i]))

    all_lines.append(one_line)
    all_lines_str.append(one_line_str)
    all_bboxes.append(bbox_line)

    return all_lines_str,all_lines,all_bboxes


def try_with_thresholding(cropped_img):
    cropped_img_thresh = skimage.util.img_as_ubyte(cropped_img > skimage.filters.threshold_li(cropped_img))
    data_dict_thresh = pytesseract.image_to_data(cropped_img_thresh, output_type=pytesseract.Output.DICT)
    data_dict_thresh = clean_the_data_dict(data_dict_thresh,del_list=['',' '])
    extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = text_to_old_format(data_dict_thresh)

    return cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh


def check_coordinates_consecutive(all_bboxes,indices,dist_limit=50):
    '''
    measures the distance between the bboxes of detected words
    if distance is > than a threshold then the two words don't belong to same group
    specificaly made for name in front of RE: . E.g. RE: XYZ junk 412-213-
    So sometimes the junk values are also detected as part of name
    '''
    bad_dist_indices = []
    for bb in range(0,len(all_bboxes[indices])-1):
        ## (left,top,width,height)
        lft_1,top_1,wid_1,hei_1 = all_bboxes[indices][bb]
        lft_2,top_2,wid_2,hei_2 = all_bboxes[indices][bb+1]
        bb_dist = calculate_distance((lft_1+wid_1,top_1),(lft_2,top_2))
        if bb_dist>dist_limit:
            bad_dist_indices.append(bb+1)


    return bad_dist_indices


def find_case_type_v2(all_lines_str,all_lines,all_bboxes,all_case_types,in_str,dist_limit=30):
    '''
    finds RE: and then finds case types from case type list. If case type distance <thres with RE then case type is returned
    '''
    (indices_1,in_str_idx_1),(lft_1,top_1,wid_1,hei_1) = get_word_idx_line_pos(all_lines_str,all_lines,all_bboxes,in_str)

    # found_tags = []
    if indices_1!=[]:
        for in_tag in all_case_types:
            try:
                (indices_2,in_str_idx_2),(lft_2,top_2,wid_2,hei_2) = get_word_idx_line_pos(all_lines_str,all_lines,all_bboxes,in_tag)
                if indices_2!=[]:
                    bb_dist = calculate_distance((lft_1,top_1+hei_1),(lft_2,top_2))
                    if bb_dist<dist_limit:
                        return in_tag, all_lines[indices_2][in_str_idx_2]
            except:
                pass

    return None,None

def get_word_idx_line_pos(all_lines_str,all_lines,all_bboxes,in_str):
    '''
    finds the line number and word position of a given string
    '''
    indices = find_element_index(all_lines_str,in_str)
    if indices!=[]:
        in_str_idx = find_element_index(all_lines[indices],in_str)
        lft_1,top_1,wid_1,hei_1 = all_bboxes[indices][in_str_idx]
        return (indices,in_str_idx),(lft_1,top_1,wid_1,hei_1)
    else:
        return None


def find_a_file_num_name(all_lines_str):
    '''In few docs there is A file number: xyz and then in above line there is name of the person '''
    indices = find_element_index(all_lines_str,'A file number')
    if indices!=[] and indices>0:
        return all_lines_str[indices-1]
    else:
        return None

def find_attorney_copy(all_lines_str):
    '''In few docs there is Attorney Copy written '''
    indices = find_element_index(all_lines_str,'Attorney Copy')
    if indices!=[]:
        return True
    else:
        return None


def filters_for_table_detection(x,y,w,h,y_limit,x_limit,img_h,img_w):
    return (w>=(img_w*x_limit)) and (w>=h) and (w<(img_w*0.9))

def filters_for_date_detection(x,y,w,h,y_limit,x_limit,img_h,img_w):
    return (w<=(img_w*x_limit)) and (w>=2*h)

def detect_tables_vs_nontables(img,display_info=False):
    init_type = None

    img = np.array(trim_image_borders(Image.fromarray(img)))

    img_h,img_w = img.shape
    # test_img = img[:int(img_h*0.06),int(img_w*0.5):]
    test_img = img[:int(img_h*0.12),:]
    test_img = skimage.util.img_as_ubyte(test_img > skimage.filters.threshold_otsu(test_img))
    # extracted_text = pytesseract.image_to_string(test_img)
    # extracted_text = delete_empty_lines(extracted_text)

    data_dict = pytesseract.image_to_data(test_img, output_type=Output.DICT)
    data_dict = clean_the_data_dict(data_dict,del_list=['',' '])
    extracted_text,all_lines,all_bboxes = text_to_old_format(data_dict)

    

    if display_info==True:
        print(' ')
        print('---------------------------------------')
        print('Tables')
    if display_info==True:
        print(extracted_text)
        display_multi(test_img)

    for in_txt in ['Action','Does not grant']:
        indices = find_element_index(extracted_text,in_txt)
        if indices!=[]:
            init_type = 'Tables'
            break

    if init_type==None:
        indices = find_element_index(extracted_text,'IN THE DISTRICT COURT OF APPEAL')
        if indices!=[]:
            init_type = 'Unidentified'

    if init_type==None:
        #####################################################
        ### Search for Normals in tables text too. For cases like oath ceremony and date that appease in a complete line

        for in_txt in ['Oath','Oath Ceremony','Naturalization']:
            indices = find_element_index(extracted_text,in_txt)
            if indices!=[]:
                init_type = 'Normal'
                break

        for in_txt in ['January','February','March','April','May','June','July','August','September','October','November','December']:
            indices = find_element_index(extracted_text,in_txt)
            if indices!=[]:
                bad_dist_indices = check_coordinates_consecutive(all_bboxes,indices,dist_limit=50)

            if indices!=[] and len(extracted_text[indices].split(' '))>5 and bad_dist_indices==[]:
                init_type = 'Useless'
                if display_info==True:
                    print(extracted_text)
                    print(extracted_text[indices])
                    display_multi(test_img)
                break
            
            if indices!=[] and find_element_index(extracted_text[indices].split(' '),in_txt)!=0:
                init_type = 'Useless'
                if display_info==True:
                    print(extracted_text)
                    print(extracted_text[indices])
                    display_multi(test_img)
                break

    #####################################################
    if init_type==None:
        eroded_img = cv2.erode(test_img,np.ones((1,5),np.uint8),iterations=5)
        eroded_img = cv2.bitwise_not(eroded_img)
        ret, eroded_img = cv2.threshold(eroded_img, 150, 255, cv2.THRESH_BINARY)
        e_img_h,e_img_w = eroded_img.shape[:2]
        contours = find_contour_bboxes(eroded_img)
        filtered_contours = filter_contour_indices(contours,img_h,img_w,y_limit=None,x_limit=0.5,custom_filters=filters_for_table_detection)
        all_contour_imgs = filter_contours(img,contours,filtered_contours=filtered_contours)
        for cnt_img in all_contour_imgs:
            cnt_img = add_image_border(cnt_img,10,255)
            extracted_text = pytesseract.image_to_string(cnt_img)
            extracted_text = delete_empty_lines(extracted_text)
            if display_info==True:
                display_multi(cnt_img)

            for in_txt in ['Action','Does not grant']:
                indices = find_element_index(extracted_text,in_txt)
                if indices!=[]:
                    init_type = 'Tables'
                    break

            if init_type!=None:
                break

    if display_info==True:
        print(' ')
        print('---------------------------------------')
        print('Normal')


    if init_type==None:

        test_img = img[:int(img_h*0.15),:int(img_h*0.25)]
        # test_img = skimage.util.img_as_ubyte(test_img > skimage.filters.threshold_otsu(test_img))
        extracted_text = pytesseract.image_to_string(test_img)
        extracted_text = delete_empty_lines(extracted_text)

        if display_info==True:
            print(extracted_text)
            display_multi(test_img)

        for in_txt in ['January','February','March','April','May','June','July','August','September','October','November','December',
                        'Oath','Oath Ceremony','Naturalization']:
            indices = find_element_index(extracted_text,in_txt)
            if indices!=[] and (in_txt in ['Oath','Oath Ceremony','Naturalization']):
                init_type = 'Normal'
                break

            elif indices!=[] and len(extracted_text[indices].split(' '))<5 and find_element_index(extracted_text[indices].split(' '),in_txt)==0:
                init_type = 'Normal'
                break

    if init_type==None:
        test_img = img[:int(img_h*0.15),:int(img_h*0.25)]
        eroded_img = cv2.erode(test_img,np.ones((3,5),np.uint8),iterations=5)
        eroded_img = cv2.bitwise_not(eroded_img)
        ret, eroded_img = cv2.threshold(eroded_img, 150, 255, cv2.THRESH_BINARY)
        e_img_h,e_img_w = eroded_img.shape[:2]
        contours = find_contour_bboxes(eroded_img)
        filtered_contours = filter_contour_indices(contours,img_h,img_w,y_limit=None,x_limit=0.4,custom_filters=filters_for_date_detection)
        all_contour_imgs = filter_contours(img,contours,filtered_contours=filtered_contours)

        for cnt_img in all_contour_imgs:
            cnt_img = add_image_border(cnt_img,10,255)
            extracted_text = pytesseract.image_to_string(cnt_img)
            extracted_text = delete_empty_lines(extracted_text)
            if display_info==True:
                print(extracted_text)
                display_multi(cnt_img)
            for in_txt in ['January','February','March','April','May','June','July','August','September','October','November','December',
                        'Oath','Oath Ceremony','Naturalization']:
                indices = find_element_index(extracted_text,in_txt)
                if indices!=[] and (in_txt in ['Oath','Oath Ceremony','Naturalization']):
                    init_type = 'Normal'
                    break

                elif indices!=[] and len(extracted_text[indices].split(' '))<5 and find_element_index(extracted_text[indices].split(' '),in_txt)==0:
                    init_type = 'Normal'
                    break

            if init_type!=None:
                break


    # if init_type==None:
    #     init_type = 'None'

    return init_type

