
from utils.common_functions import *

import pytesseract
import re
import cv2

import math
from pytesseract import Output
from tqdm import tqdm
import numpy as np
import skimage


def find_re_v2(all_lines_str,all_lines,all_bboxes,in_str='RE:',dist_limit=200,dist_check='consecutive'):
    assert dist_check=='fixed' or dist_check=='consecutive'
    indices = find_element_index(all_lines_str,in_str)

    if indices!=[] and (all_lines_str[indices].strip().lower()[:3] != in_str.lower()):
        ## meaning RE: doesn't appear at the start of the line
        return None

    if indices!=[] and dist_check=='fixed':
        in_str_idx = find_element_index(all_lines[indices],in_str)

    if indices!=[] and dist_check=='fixed':
        bad_dist_indices = check_coordinates_fixed(all_bboxes,indices,in_str_idx,dist_limit=dist_limit)
        ## remove the RE iteself too
        # bad_dist_indices.append(in_str_idx)
        re_list = delete_list_indices(all_lines[indices],bad_dist_indices)
        re_list = ' '.join(re_list)

    elif indices!=[] and dist_check=='consecutive':
        bad_dist_indices = check_coordinates_consecutive(all_bboxes,indices,dist_limit=dist_limit)
        re_list = delete_list_indices(all_lines[indices],bad_dist_indices)
        re_list = ' '.join(re_list)
    else:
        re_list = None

    return re_list


def check_coordinates_fixed(all_bboxes,indices,in_str_idx,dist_limit=200):
    '''
    measures the distance between the bboxes of detected words
    if distance is > than a threshold then the two words don't belong to same group
    specificaly made for name in front of RE: . E.g. RE: XYZ junk 412-213-
    So sometimes the junk values are also detected as part of name
    '''
    lft_1,top_1,wid_1,hei_1 = all_bboxes[indices][in_str_idx]
    bad_dist_indices = []
    for bb in range(0,len(all_bboxes[indices])):
        if bb!=in_str_idx:
            ## (left,top,width,height)
            lft_2,top_2,wid_2,hei_2 = all_bboxes[indices][bb]
            bb_dist = calculate_distance((lft_1+wid_1,top_1),(lft_2,top_2))
            if bb_dist>dist_limit:
                bad_dist_indices.append(bb)


    return bad_dist_indices


def detect_and_extract_document_type(image,config,all_tags,all_tags_partial,adjust_img=True):
    im_h,im_w = image.shape

    if adjust_img==True:
        ## in_image = image[int(im_h*0.25):int(im_h*0.5) , int(im_w*0.0):int(im_w*0.8)] ## adjust for document type
        in_image = image[int(im_h*0.0):int(im_h*0.5) , int(im_w*0.0):int(im_w*0.8)] ## adjust for document type
    else:
        in_image = image

    data_dict = pytesseract.image_to_data(in_image, output_type=pytesseract.Output.DICT)
    data_dict = clean_the_data_dict(data_dict,del_list=['',' '])
    extracted_text,all_lines,all_bboxes = text_to_old_format(data_dict)

    if extracted_text != '\x0c':
        # extracted_text = delete_empty_lines(extracted_text)
        doc_type = match_document_tags(all_tags,all_tags_partial,extracted_text)
    else:
        doc_type = None


    return doc_type,(in_image,extracted_text,all_lines,all_bboxes)

def match_document_tags(all_tags,all_tags_partial,extracted_text):
    '''
    finds tag lines in extracted text
    '''
    for tag in list(all_tags.values()):
        if tag not in all_tags_partial:
            det_idx = find_element_index_complete(extracted_text,tag)
        else:
            det_idx = find_element_index(extracted_text,tag)

        if det_idx!=[]:
            return find_key_by_val(all_tags,tag)

    return None


def find_dear_name(extracted_text):
    '''
    if RE wasn't found then searches for dear NAME (can be seen in NIVCC)
    '''
    det_dear_name = None
    det_idx = find_element_index(extracted_text,'Dear')
    if det_idx!=[]:
        det_dear_name = extracted_text[det_idx].strip('Dear').strip(' ').replace(':','')


    return det_dear_name

def img_to_all_data_dicts(img,do_filtering=True,erode_iters=10):
    # in_img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # in_img_thresh = skimage.filters.threshold_mean(img)
    # in_img_thresh = img > in_img_thresh

    img_h,img_w = img.shape
    all_data_dicts = []

    eroded_img = generate_contour_image(img,iterations=erode_iters)
    contours = find_contour_bboxes(eroded_img)
    if do_filtering==True:
        filtered_contours = filter_contour_indices(contours,img_h,img_w,y_limit=0.6,x_limit=1.0,custom_filters=in_filters)
    else:
        filtered_contours = None

    all_contour_imgs = filter_contours(img,contours,filtered_contours=filtered_contours)

    for cont_idx in range(0,len(all_contour_imgs)):
        in_img_thresh = all_contour_imgs[cont_idx]
        data_dict = pytesseract.image_to_data(in_img_thresh, output_type=Output.DICT)
        data_dict = clean_the_data_dict(data_dict,del_list=['',' '])
        all_lines_str,all_lines,all_bboxes = text_to_old_format(data_dict)
        all_data_dicts.append((all_lines_str,all_lines,all_bboxes))

    return all_data_dicts


def data_dict_to_data(all_data_dicts,all_doc_tags,all_doc_tags_partial,all_case_types):
    doc_type = None
    re = None
    case_type = None
    dear_name = None
    co = None
    name_co = None
    a_file_num_name = None
    attorney_copy = None
    case_type_attorney = None

    all_doc_info = {}
    for data_idx in range(0,len(all_data_dicts)):
        (extracted_text,all_lines,all_bboxes) = all_data_dicts[data_idx]

        if doc_type==None:
            doc_type = match_document_tags(all_doc_tags,all_doc_tags_partial,extracted_text)

        if re==None:
            for re_end in [':',';']:
                re = find_re_v2(extracted_text,all_lines,all_bboxes,in_str='RE'+re_end,dist_check='consecutive',dist_limit=50)## dist_limit depends on image size
                if re!=None:
                    re = re.replace('RE','').strip(re_end).strip()
                    if case_type==None:
                        case_type, case_type_org = find_case_type_v2(extracted_text,all_lines,all_bboxes,all_case_types,'RE'+re_end,dist_limit=30)
                if re!=None:
                    break

        if dear_name==None:
            dear_name = find_dear_name(extracted_text)

        if co==None or name_co==None:
            co,name_co = find_name_co_v2(extracted_text,all_lines,all_bboxes,dist_limit=30)
        
        if a_file_num_name==None:
            a_file_num_name = find_a_file_num_name(extracted_text)
            if a_file_num_name!=None:
                '''In few cases there is RE: before the name '''
                a_file_num_name = a_file_num_name.strip('RE:')
                a_file_num_name = a_file_num_name.strip('RE;')
                a_file_num_name = a_file_num_name.strip()

        if attorney_copy==None:
            attorney_copy = find_attorney_copy(extracted_text)

        if attorney_copy!=None and case_type_attorney==None:
            for data_idx_atc in range(0,len(all_data_dicts)):
                (extracted_text_atc,all_lines_atc,all_bboxes_atc) = all_data_dicts[data_idx_atc]

                for ct_at in ['Case Type','CaseType']:
                    indices_ct = find_element_index(extracted_text_atc,ct_at)
                    if indices_ct!=[]:
                        for c_t_2 in all_case_types:
                            if c_t_2 in extracted_text_atc[indices_ct]:
                                case_type_attorney = c_t_2
                                break
                    if case_type_attorney!=None:
                        break
                if case_type_attorney!=None:
                        break

    all_doc_info['doc_type'] = doc_type
    all_doc_info['case_type'] = case_type
    all_doc_info['re'] = re
    all_doc_info['dear_name'] = dear_name
    all_doc_info['co'] = co
    all_doc_info['name_co'] = name_co
    all_doc_info['a_file_num_name'] = a_file_num_name
    all_doc_info['attorney_copy'] = attorney_copy
    all_doc_info['case_type_attorney'] = case_type_attorney
    
    return all_doc_info
