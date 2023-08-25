import argparse
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import shutil
import re
import cv2
import numpy as np
import skimage
# from skimage.transform import rotate
from PyPDF2 import PdfWriter, PdfReader

import math
from tqdm import tqdm
from glob import glob
import shutil
import json

from utils.load_and_preprocess_utils import preprocess_image
from utils.table_utils import extract_all_tables,cropped_tables_to_data
from utils.non_table_utils import img_to_all_data_dicts,data_dict_to_data
from utils.non_table_utils import erode_and_find_contours
from utils.formatting_utils import *
from utils.notice_and_case_types import *
from utils.common_functions import *

def process_and_save(all_pages,junk_folder=None,display_info=False,st_i=None,end_i=None,save_jpegs=False,op_orientation=True):
    if save_jpegs==True:
        assert junk_folder!=None

    ## Settings
    ## pre processing options
    op_dpi = (300,300)
    op_deskew = True
    op_img_thresh = False   ## True makes I to 1 so keep it false
    op_borders = (False,10,(0,0,0))

    ## thresholding and morphology
    thresh_filter = None
    op_morph = None 

    ## Limit name for formatting to 2 
    limit_name_idx = 2

    all_final_names = []
    all_final_appends = []
    ## Init dictionaries
    all_tables_info = {}
    all_doc_info = {}

    info_dict_old = {}
    info_dict_old['name'] = None
    info_dict_old['case_type'] = None
    info_dict_old['doc_type'] = None

    info_dict = {}
    info_dict['name'] = None
    info_dict['case_type'] = None
    info_dict['doc_type'] = None

    corrupts = []
    if st_i==None:
        st_i = 0
    if end_i==None:
        end_i = len(all_pages)

    for i in tqdm(range(st_i,end_i)):
        try:
            pg_num = i+1

            ## Init dictionaries
            all_tables_info = {}
            all_doc_info = {}
            info_dict = {}

            sp_name = 'page_'+str(pg_num)
            img = cv2.cvtColor(np.array(all_pages[i]),cv2.COLOR_RGB2GRAY)

            if op_orientation==True:
                try:
                    img = check_orientation_tesseract(np.array(img))
                except:
                    pass

            img = preprocess_image(img,deskew=op_deskew,img_thresh=op_img_thresh,\
                                    add_border=op_borders[0],border_width=op_borders[1],border_color=op_borders[2],img_type='gray')

            img_h,img_w = img.shape

            init_type = detect_tables_vs_nontables(img,display_info=display_info)

            ## ------------------------------------------
            ### Detection using tables
            ## ------------------------------------------
            ## resize image to make sure the capital letters are 30pixels in height
            if init_type=='Tables':
                table_resize = 2
                fake_table = True
                all_tables,table_coordinates = extract_all_tables(img.copy(),sp_name,path_cropped="page_tables/",filter_tables=True,
                                                                save_crops=False,save_overlay=False,fake_table=fake_table,fake_line=False,
                                                                resize=table_resize)


                narrow_cell = True
                all_tables_info = cropped_tables_to_data(all_tables,table_coordinates,fake_table,notice_types_tables,all_case_types_tables,thresh_filter=thresh_filter,
                                            op_borders=10,display_info=display_info,op_morph=op_morph,narrow_cell=narrow_cell)

                all_tables_info['init_type'] = init_type
                # if all_tables_info['notice_type'] in notice_types_tables.keys():
                info_dict = final_naming_tables(all_tables_info)

            else:
                ## ------------------------------------------
                ### Detection without tables
                ## ------------------------------------------
                all_data_dicts = img_to_all_data_dicts(img)
                all_doc_info = data_dict_to_data(all_data_dicts,all_doc_tags,all_doc_tags_partial,all_case_types_tables)
                all_doc_info['init_type'] = init_type
                info_dict = final_naming_nontables(all_doc_info)


            append_page = decide_page(info_dict,info_dict_old,notice_types_tables,all_doc_tags)
            ## update old dict
            info_dict_old = info_dict

            ## Extract important info
            final_name = info_dict['name']
            case_type = info_dict['case_type']
            doc_type = info_dict['doc_type']

            ## Correct final formats
            final_name = format_name(final_name)
            case_type = correct_final_case_type(case_type)
            doc_type = correct_final_doc_type(doc_type)

            ## If doc_type==ASC read from barcode
            if doc_type=='ASC':
                try:
                    bcode_417 = read_bcode_from_img(img)
                    bcode_name = get_name_from_bcode(bcode_417,limit_name_idx=limit_name_idx)
                    final_name = bcode_name
                except:
                    pass

            ## ------------------------------------------
            ## saving images
            ## ------------------------------------------

            save_name_full = None
            save_name_full = str(pg_num)+'_'+str(final_name)+'_'+str(case_type)+'_'+str(doc_type)+".jpg"
            if save_jpegs==True:
                save_name_full = os.path.join(junk_folder,save_name_full)
                cv2.imwrite(save_name_full,img,[cv2.IMWRITE_JPEG_QUALITY, 10])

            all_final_names.append(save_name_full)
            all_final_appends.append(append_page)


        except:
            print('')
            print('########################################################')
            print('Corrupt:',pdf_name,pg_num)
            print('########################################################')
            corrupts.append((pdf_name,pg_num))

    return all_final_names,all_final_appends,corrupts


def process_all_documents(path_write_all,all_documents):
    display_info=False
    save_junk = True
    save_images = True
    save_zips = True
    op_orientation = True

    if save_zips==True:     assert save_images==True

    make_folder(path_write_all)

    path_write_pdfs = os.path.join(path_write_all,'pdfs/')
    make_folder(path_write_pdfs)

    path_write_junk = os.path.join(path_write_all,'junk/')
    if save_junk==True:     make_folder(path_write_junk)

    path_write_images = os.path.join(path_write_all,'images/')
    if save_images==True:   make_folder(path_write_images)

    path_write_zips = os.path.join(path_write_all,'zips/')
    if save_zips==True:     make_folder(path_write_zips)


    st_i = None
    num_forms = None
    end_i = None

    all_corrupts = []
    for doc_index in tqdm(range(0,len(all_documents[:]))):
        pdf_path = all_documents[doc_index]
        pdf_name,_ = os.path.splitext(pdf_path.split('/')[-1])
        print(pdf_path,pdf_name)
        print('Size(Mb):',round(os.path.getsize(pdf_path)/1024/1024))

        ## convert pdf to images
        all_pages = convert_from_path(pdf_path)
        print('Total Pages:',len(all_pages))

        junk_folder = os.path.join(path_write_junk,pdf_name+'_junk/')
        if save_junk==True:    make_folder(junk_folder)

        pdf_folder = os.path.join(path_write_pdfs,pdf_name+'_pdfs/')
        make_folder(pdf_folder)

        images_folder = os.path.join(path_write_images,pdf_name+'_images/')
        if save_images==True:   make_folder(images_folder)

        all_final_names,all_final_appends,corrupts = process_and_save(all_pages,junk_folder,display_info=display_info,
                                                                      st_i=st_i,end_i=end_i,save_jpegs=save_junk,op_orientation=op_orientation)
        all_corrupts.append(all_corrupts)

        all_files_pagewise = make_pagewise_list(all_final_names,all_final_appends)
        break_pdf_to_files(pdf_path,pdf_folder,all_files_pagewise,compress_pdf=True)

        selected_pages = all_pages[st_i:end_i]
        if save_images==True:   break_pdf_to_images(selected_pages,images_folder,all_files_pagewise,jpeg_quality=10)

        zip_name = os.path.join(path_write_zips,pdf_name+'.zip')
        if save_zips==True:     shutil.make_archive(zip_name.split('.zip')[0], 'zip', images_folder)
        ##if save_zips==True:     !zip -r {zip_name} {images_folder}

        with open(os.path.join(path_write_all,'corrupt_pages.json'), 'w', encoding='utf-8') as f:
            json.dump(all_corrupts, f, ensure_ascii=False, indent=4)
