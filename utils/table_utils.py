
from utils.common_functions import *
from utils.load_and_preprocess_utils import add_image_border
import pytesseract
import os
import cv2
#from pytesseract import Output
import numpy as np
import skimage
from utils.load_and_preprocess_utils import display_multi

def apply_morph(img,morph_operation=cv2.MORPH_CLOSE,kernel=(3,3),iterations=1):
    img = cv2.morphologyEx(img, morph_operation, kernel=kernel,iterations=iterations)
    return img

## case type in case of tables
def find_case_type_from_tables(extracted_text,all_case_types):
    '''
    finds tag lines in extracted text
    '''
    det_case_type = None
    for c in ['Case Type','CaseType']:
        c_t_idx = find_element_index(extracted_text,c)
        if c_t_idx!=[]:
            break


    if c_t_idx!=[] and len(extracted_text)>1:
        tag_line = extracted_text[c_t_idx+1]
        for in_tag in all_case_types:
            if in_tag in tag_line:
                return in_tag,tag_line
        else:
            return None,None
    else:
        return None,None



def test_cropped_to_text(in_crop):
    extracted_text = pytesseract.image_to_string(in_crop)
    print(extracted_text)
    if extracted_text != '\x0c':
        extracted_text = delete_empty_lines(extracted_text)

    return extracted_text
    # print('----------------------')
    # print(extracted_text)

def draw_fake_line(img,line_hei=0.5,line_thick=4):
    '''
    drawing a fake line in the middle of the document to
    cater for entries which didnt come inside a table at first i.e name, c/o, etc
    '''
    return cv2.line(img, (0,int(line_hei*img.shape[0])), (img.shape[1],int(line_hei*img.shape[0])), 0, line_thick)


def make_fake_table(img,table_coordinates,h_limit=0.35,w_limit=0.6):
    '''
    pre_reqs = other tables were found
    generate a fake table by finding the maximum y from old tables
    then from y_max to img_height*h_limit and 0:w_limit crops a fake table
    '''
    img_h,img_w = img.shape
    coordis_array = np.array(table_coordinates['cropped'])
    max_y_idx = np.argmax(coordis_array[:,5]) ## format = (x,y,w,h,x+w,y+h)
    max_y = coordis_array[:,5][max_y_idx]
    fake_table = img[max_y:int(img_h*h_limit), :int(img_w*w_limit)]

    ## appending fake_table_coordinates to table_coordinates too
    x,y,w,h = 0, max_y, int(img_w*w_limit), int(img_h*h_limit)-max_y
    table_coordinates['cropped'].append((x,y,w,h,x+w,y+h))

    return fake_table,table_coordinates


## Notice type
def find_notice_tags(all_tags,extracted_text):
    '''
    finds tag lines in extracted text
    '''
    det_notice_type = None
    for nt_tag in ['Notice Type','NoticeType']:
        n_t_idx = find_element_index(extracted_text,nt_tag)
        if n_t_idx!=[]:
            break

    if n_t_idx!=[]:
        if len(extracted_text)==1:
            for tag in list(all_tags.values()):
                if tag.lower() in extracted_text[n_t_idx].lower():
                    det_notice_type = tag
                    break
        elif len(extracted_text)>1:
            for tag in list(all_tags.values()):
                for tag_index in range(n_t_idx,len(extracted_text)):
                    if tag.lower() in extracted_text[tag_index].lower():
                        det_notice_type = tag
                        break
        else:
            pass

    else:
        if len(extracted_text)==1:
            for tag in list(all_tags.values()):
                det_notice_idx = find_element_index(extracted_text,tag)
                if det_notice_idx!=[]:
                    det_notice_type = tag
                    break
        else:
            for tag in ['Please come to:']:#list(all_tags.values()):
                '''
                this is just to cater "please come to:" forms this phrase appear
                in the top line of the big box
                '''
                det_notice_idx = find_element_index(extracted_text,tag)
                if det_notice_idx!=[] and det_notice_idx==0:
                    det_notice_type = tag
                    break

    if det_notice_type!=None:
        return find_key_by_val(all_tags,det_notice_type)
    else:
        return det_notice_type

def extract_all_tables(img,sp_name,path_cropped,filter_tables=True,save_crops=False,save_overlay=True,save_ext='.jpg',fake_table=True,fake_line=False,resize=2):
    ## https://github.com/arnavdutta/Table-Detection-Extraction/tree/master

    if save_crops==True or save_overlay==True:
        if not os.path.isdir(path_cropped):
            os.mkdir(path_cropped)

    ## Resizing to make text predictions better
    if resize!=None:
        in_img_h,in_img_w = img.shape
        img = cv2.resize(img,(int(in_img_w*resize),int(in_img_h*resize)))


    img_gray = img.copy()
    im_h,im_w = img_gray.shape

    if fake_line==True:
        img_gray = draw_fake_line(img_gray.copy(),line_hei=0.4,line_thick=4)


    ret,img_bin = cv2.threshold(img_gray,220,255,cv2.THRESH_BINARY)
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = (np.array(img_gray).shape[1])//120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=2)#3
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    kernel_length_h = (np.array(img_gray).shape[1])//40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=2)#3
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    overlayed_tables = []
    all_tables = {}
    all_tables['cropped'] = []
    table_coordinates = {}
    table_coordinates['cropped'] = []

    count = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cropped = img_gray[y:y+h, x:x+w]
        # if (w>80 and h>20) and w>3*h:
        if filter_tables==True: ## in case of resize==2
            if (w>50 and h>20) and (y+h)<(im_h*0.5) and y>(im_h*0.05) and len(np.unique(cropped))!=1 and (w>h):
                count += 1
                cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0), 2)
                all_tables['cropped'].append(cropped)
                table_coordinates['cropped'].append((x, y, w,h, x+w,y+h))

                if save_crops==True:
                    path_crp = os.path.join(path_cropped,sp_name+'_'+'crop_'+str(count)+save_ext)
                    cv2.imwrite(path_crp, cropped)
        else:
            count += 1
            cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 0), 2)
            all_tables['cropped'].append(cropped)
            table_coordinates['cropped'].append((x, y, w,h, x+w,y+h))

            if save_crops==True:
                path_crp = os.path.join(path_cropped,sp_name+'_'+'crop_'+str(count)+save_ext)
                cv2.imwrite(path_crp, cropped)

    if save_overlay==True:
        path_overlay = os.path.join(path_cropped,sp_name+'_bb'+save_ext)
        cv2.imwrite(path_overlay, img)
        # cv2.imwrite("/content/tb/table_detect_" + sp_name+save_ext, table_segment)

    all_tables['table_segment'] = table_segment
    all_tables['table_overlayed'] = img

    if fake_table==True:

        all_tables_fake = all_tables.copy()
        table_coordinates_fake = table_coordinates.copy()

        ## fake table
        if table_coordinates['cropped']!=[]:
            fake_table,table_coordinates_fake = make_fake_table(img,table_coordinates_fake,h_limit=0.35,w_limit=0.6)
            all_tables_fake['cropped'].append(fake_table)
            '''Returns fake table with fake coordinates '''
            return all_tables_fake,table_coordinates_fake
    '''Else return original table with original coordinates '''
    return all_tables,table_coordinates


def cropped_tables_to_data(all_tables,table_coordinates,fake_table,notice_types_tables,all_case_types,
                           thresh_filter=False,op_borders=None,display_info=False,op_morph=None,narrow_cell=True):
    name_co = None
    co = None
    # case_type = None
    case_type_tables = None
    notice_type = None

    applicant = None
    petitioner = None
    beneficiary = None
    page_num = None
    for cr in range(0,len(all_tables['cropped'])):
        '''Fake table condition is to limit the finding of only co and name_co in the fake table. As only name and name_co exist in that.
        For everything else do not look into the fake table. It also assumes that fake table is appended at the last index.
        So if fake_table==False look in all the indexes but if its true, do not look into it exept for name and name_co'''
        fake_table_condition = (fake_table==True and cr<len(all_tables['cropped'])-1) or (fake_table==False)
        cropped_img = all_tables['cropped'][cr]

        if min(cropped_img.shape)==0:
            continue

        if narrow_cell==True:
            cropped_contours,eroded_img = erode_and_find_contours(cropped_img,erode_iters=10)
            if display_info==True:
                display_multi(eroded_img)

            if len(cropped_contours)!=0:
                cropped_img = cropped_contours[0]


        crp_h,crp_w = cropped_img.shape

        if op_borders!=None:
            ## add border
            cropped_img = add_image_border(np.array(cropped_img),op_borders,(255,255,255))  ## adding white border to make text not so narroly cropped

        ## testing simple vs otsu thresholding
        # cropped_img = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if thresh_filter!=None:
            cropped_img = skimage.util.img_as_ubyte(cropped_img > thresh_filter(cropped_img))

        if op_morph!=None:
            cropped_img = op_morph(cropped_img)

        ###extracted_text = pytesseract.image_to_string(cropped_img)
        data_dict = pytesseract.image_to_data(cropped_img, output_type=pytesseract.Output.DICT)
        data_dict = clean_the_data_dict(data_dict,del_list=['',' '])
        extracted_text,all_lines,all_bboxes = text_to_old_format(data_dict)

        if display_info==True:
            print(extracted_text)
            display_multi(cropped_img)


        if extracted_text != '\x0c':
            ##extracted_text = delete_empty_lines(extracted_text)

            if co==None and name_co==None:
                ## for searching c/o and name_co
                cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = try_with_thresholding(cropped_img)
                co,name_co = find_name_co_v2(extracted_text_thresh,all_lines_thresh,all_bboxes_thresh,dist_limit=30)
                if display_info==True:
                    print(extracted_text_thresh)
                    display_multi(cropped_img_thresh)


            if applicant==None and fake_table_condition==True:
                '''
                think of limiting the len(extracted_text)<=3
                because in some cases there appears "Applicant Name/Email address"
                which you dont want to detect
                in some cases its Applicant then next line name,
                in other case its Applicant then next line some junk id then in third line some name
                '''

                for ap_tag in ['applicant','applica']:
                    indices = find_element_index(extracted_text,ap_tag)
                    if indices!=[] and ('Application'.lower() not in extracted_text[indices].lower()):
                        applicant = extracted_text[-1]
                        break



                applicant = clean_text(applicant)


            if petitioner==None  and fake_table_condition==True:
                indices = find_element_index(extracted_text,"petitioner")
                if indices!=[] and len(extracted_text)!=1:
                    petitioner = extracted_text[-1]
                elif indices!=[] and len(extracted_text)==1:
                    cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = try_with_thresholding(cropped_img)
                    indices = find_element_index(extracted_text_thresh,"petitioner")
                    petitioner = extracted_text_thresh[-1]
                    if display_info==True:
                        print('From Thresh Image:')
                        print(extracted_text_thresh)
                        display_multi(cropped_img_thresh)

                petitioner = clean_text(petitioner)

            if beneficiary==None  and fake_table_condition==True:
                indices = find_element_index(extracted_text,"beneficiary")
                if indices!=[] and len(extracted_text)!=1:
                    beneficiary = extracted_text[-1]
                elif indices!=[] and len(extracted_text)==1:
                    cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = try_with_thresholding(cropped_img)
                    indices = find_element_index(extracted_text_thresh,"beneficiary")
                    beneficiary = extracted_text_thresh[-1]
                    if display_info==True:
                        print('From Thresh Image:')
                        print(extracted_text_thresh)
                        display_multi(cropped_img_thresh)

                beneficiary = clean_text(beneficiary)


            if case_type_tables==None  and fake_table_condition==True:
                case_type_tables,case_type_line = find_case_type_from_tables(extracted_text,all_case_types)
                if case_type_tables==None and find_element_index(extracted_text,'Case Type')!=[] or find_element_index(extracted_text,'CaseType')!=[]:
                    cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = try_with_thresholding(cropped_img)
                    case_type_tables,case_type_line = find_case_type_from_tables(extracted_text_thresh,all_case_types)
                    if display_info==True:
                        print('From Thresh Image:')
                        print(extracted_text_thresh)
                        display_multi(cropped_img_thresh)

            if notice_type==None  and fake_table_condition==True:
                notice_type = find_notice_tags(notice_types_tables,extracted_text)
                if notice_type==None and (find_element_index(extracted_text,"Notice Type")!=[] or find_element_index(extracted_text,"NoticeType")!=[]):
                    '''if notice type is found but text under it is not found(happens sometimes in case of receipt) then try thresholding'''
                    cropped_img_thresh,extracted_text_thresh,all_lines_thresh,all_bboxes_thresh = try_with_thresholding(cropped_img)
                    notice_type = find_notice_tags(notice_types_tables,extracted_text_thresh)

                    if display_info==True:
                        print('From Thresh Image:')
                        print(extracted_text_thresh)
                        display_multi(cropped_img_thresh)


            if page_num==None  and fake_table_condition==True:
                indices = find_element_index_complete(extracted_text,"page")
                if indices!=[]:
                    '''
                    sometimes page number page k agay bhe likha ho ga us ka kia karay ga bc
                    '''
                    page_num = extracted_text[indices+1]
                    page_num = page_num.strip()[0]


    all_tables_info = {}
    all_tables_info['name_co'] = name_co
    all_tables_info['co'] = co
    # all_tables_info['case_type'] = case_type
    all_tables_info['case_type_tables'] = case_type_tables
    all_tables_info['notice_type'] = notice_type

    all_tables_info['applicant'] = applicant
    all_tables_info['petitioner'] = petitioner
    all_tables_info['beneficiary'] = beneficiary

    all_tables_info['page_num'] = page_num

    return all_tables_info
