import os
import re
import cv2
from utils.common_functions import delete_characters
import numpy as np
from PyPDF2 import PdfWriter, PdfReader
from utils.common_functions import delete_characters,find_element_index

def format_name(in_name):

    if in_name==None:
        return in_name

    in_name_clean = delete_characters(in_name,[',','.'])
    in_name_clean = in_name_clean.split(' ')

    in_name = in_name.split(' ')
    indices = find_element_index(in_name,',')

    if indices==0 and len(in_name_clean)>1:
        f_name = in_name_clean[0]+', '+in_name_clean[1]
    elif len(in_name_clean)>1 and len(in_name_clean)!=4 and indices==[]:
        f_name = in_name_clean[-1]+', '+in_name_clean[0]
    elif len(in_name_clean)>1 and len(in_name_clean)==4 and indices==[]:
        f_name = in_name_clean[-2]+' '+in_name_clean[-1]+', '+in_name_clean[0]
    else:
        f_name = in_name_clean[0]


    return f_name


def correct_final_doc_type(doc_type):
    ''' if there is "Receipt_2" then make it "Receipt"   if its Decision_Notice then leave it as it is'''
    if doc_type==None or doc_type=='Unidentified':
        return doc_type

    if '_' in doc_type:
        if doc_type.split('_')[1] in ['0','1','2','3','4','5','6','7','8','9','10']:
            doc_type = doc_type.split('_')[0]

    return doc_type

def correct_final_case_type(case_type):

    if case_type!=None:
        case_type = re.sub(r'[^\w\s\/]', '',case_type)
        if case_type=='400':
            case_type = 'N-'+case_type
        elif case_type!='NA' and case_type!='400':
            case_type = 'I-'+case_type
        else:
            case_type = 'NA'

    return case_type

def decide_page(info_dict,info_dict_old,notice_types_tables,all_doc_tags):

    ## tables
    if (info_dict['doc_type']in['Interview','Interview_2']) and (info_dict_old['doc_type']in['Interview','Interview_2']) \
        and (info_dict['case_type']in['485','-485']) and (info_dict_old['case_type']in['485','-485']) \
        and (info_dict['name']==None) and (info_dict_old['name']!=None):
        return True

    elif (info_dict_old['doc_type']in notice_types_tables.keys()) and (info_dict['doc_type']==None) and (info_dict['case_type']==None) and (info_dict['name']==None):
        return True

    elif (info_dict['doc_type']==None) and (info_dict['case_type']==None) and (info_dict['name']==None) and \
        (info_dict_old['doc_type']==None) and (info_dict_old['case_type']==None) and (info_dict_old['name']==None):
        return True

    ## non tables
    elif (info_dict['doc_type']==None) and (info_dict['case_type']==None) and (info_dict['name']==None) and \
        (info_dict_old['doc_type'] in all_doc_tags.keys()) and (info_dict_old['case_type']!=None) and (info_dict_old['name']!=None):
        return True

    elif (info_dict['doc_type']==None) and (info_dict['case_type']==None) and (info_dict['name']==None):
        return True

    else:
        return False


def final_naming_tables(all_tables_info):
    name = None
    case_type = None
    doc_type = None

    ## missing 'USCIS'
    if all_tables_info['notice_type']in['Reopen','Transfer','Approval','Receipt','Receipt_2',
                                        'Rejection','ASC','Interview','Interview_2','Cancellation','Applicants','Biometric']:
        name = all_tables_info['name_co']
        case_type = all_tables_info['case_type_tables']
        doc_type = all_tables_info['notice_type']

    elif all_tables_info['notice_type']=='USCIS':
        name = all_tables_info['applicant']
        case_type = all_tables_info['case_type_tables']
        doc_type = all_tables_info['notice_type']


    if (all_tables_info['name_co']!=None) and \
     ((all_tables_info['name_co'].lower() in 'SOLOMON IMMIGRATION LAW LLC'.lower())  or 'SOLOMON IMMIGRATION'.lower()in  all_tables_info['name_co'].lower()):
        for nm in ['applicant','petitioner']:#'beneficiary'
            if all_tables_info[nm]!=None:
                name = all_tables_info[nm]
                break


    info_dict = {}
    info_dict['name'] = name
    info_dict['case_type'] = case_type
    info_dict['doc_type'] = doc_type
    # info_dict['page'] = 1
    return info_dict


def final_naming_nontables(all_doc_info):
    name = None
    case_type = None
    doc_type = None

    if all_doc_info['init_type']=='Unidentified':
        name = 'Unidentified'
        case_type = 'NA'
        doc_type = 'NA'

    elif (all_doc_info['doc_type']=='RFE') and (all_doc_info['case_type']in['-485','485']):
        name = all_doc_info['re']
        case_type = all_doc_info['case_type']
        doc_type = all_doc_info['doc_type']

    elif (all_doc_info['doc_type']=='RFE') and (all_doc_info['case_type']in['-129F','-130','129F','130']):
        name = all_doc_info['name_co']
        case_type = all_doc_info['case_type']
        doc_type = all_doc_info['doc_type']

    elif (all_doc_info['doc_type']=='Oath_Ceremony'):
        name = all_doc_info['name_co']
        case_type = 'N-400'#all_doc_info['case_type']
        doc_type = all_doc_info['doc_type']

    elif all_doc_info['doc_type'] in ['Deficiency_Notice','Courtesy_letter','Withdrawal','Decision','Decision_Notice','Withdrawal_Acknowledgment']:
        if all_doc_info['name_co']!=None:
            name = all_doc_info['name_co']
        else:
            name = all_doc_info['re']

        case_type = all_doc_info['case_type']
        doc_type = all_doc_info['doc_type']

    elif (all_doc_info['doc_type']=='NIVCC'):
        name = all_doc_info['dear_name']
        case_type = 'NA'#all_doc_info['case_type']
        doc_type = all_doc_info['doc_type']

    elif all_doc_info['doc_type']==None and all_doc_info['a_file_num_name']!=None:
        name = all_doc_info['a_file_num_name']
        case_type = 'NA'
        doc_type = 'NA'
    
    elif all_doc_info['doc_type']==None and all_doc_info['dear_name']!=None and all_doc_info['case_type_attorney']==None \
            and ['Sir'.lower() not in all_doc_info['dear_name'].lower()] and ['Madam'.lower() not in all_doc_info['dear_name'].lower()]:
        name = all_doc_info['dear_name']
        case_type = 'NA'
        doc_type = 'NA'
    
    elif all_doc_info['doc_type']==None and all_doc_info['case_type_attorney']!=None:
        name = all_doc_info['dear_name']
        case_type = all_doc_info['case_type_attorney']
        doc_type = 'NA'
    
    
    info_dict = {}
    info_dict['name'] = name
    info_dict['case_type'] = case_type
    info_dict['doc_type'] = doc_type
    # info_dict['page'] = 1
    return info_dict


def make_pagewise_list(all_final_names,all_final_appends):
    all_files_pagewise = []
    for i in range(0,len(all_final_appends)):
        one_file = []
        if all_final_appends[i]==False:
            curr_name = all_final_names[i]
            one_file.append((curr_name,i))
            for j in range(i+1,len(all_final_appends)):
                if all_final_appends[j]==True:
                    one_file.append(j)
                else:
                    break

            all_files_pagewise.append(one_file)

    return all_files_pagewise


def break_pdf_to_files(path_in_pdf,path_write_pdf,all_files_pagewise,compress_pdf=False,pg_num=True):
    inputpdf = PdfReader(open(path_in_pdf, "rb"))
    for i in range(0,len(all_files_pagewise)):
        one_pdf = all_files_pagewise[i]
        pdf_writer = PdfWriter()
        pdf_name,_ = os.path.splitext(one_pdf[0][0].split('/')[-1])

        one_page = inputpdf.pages[one_pdf[0][1]]

        if compress_pdf==True:
            one_page.compress_content_streams()

        pdf_writer.add_page(one_page)

        for j in range(1,len(one_pdf)):
            one_page = inputpdf.pages[one_pdf[j]]
            if compress_pdf==True:
                one_page.compress_content_streams()

            pdf_writer.add_page(one_page)

        if pg_num==False:
            page_num =  pdf_name.split('_')[0]
            pdf_name = pdf_name.split(page_num+'_')[1]

        pdf_name = os.path.join(path_write_pdf,pdf_name+'.pdf')
        with open(pdf_name, "wb") as outputStream:
            pdf_writer.write(outputStream)

def break_pdf_to_images(all_pages,path_write_images,all_files_pagewise,jpeg_quality=95,pg_num=True):

    for i in range(0,len(all_files_pagewise)):
        one_pdf = all_files_pagewise[i]
        pdf_name,_ = os.path.splitext(one_pdf[0][0].split('/')[-1])
        one_page = np.array(all_pages[one_pdf[0][1]])

        final_img = np.array(one_page)

        for j in range(1,len(one_pdf)):
            one_page = all_pages[one_pdf[j]]
            final_img = np.concatenate((final_img,one_page),axis=1)

        if pg_num==False:
            page_num =  pdf_name.split('_')[0]
            pdf_name = pdf_name.split(page_num+'_')[1]

        pdf_name = os.path.join(path_write_images,pdf_name+'.jpg')
        cv2.imwrite(pdf_name,final_img,[cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
