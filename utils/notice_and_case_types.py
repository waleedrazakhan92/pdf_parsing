##############################################################
## Notice types( from Tables)
##############################################################

notice_types_tables = {}
## same lines i.e "Notice Type: XYZ"
notice_types_tables['Reopen'] = 'Reopen Notice'
notice_types_tables['Transfer'] = 'Transfer Notice'
notice_types_tables['USCIS'] = 'USCIS Account Access'
notice_types_tables['Approval'] = 'Approval Notice'
## Next lines i.e:
## Notice Type
## Receipt
notice_types_tables['Receipt'] = 'Receipt'
notice_types_tables['Rejection'] = 'Rejection Notice'

'''
Receipt with no "notice type" heading "same as ASC Appointment"  same as "Initial interview"
'''
notice_types_tables['Interview'] = 'Request for Applicant to Appear for '
notice_types_tables['ASC'] = 'ASC Appointment Notice'

# notice_types_tables['Receipt_2'] = 'Receipt'
notice_types_tables['Receipt_2'] = 'Recei' ## if it was cut by line

notice_types_tables['Interview_2'] = 'Please come to:'
notice_types_tables['Cancellation'] = 'Notice of Interview Cancellation'
notice_types_tables['Applicants'] = 'Notice to Applicants'
notice_types_tables['Biometric'] = 'Biometric'


##############################################################
## Document types
##############################################################
all_doc_tags = {}
all_doc_tags['Courtesy_letter'] = 'COURTESY LETTER TO APPLICANT'
all_doc_tags['NIVCC'] = 'NOTICE OF IMMIGRANT VISA CASE CREATION'
all_doc_tags['Withdrawal_Acknowledgment'] = 'ACKNOWLEDGMENT OF WITHDRAWAL'
all_doc_tags['Withdrawal'] = 'WITHDRAWAL'
all_doc_tags['Decision'] = 'DECISION'
all_doc_tags['Decision_Notice'] = 'NOTICE OF DECISION'
all_doc_tags['RFE'] = 'REQUEST FOR EVIDENCE'
# all_doc_tags['RFE_2'] = 'REQUEST FOR EVIDENCE (FORM I-485)'
# all_doc_tags['RFE_3'] = 'REQUEST FOR EVIDENCE (FORM 1-485)'
all_doc_tags['Deficiency_Notice'] = 'DEFICIENCY NOTICE'
# all_doc_tags['DefN_2'] = 'I-693 DEFICIENCY NOTICE'
# all_doc_tags['DefN_3'] = '1-693 DEFICIENCY NOTICE'
all_doc_tags['Oath_Ceremony'] = 'NOTICE OF NATURALIZATION'

## Partial Tags
all_doc_tags_partial = ['REQUEST FOR EVIDENCE','DEFICIENCY NOTICE','NOTICE OF NATURALIZATION']

##############################################################
## Case types
##############################################################
# ['I-129F','I-130','I-485','I-131','I-765','N-400','I-751','I-90','I-539']
all_case_types_org = ['-129F','-130','-485','-131','-765','-400','-751','-539','-290B','-90']
all_case_types_tables = ['129F','130','485','131','765','400','751','539','290B','90']
