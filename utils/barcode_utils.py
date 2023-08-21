import zxingcpp
def read_bcode_from_img(img):
    bcode_results = zxingcpp.read_barcodes(img)
    for result in bcode_results:
        if result.format==zxingcpp.BarcodeFormat.PDF417:
            return result

    return None

def get_name_from_bcode(bcode_417):
    name_keys = ['g','h','i']
    bcode_text = bcode_417.text.split('name')[-1].split(',')

    bcode_name = []
    for i in range(0,len(bcode_text)):
        if bcode_text[i][0]==name_keys[0] and bcode_text[i+1][0]==name_keys[1] and bcode_text[i+2][0]==name_keys[2]:
            bcode_name.append(bcode_text[i][1:])
            bcode_name.append(bcode_text[i+1][1:])
            bcode_name.append(bcode_text[i+2][1:])
        elif bcode_text[i][0]==name_keys[0] and bcode_text[i+1][0]==name_keys[1]:
            bcode_name.append(bcode_text[i][1:])
            bcode_name.append(bcode_text[i+1][1:])
        elif bcode_text[i][0]==name_keys[0]:
            bcode_name.append(bcode_text[i][1:])

    return bcode_name
