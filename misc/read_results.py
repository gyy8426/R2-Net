import dill as pkl
import os
filePath = './output/'
fileNameHead = 'rel_cat_recall_'

def file_name(file_dir):
    file_list_r = []
    for root, dirs, files in os.walk(file_dir):
        file_list=files
    for file_i in file_list:
        if fileNameHead in file_i:
            file_list_r.append(file_i)
    return file_list_r
    
def print_results(filePath):
    fileNameList = file_name(filePath)
    #print(fileNameList)
    list.sort(fileNameList)
    max_ind = ''
    max_value = 0
    for fileName_i in fileNameList:
        print(fileName_i)
        result_i = pkl.load(open(filePath+fileName_i,'rb'))
        for result_k in result_i.keys():
            if 'mean_rel_recall' in result_i[result_k].keys():
                print(result_k,': ',result_i[result_k]['mean_rel_recall'])
                if result_k == 20:
                    if max_value < result_i[result_k]['mean_rel_recall']:
                        max_value = result_i[result_k]['mean_rel_recall']
                        max_ind = fileName_i
    print(max_ind,max_value)
print_results(filePath)
