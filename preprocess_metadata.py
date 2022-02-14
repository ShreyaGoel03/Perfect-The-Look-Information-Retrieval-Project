import pandas as pd
import numpy as np
import json
from skimage import io
import cv2
from google.colab.patches import cv2_imshow
from urllib import error
import sys
from sklearn.model_selection import train_test_split as split

def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)

def generate_url():
    column_headers = ['product_id','scene_id','bbox_left','bbox_top','bbox_right','bbox_bottom','category','scene_url','product_url']
    data_list = []
    for obj in fashion_list:
    inner_list = []
    for key,value in obj.items():
        if key == 'bbox':
        for val in value:
            inner_list.append(val)
        else:
        inner_list.append(value)
    inner_list.append(category_dict[obj['product']])
    scene_url = convert_to_url(obj['scene'])
    product_url = convert_to_url(obj['product'])
    inner_list.append(scene_url)
    inner_list.append(product_url)
    data_list.append(inner_list)
    meta_data = pd.DataFrame(data_list,columns=column_headers)
    meta_data.to_csv('data/STL-Dataset/meta_data.csv',index=False)

def remove_invalid_links():
    data = pd.read_csv('data/STL-Dataset/meta_data.csv')
    defective_links = []
    for i,row in data.iterrows():        
        error_flag = 0
        try:
            img = io.imread(row['product_url'])
        except error.HTTPError as e:
            error_flag = 1    
        try:
            img = io.imread(row['scene_url'])
        except error.HTTPError as e:
            error_flag = 1
        if error_flag:
            defective_links.append(i)
    data_proper = data.drop(defective_links)
    data_proper.to_csv('data/STL-Dataset/meta_data_final.csv',index=False)

if __name__ == "__main__":
    fashion_list = []
    for line in open('data/STL-Dataset/fashion.json','r'):
        fashion_list.append(json.loads(line))
    generate_url(fashion_list)
    remove_invalid_links()
    data = pd.read_csv('data/STL-Dataset/meta_data_final.csv')
    train, validate, test = np.split(data.sample(frac=1, random_state=42), [int(.8*len(data)), int(.9*len(data))])
    train.to_csv('data/STL-Dataset/train_data.csv',index=False)
    validate.to_csv('data/STL-Dataset/validate_data.csv',index=False)
    test.to_csv('data/STL-Dataset/test_data.csv',index=False)


