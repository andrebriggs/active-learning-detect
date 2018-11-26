from collections import defaultdict
import tensorflow as tf
import numpy as np
import csv
import hashlib
from pathlib import Path
import re
from azure.storage.blob import BlockBlobService
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from functions.pipeline.shared.db_access import ImageTagDataAccess
from functions.pipeline.shared.db_provider import PostGresProvider, DatabaseInfo
from functions.pipeline.shared.db_access.db_access_v2 import generate_test_labels,generate_test_image_tags,TestClassifications

# class Config():
#     @staticmethod
#     def parse_file(file_name):
#         config = {}
#         with open(file_name) as file_:
#             for line in file_:
#                 line = line.strip()
#                 if line and line[0] is not "#":
#                     var,value = line.split('=', 1)
#                     config[var.strip()] = value.strip()

#         return config

def extract_image_name(url):
    start_idx = url.rfind('/')+1
    result = url[start_idx:]
    return result

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(file_name, labels, raw_img, tag_map):
    filename = file_name
    height = int(labels[0].image_height)
    width = int(labels[0].image_width)
    key = hashlib.sha256(raw_img).hexdigest()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for label in labels:
        if label.classification_name !="NULL":
            ymin.append(label.y_min)
            xmin.append(label.x_min)
            ymax.append(label.y_max)
            xmax.append(label.x_max)
            tag_name = label.classification_name
            classes_text.append(tag_name.encode('utf8'))
            classes.append(tag_map[tag_name])
            truncated.append(0)
            poses.append("Unspecified".encode('utf8'))
            difficult_obj.append(0)
    
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature([height]),
      'image/width': int64_feature([width]),
      'image/filename': bytes_feature([
          filename.encode('utf8')]),
      'image/source_id': bytes_feature([
          filename.encode('utf8')]),
      'image/key/sha256': bytes_feature([key.encode('utf8')]),
      'image/encoded': bytes_feature([raw_img]),
      'image/format': bytes_feature(['jpeg'.encode('utf8')]),
      'image/object/bbox/xmin': float_feature(xmin),
      'image/object/bbox/xmax': float_feature(xmax),
      'image/object/bbox/ymin': float_feature(ymin),
      'image/object/bbox/ymax': float_feature(ymax),
      'image/object/class/text': bytes_feature(classes_text),
      'image/object/class/label': int64_feature(classes),
      'image/object/difficult': int64_feature(difficult_obj),
      'image/object/truncated': int64_feature(truncated),
      'image/object/view': bytes_feature(poses),
    }))
    return example

def create_tf_record(list_of_image_labels, record_file, image_loc, user_folders, split_names=["train","val"],
        split_percent=[.7,.3], tag_names = ["stamp"], test_file=None):
    

    #all_preds = list_of_image_labels

    all_files = defaultdict(list)
    '''
    if test_file is not None:
        with open(test_file, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            all_test = set((row[0] for row in reader))
        for row in all_preds:
            if row[0] not in all_test:
                all_files[row[0]].append(row)
    else:
    '''
    for label in list_of_image_labels:
        all_files[extract_image_name(label.image_location)].append(label)

    rand_list = list(all_files)
    np.random.shuffle(rand_list)
    split_percent = np.cumsum(split_percent)
    split_percent = split_percent[:-1]
    split_percent *= len(rand_list)
    split_percent = split_percent.round().astype(np.int)
    split_preds = np.split(rand_list,split_percent)

    
    tag_map = {name: index for index, name in enumerate(tag_names, 1)}
    record_file = Path(record_file)

    for name, filenames in zip(split_names, split_preds):
        with_suffix = record_file.with_suffix('')
        tf_path = "{}_{}".format(with_suffix, name) + record_file.suffix
        writer = tf.python_io.TFRecordWriter(tf_path)
        for filename in filenames:
            predictions = all_files[filename]
            # if user_folders:
            #     file_loc = str(Path(image_loc)/predictions[0][FOLDER_LOCATION]/filename)
            # else:
            file_loc = str(Path(image_loc)/filename)
            with open(file_loc, "rb") as img_file:
                raw_img = img_file.read()
            tf_example = create_tf_example(filename, predictions, raw_img, tag_map)
            writer.write(tf_example.SerializeToString())

        writer.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the config.ini file')
    args = parser.parse_args()
 
    from configparser import ConfigParser, ExtendedInterpolation
    config_file = ConfigParser(interpolation=ExtendedInterpolation())
    #config_file.read('/Users/andrebriggs/Desktop/MyConfig.ini')
    config_file.read(args.config_path)
    

    db_config = DatabaseInfo("abrig-db.postgres.database.azure.com","uranus","abrigtest@abrig-db","abcdABCD123")
    data_access = ImageTagDataAccess(PostGresProvider(db_config))
    #user_id = data_access.create_user(getpass.getuser())

    #Get latest labels and write to config_file["tagged_output"]  
    labels_file_path = config_file['Calculated']["tagged_output"]
    tf_record_path = config_file['Calculated']["tf_record_location"]
    local_images_path = config_file['Calculated']["image_dir"]
    classification_names = ("defect","knot","cat")#TestClassifications#config_file["classes"].split(",")
    #We should have 
      
    #list_of_image_labels = generate_test_labels(generate_test_image_tags([1,2,3,4,5],4,4))
    list_of_image_labels = data_access.get_labels()

    create_tf_record(list_of_image_labels,tf_record_path,local_images_path, False, tag_names=classification_names)


