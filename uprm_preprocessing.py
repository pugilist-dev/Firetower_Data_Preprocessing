# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 08:34:34 2021

@author: ritwfc2
"""

from osgeo import gdal
import numpy as np
import os
from PIL import Image
import argparse
import json
import laspy as lp
import open3d as o3d
import pandas as pd
import utm
from sklearn.neighbors import NearestNeighbors
import random

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="/Missions/1930/makoG419/processed/", help="3 band mako sensor data")
parser.add_argument("--dest", type=str, default="/data/interim/images/", help="path to store .png image")
parser.add_argument("--pc_path", type=str, default="/data/raw/point_cloud/mako_1930_FullRes_reprocessed_2021_group1_densified_point_cloud.laz", help="Point cloud path")
parser.add_argument("--xyz_path", type=str, default="/data/raw/point_cloud/mako_1930_FullRes_reprocessed_2021_group1_densified_point_cloud.xyz", help="Point Cloud lat, long data file")
parser.add_argument("--pc_dest", type=str, default="/data/interim/point_cloud/", help = "sampled point cloud destination")
opt = parser.parse_args()
print(opt)

#### Dictionary class to handle the image and geo data ####
class my_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value
        
#### multi band sensor image to PIL Image ####
def read_mako_img(img_path):
    data = gdal.Open(img_path, gdal.GA_ReadOnly) 
    band_1, band_2, band_3 = data.GetRasterBand(1), data.GetRasterBand(2), data.GetRasterBand(3)
    arr_1 = np.expand_dims(band_1.ReadAsArray(), axis=2)
    arr_2 = np.expand_dims(band_2.ReadAsArray(), axis=2)
    arr_3 = np.expand_dims(band_3.ReadAsArray(), axis=2)
    arr = np.concatenate((arr_1,arr_2, arr_3), axis = 2)
    img = Image.fromarray(arr)
    return img

#### saves the image in the intermediate location ####
def change_image_format(imglist, opt, mode):
    for names in imglist:
        img_path = opt.src+names
        img = read_mako_img(img_path)
        img.save(opt.dest+mode+"/"+names[0:-3]+ "png")
        

#### Function to get the center geo location #### 
def get_description_lat_long(hdrlist, opt):
    data_dict = my_dictionary()
    for hdr in hdrlist:
        
        file1 = open(opt.src+hdr, 'r')
        Lines = file1.readlines()
        img_name = ""
        for line in Lines:
            line = line.strip().split("=")
            if line[0].strip() == "description":
                img_name = line[1].strip()[1:-1]
            elif line[0].strip() == "geo points":
                geo_data = line[1].strip()[1:-1].split(",")
                lat = float(geo_data[2].strip())
                long = float(geo_data[3].strip())
                geo_loc = {"lat": lat,
                           "long": long}
                data_dict.add(img_name, geo_loc)
                
#### function to get the point cloud
def get_point_cloud(pc_path):
    point_cloud = lp.file.File(pc_path, mode="r")
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def lat_long(utm_frame):
    n = list(utm_frame.values)
    lat = []
    long = []
    for elm in n:
        utm_data = elm[0].split(',')
        x = float(utm_data[0])
        y = float(utm_data[1])
        coord = utm.to_latlon(x, y, 18, 'N')
        lat.append(coord[0])
        long.append(coord[1])
    return lat, long

def get_lat_long(file_path):
    xyz_point_cloud = pd.read_table(file_path)
    lat, long = lat_long(xyz_point_cloud)
    return lat, long

def get_nn_sampler(df):
    nbrs = NearestNeighbors(n_neighbors=2097152, algorithm='ball_tree').fit(df[['lat', 'long']])
    return nbrs

def process_img_pc(hdrlist, imglist, opt, mode):
    change_image_format(imglist, opt, mode)
    data_dict = get_description_lat_long(hdrlist, opt)
    out_file = open(opt.dest+mode+"/"+"images.json", "w")
    json.dump(data_dict, out_file, indent = 4) 
    pcd = get_point_cloud(opt.pc_path)
    lat,  long = get_lat_long(opt.xyz_path)
    data_points = np.asarray(pcd.points)[0:-1,:]
    df = pd.DataFrame(data_points, columns=['x','y','z'])
    df['lat'], df['long'] = lat, long
    nbrs = get_nn_sampler(df)
    for i in imglist:
       la = data_dict[i]['lat']
       lo = data_dict[i]['long']
       dat = np.asarray([la,lo])
       dat = dat.reshape(1,-1)
       distances, indices = nbrs.kneighbors(dat)
       interim_pc = df.iloc[list(indices[0])]
       points = np.vstack((interim_pc.x, interim_pc.y, interim_pc.z)).transpose()
       pc = o3d.geometry.PointCloud()
       pc.points = o3d.utility.Vector3dVector(points)
       o3d.io.write_point_cloud(opt.pc_dest +mode+"/"+ i[:-3] + "ply", pc)

if __name__=="__main__":
    root_path = os.listdir(opt.src)
    np.random.seed(0)
    imglist = []
    hdrlist = []
    for names in root_path:
        if names.endswith(".img"):
            imglist.append(names)
    
    imglist = np.asarray(sorted(imglist))
    data_size = np.arange(len(imglist))
    np.random.shuffle(data_size)
    val_length = int(0.2*len(imglist))
    train_length = len(imglist) - val_length
    for names in root_path:
        if names.endswith(".hdr"):
            hdrlist.append(names)
    hdrlist = np.asarray(sorted(hdrlist))
    val_img = imglist[data_size[0:val_length]]
    train_img = imglist[data_size[val_length:]]
    val_hdr = hdrlist[data_size[0:val_length]]
    train_hdr = hdrlist[data_size[val_length:]]
    
    process_img_pc(train_hdr, train_img, opt, mode="train")