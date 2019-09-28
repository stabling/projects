# coding:utf-8
import SimpleITK as sitk
import cv2

def dcmtopng(filename,outpath,data):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    for img_item in img_array:
        cv2.imwrite("%s/%s.png" % (outpath,data.split('.')[0]), img_item)