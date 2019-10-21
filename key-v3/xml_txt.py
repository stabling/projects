import os
import xml.etree.ElementTree as ET
import glob


def xml_to_txt(indir, outdir):
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')

    for i, file in enumerate(annotations):

        file_save = file.split('.')[0] + '.txt'
        file_txt = os.path.join(outdir, file_save)
        f_w = open(file_txt, 'w')

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('item'):
            print(obj)
            # current = list()
            name = obj.find('name').text

            xmlbox = obj.find('bndbox')
            xn = xmlbox.find('xmin').text
            xx = xmlbox.find('xmax').text
            yn = xmlbox.find('ymin').text
            yx = xmlbox.find('ymax').text
            w = int(xx) - int(xn)
            h = int(yx) - int(yn)
            x1 = int(xn) + int(0.5 * w)
            y1 = int(yn) + int(0.5 * h)
            # print xn
            f_w.write("{}.jpg".format(i) + ' ')
            f_w.write(str(x1) + ' ' + str(y1) + ' ' + str(w) + ' ' + str(h) + ' ')
            f_w.write(name + '\n')
            i = i + 1


indir = '/home/yzs/datasets/outputs'  # xml目录
outdir = '/home/yzs/myproject/yolo-v3/Anno'  # txt目录

xml_to_txt(indir, outdir)
