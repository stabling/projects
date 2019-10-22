import os

line = open(os.path.join("vocab.txt"),"r+",encoding="utf-8").read()
print(len(line.split()))