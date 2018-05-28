import os
import shutil

parent = '/ai/sources/keras.ins.facenet'
input = '/lfw.original.from.internet'
output = '/lfw.more.than.eleven'
for dir,subdir,files in os.walk(parent + input):
	if len(files) > 11 :
		print dir + " has crossed limit, " + "total files: " + str(len(files))
		shutil.copytree(dir, dir.replace(input, output))
