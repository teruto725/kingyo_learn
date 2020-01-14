import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from chainer import Sequential
import chainer.functions as F
import chainer.links as L
import chainer
from pathlib import Path
from chainer import Chain, optimizers, Variable, serializers
import copy
from abc import ABCMeta
from abc import abstractmethod


class UnknownObject():
    def __init__(self):
        self.frame_nolist = list()
        self.rectlist = list()
        self.imagelist = list()
        self.name = "Unkonwn"

    def setName(self,name):
        self.name = name

    def getName(self):
        return self.name

    def tracking(self,frame,frame_no):#フレーム画像とフレーム番号を受け取ってtrackerを更新、認識できたかどうかを返す
        success,rect = self.tracker.update(frame)
        if success:
            rect = np.array(rect,dtype=np.int32)
            image = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            ok, ajimage = self.adjustRectImage(image)
            if ok:#リサイズできたなら
                self.rectlist.append(rect)
                self.imagelist.append(ajimage)
                self.frame_nolist.append(frame_no)
            return True#trueはトラッキング結果なのでそのまま返す
        return False

    def check(self,frame_no,point):#frame_no,point座標に自身が該当しているかどうか
        if frame_no in self.frame_nolist:
            rect = self.rectlist[self.frame_nolist.index(frame_no)]
            if rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]:
                return True
        return False

    def setTracker(self,tracker):#trackerをせっていする
        self.tracker = tracker

    def rmTracker(self):#trackerを削除する
        self.tracker = None

    def adjustRectImage(self,image):#画像サイズを変更する
        try:
            image = cv2.resize(image, (28,28),0,0 ,cv2.INTER_NEAREST)#28×28領域に調整
            return True, image.flatten()
        except:
            return False, None

    def checkLatestFrame(self,irect):#rectを受け取ってそれに合致しているかどうか
        point = [irect[0]+(int(irect[2]/2)),irect[1]+(int(irect[3]/2))]
        rect = self.rectlist[-1]
        if rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]:
            return True
        return False


class NamedObject():
    nobj_con = 0#nobjのidカウント用クラス変数
    def __init__(self,name,imagelist):
        self.name = name
        self.imagelist= list()
        self.imagelist = imagelist
        self.id = NamedObject.nobj_con
        NamedObject.nobj_con += 1
        #self.rectlist = list()

    def getImagelist(self):
        return self.imagelist
    def getId(self):
        return self.id
    def getName(self):
        return self.name
    def addImagelist(self,imglist):
        self.imagelist.extend(imglist)


#CNN
class CNN(Chain):#出力数を受け取ってcnnを作成する
    def __init__(self,output_num):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, 20, 5), # filter 5
            conv2 = L.Convolution2D(20, 50, 5), # filter 5
            l1 = L.Linear(800, 300),
            l2 = L.Linear(300, 300),
            l3 = L.Linear(300, output_num, initialW=np.zeros((output_num, 300), dtype=np.float32))
        )
    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h
