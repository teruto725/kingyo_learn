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

import kingyo_v2 as K


cap = cv2.VideoCapture(1) # 0はカメラのデバイス番号
frame_no = 0
while True:
    ret, frame = cap.read()#frame読み込み
    output_frame = K.learnFrame(frame,frame_no)
    #####キーボード入力#######################
    k = cv2.waitKey(1) # 1msec待つ
    if k == 13:#エンターキー
        print("名前付け")
        print("Enter new Name:")
        name = input()
        K.nameNewKingyo(name,frame_no)
    elif k == 32:#スペースキー
        print("名前の更新")
        print("Enter new Name:")
        name = input()
        K.renameKingyo(name,frame_no)
    elif k == 27: # ESCキーで終了
        break

    ####描画処理#########################
    cv2.imshow('camera capture', output_frame)#表示


    frame_no += 1
cap.release()
cv2.destroyAllWindows()
