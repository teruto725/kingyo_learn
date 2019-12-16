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

class NameLessObject():
    def __init__(self):
        self.frame_nolist = list()
        self.rectlist = list()
        self.imagelist = list()

    def addTracker(self,tracker):
        self.tracker = tracker

    def rmTracker(self):
        self.tracker = None

    def tracking(self,frame,frame_no):#フレーム画像とフレーム番号を受け取ってtrackerを更新、認識できたかどうかを返す
        self.frame_nolist.append(frame_no)
        success,rect = self.tracker.update(frame)
        print(rect)
        if success and rect[0] > 0  and rect[2] > 0 and frame.shape[0] > rect[0]+rect[2] and frame.shape[1]>rect[1]+rect[3] :
            rect = np.array(rect,dtype=np.int32)
            self.rectlist.append(rect)
            self.imagelist.append(frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
            return True
        return False

    def check(self,frame_no,point):#frame_no,point座標に自身が該当しているかどうか
        if frame_no in self.frame_nolist:
            rect = self.rectlist[self.frame_nolist.index(frame_no)]
            if rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]:
                return True
        return False

    def checkLatestFrame(self,irect):#rectを受け取ってそれに合致しているかどうか
        point = [irect[0]+(int(irect[2]/2)),irect[1]+(int(irect[3]/2))]
        rect = self.rectlist[-1]
        if rect[0]<point[0]<rect[0]+rect[2] and rect[1]<point[1]<rect[1]+rect[3]:
            return True
        return False

def getRectList(frame):
    f_frame = frame.astype(np.uint8)#int変換
    bulr_img =cv2.GaussianBlur(f_frame,(15,15), 0)#平滑化
    gray_img = cv2.cvtColor(bulr_img, cv2.COLOR_RGB2GRAY)#グレースケール
    threshold_value = 100
    ret, thresh_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)#二値化
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #fc_img,contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#version依存
    rect_list = list()#矩形領域の座標リスト
    for layer in range(5):#layer = 階層の深さ
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                if hierarchy[0][i][3] != layer:
                    continue
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)
                rect_list.append([x,y,w,h])
        if len(rect_list) != 0:#矩形領域を抽出できたら
            break
    else:#矩形領域を抽出できなかったら
        return []
    return rect_list

def drawFrame(now_objlist,now_nlobjlist,frame):
    for obj in now_objlist:
        rect = obj.rectlist[-1]
        cv2.putText(frame, str(obj.name), (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, 8)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 0), 2)
    for nlobj in now_nlobjlist:
        rect = nlobj.rectlist[-1]
        cv2.putText(frame, str("NameLess"), (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, 8)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 0), 2)
    return frame

def drawFrameRect(rect_list,frame):
    for rect in rect_list:
        cv2.rectangle(frame, (rect[0],rect[1]),(rect[0] + rect[2], rect[1] + rect[3]),(100,0,0),2)

#画面上すべてのnloに対してtrackingを更新し金魚消失していないか調べる
def renewTrackingNLObj(frame,frame_no,now_nlobjlist,past_nlobjlist,rect_list):
    for nlobj in now_nlobjlist:#画面内に存在しているnlobj
        success = nlobj.tracking(frame,frame_no)#
        if success == False:#金魚消滅してたら
            print("金魚消失")
            nlobj.rmTracker()
            now_nlobjlist.remove(nlobj)
            past_nlobjlist.append(nlobj)

def appearNLObj(frame,frame_no,now_nlobjlist,rect_list):
    for rect in rect_list:
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(rect))
        nlo = NameLessObject()
        nlo.addTracker(tracker)
        nlo.tracking(frame,frame_no)
        now_nlobjlist.append(nlo)

if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # 0はカメラのデバイス番号
    adding_mode="OFF"
    last_num = 0#前回フレームの金魚数
    frame_no = 0
    now_objlist = list()#画面内のobj
    past_objlist = list()#画面外のobj
    now_nlobjlist = list()#画面内のnlobj
    past_nlobjlist = list()#画面外のnl

    while True:
        ret, frame = cap.read()#frame読み込み
        rect_list = getRectList(frame)#画像内の矩形領域リスト
        ######NLObj################
        if adding_mode == "ON":
            renewTrackingNLObj(frame,frame_no,now_nlobjlist,past_nlobjlist,rect_list)#tracker更新＋金魚消失判定
            rect_list_copy = copy.deepcopy(rect_list)
            for nlobj in now_nlobjlist:#すべてのnow_nlobjlistで一致するrectがあるか
                for rect in rect_list_copy:#rect_listからtrackerで追跡済みの物を削除
                    if nlobj.checkLatestFrame(rect) == True:#一致するrectがあった
                        rect_list.remove(rect)
                        break
                else:
                    print("TrackerERROR")
            if len(now_nlobjlist) < len(rect_list):#認識されていない金魚がいるとき
                print("金魚出現")
                appearNLObj(frame,frame_no,now_nlobjlist,rect_list)#NLObj生成



        ####描画処理#########################
        drawFrame(now_objlist,now_nlobjlist,frame)#認識結果描画
        drawFrameRect(rect_list,frame)
        cv2.imshow('camera capture', frame)#表示
        k = cv2.waitKey(1) # 1msec待つ
        if k == 13: #enterキーで追加モード
            adding_mode = "ON"
            print("adding_mode=ON")




        if k == 27: # ESCキーで終了
            break

        frame_no += 1
    cap.release()
    cv2.destroyAllWindows()
