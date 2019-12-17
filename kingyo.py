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
        if success:
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

class Object():
    obj_con = 0#objのidカウント用クラス変数
    def __init__(self,name,imagelist):
        self.name = name
        self.imagelist= list()
        self.imagelist = imagelist
        self.id = Object.obj_con
        Object.obj_con += 1
        self.rectlist = list()
        self.imagelist_temp = list()


    def addTracker(self,tracker):
        self.tracker = tracker

    def rmTracker(self,tracker):
        self.tracker = None

    def tracking(self,frame,frame_no):#フレーム画像とフレーム番号を受け取ってtrackerを更新、認識できたかどうかを返す
        self.frame_nolist.append(frame_no)
        success,rect = self.tracker.update(frame)
        if success:
            rect = np.array(rect,dtype=np.int32)
            self.rectlist.append(rect)
            self.imagelist_temp.append(frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
        return success

    def saveImageTemp(self):
        self.imagelist.append(self.imagelist_temp)
    def getImageTemp(self):
        return self.imagelist_temp
    def getImagelist(self):
        return self.imagelist
    def getId(self):
        return self.id
#CNN
class CNN(Chain):#出力数を受け取ってcnnを作成する
    def __init__(self,output_num):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(3, 20, 5), # filter 5
            conv2 = L.Convolution2D(20, 50, 5), # filter 5
            l1 = L.Linear(800, 500),
            l2 = L.Linear(500, 500),
            l3 = L.Linear(500, output_num, initialW=np.zeros((output_num, 500), dtype=np.float32))
        )
    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

#フレームを受け取り矩形領域リストを取り出す
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

#CNNように画像を調整
def adjustRectImage(imagelist):
    resize_imglist=list()
    for i in range(len(imglist)):
        resize_imglist.append(cv2.resize(imglist[i], (28,28),0,0 ,cv2.INTER_NEAREST))#28×28領域に調整
    flat_imglist=list()
    for i in range(len(resize_imglist)):
        flat_imglist.append(resize_imglist[i].flatten())#2次元配列から1次元に変換
    return flat_imglist

#frameに認識結果を書き込みframeを返す
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

#新しくできた領域に対しtrackerを設定し、新たなNlobjを生成
def appearNLObj(frame,frame_no,now_nlobjlist,rect_list):
    for rect in rect_list:
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(rect))
        nlo = NameLessObject()
        nlo.addTracker(tracker)
        nlo.tracking(frame,frame_no)
        now_nlobjlist.append(nlo)

#nloが画面上から外に出た時の処理
def disappearNLObj(nlobj,now_nlobjlist,past_nlobjlist):
    print("金魚消失")
    nlobj.rmTracker()
    now_nlobjlist.remove(nlobj)
    past_nlobjlist.append(nlobj)

#frame_noとprintから対象のNLOを特定し、nameとともにobj生成,past_nlobjはpast_opjにnowはnowに分類される
def naming(frame_no,now_objlist,now_nlobjlist,past_objlist,past_nlobjlist):
    print("Naming")
    print("Enter new Name:")
    name = input()
    #print("Enter Point:")
    #point =list(map(int,input().split()))	#point = [x,y]
    point = [rect_list[0][0]+10,rect_list[0][1]+10]#テスト用に座標を固定よってrect_listは引数に不必要
    all_nlobjlist = list()
    all_nlobjlist.extend(now_nlobjlist)
    all_nlobjlist.extend(past_nlobjlist)
    for nlobj in all_nlobjlist:
        if nlobj.check(frame_no,point):
            new_obj = Object(name,nlobj.imagelist)
            if nlobj in now_nlobjlist:#new_objが現在画面内か、画面外かでappedn先が変わる
                new_obj.addTracker(nlobj.tracker)
                now_objlist.append(new_obj)
            else:
                past_objlist.append(new_obj)
            break
    else:
        print("座標がずれています。もう一度指定しなおしてください")

def appearObj(now_objlist):
    print
#初回の学習
def createCNN(cnn,optimizer,all_objlist):
    batch_size = 500#バッチサイズ
    n_epoch = 2#エポック数
    x_data = np.zeros((1,28*28), np.float32)#入力値
    for obj in all_objlist:
        x_data = np.append(x_data, adjustRectImage(obj.getImagelist()), 0)#怪しい
    x_data = np.delete(x_data,0,0)
    x_data = x_data.reshape((len(x_data), 1, 28, 28))
    t_data = np.zeros((1,1), np.int32)#目標値
    for obj in all_objlist:
        t_data = np.append(t_data, all_objlist.getId())
    t_data = np.delete(t_data,0,0)
    perm = np.random.permutation(len(x_data))
    for epoch in range(n_epoch):
        for i in range(0, len(x_data), batch_size):
            x = Variable(x_data[perm[i:i+batch_size]])
            t = Variable(t_data[perm[i:i+batch_size]])
            y = model.forward(x)
            model.zerograds()
            #loss = F.softmax_cross_entropy(y, t)
            #acc = F.accuracy(y, t)
            loss.backward()
            optimizer.update()

def updateCNN(cnn,optimizer,train_img,id):
    batch_size = 500
    n_epoch= 2
    x_data = np.array(train_img)
    x_data = x_data.reshape((len(train_img), 1, 28, 28))
    t_data = numpy.full(len(train_img),id, np.int32)
    perm = np.random.permutation(len(x_data))
    for epoch in range(n_epoch):
        for i in range(0, len(x_data), batch_size):
            x = Variable(x_data[perm[i:i+batch_size]])
            t = Variable(t_data[perm[i:i+batch_size]])
            y = model.forward(x)
            model.zerograds()
            #loss = F.softmax_cross_entropy(y, t)
            #acc = F.accuracy(y, t)
            loss.backward()
            optimizer.update()



if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # 0はカメラのデバイス番号
    now_objlist = list()#画面内のobj
    past_objlist = list()#画面外のobj
    now_nlobjlist = list()#画面内のnlobj
    past_nlobjlist = list()#画面外のnl
    adding_mode="OFF"
    last_num = 0#前回フレームの金魚数
    frame_no = 0

    cnn = None #CNN
    optimizer = None #optimizer

    while True:
        ret, frame = cap.read()#frame読み込み
        rect_list = getRectList(frame)#画像内の矩形領域リスト

        ######NLObj################
        if adding_mode == "ON":
            for nlobj in now_nlobjlist:
                success = nlobj.tracking(frame,frame_no)#trackerを更新
                if success == False:#金魚消滅してたら
                    disappearNLObj(nlobj,now_nlobjlist,past_nlobjlist)
                else:
                    for rect in rect_list:#rect_listからtrackerで追跡済みの物を削除
                        if nlobj.checkLatestFrame(rect) == True:#一致するrectがあったつまり認識可能
                            rect_list.remove(rect)
                            break
                    else:#一致するrectがない⇒画面端にいるときつまり認識不可能
                        disappearNLObj(nlobj,now_nlobjlist,past_nlobjlist)
            if 0 < len(rect_list):#認識されていない金魚がいるとき
                print("金魚出現")
                appearNLObj(frame,frame_no,now_nlobjlist,rect_list)#NLObj生成


        ######Obj#################
        elif adding_mode == "OFF":
            #renewTrackingObj()
            success_frag = False
            for obj in now_objlist:#すべてのtrackerを更新する
                success = obj.tracking(frame, frame_no)
                if success == False:#消失
                    print(name+":消滅")
                    obj.rmTracker()
                    now_objlist.remove(nlobj)
                    past_objlist.append(nlobj)
                    success_frag=True
                    updateCNN(cnn,optimizer,obj.getImageTemp(),obj.getId())

            if len(now_objlist) < len(rect_list):#金魚が入ってきた
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, tuple(rect_list.pop(0)))#0でいいの？

        ####描画処理#########################
        drawFrame(now_objlist,now_nlobjlist,frame)#認識結果描画
        cv2.imshow('camera capture', frame)#表示
        k = cv2.waitKey(1) # 1msec待つ
        if k == 13: #enterキーで追加モード
            adding_mode = "ON"
            print("adding_mode=ON")

        if k == ord("g"):#gキーで名前登録
            naming(frame_no,now_objlist,now_nlobjlist,past_objlist,past_nlobjlist)#名前からobject生成
            all_objlist.extend(now_objlist)
            all_objlist.extend(past_objlist)
            cnn = CNN(len(all_objlist))
            optimizer = chainer.optimizers.Adam()
            optimizer.setup(model)
            createCNN(cnn,optimizer,all_objlist)
            adding_mode = "OFF"


        if k == 27: # ESCキーで終了
            break

        frame_no += 1
    cap.release()
    cv2.destroyAllWindows()
