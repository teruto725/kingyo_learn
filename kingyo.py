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

class Object(metaclass = ABCMeta):#抽象クラス

    #抽象メソッド
    @abstractmethod
    def tracking(self,frame,frame_no):
        pass

    #共通メソッド
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

    def setTracker(self,tracker):#trackerをせっていする
        self.tracker = tracker

    def rmTracker(self):#trackerを削除する
        self.tracker = None


class UnknownObject(Object):
    def __init__(self):
        self.frame_nolist = list()
        self.rectlist = list()
        self.imagelist = list()

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

class NamedObject(Object):
    nobj_con = 0#nobjのidカウント用クラス変数
    def __init__(self,name,imagelist):
        self.name = name
        self.imagelist= list()
        self.imagelist = imagelist
        self.id = NamedObject.nobj_con
        NamedObject.nobj_con += 1
        self.imagelist_temp = list()
        self.rectlist = list()

    def tracking(self,frame):#フレーム画像とフレーム番号を受け取ってtrackerを更新、認識できたかどうかを返す
        success,rect = self.tracker.update(frame)
        if success:
            rect = np.array(rect,dtype=np.int32)
            image = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            ok,ajimage = self.adjustRectImage(image)
            if ok:
                self.imagelist_temp.append(ajimage)
                self.rectlist.append(rect)
            return True
        return False

    def saveImageTemp(self):
        self.imagelist.extend(self.imagelist_temp)
        self.imagelist_temp.clear()

    def getImageTemp(self):
        return self.imagelist_temp
    def getImagelist(self):
        return self.imagelist
    def getId(self):
        return self.id
    def getName(self):
        return self.name


#CNN
class CNN(Chain):#出力数を受け取ってcnnを作成する
    def __init__(self,output_num):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, 20, 5), # filter 5
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
                if w >= 28 and h >= 28:#十分に多きサイズであれば
                    rect_list.append([x,y,w,h])
        if len(rect_list) != 0:#矩形領域を抽出できたら
            break
    else:#矩形領域を抽出できなかったら
        return []
    return rect_list


#frameに認識結果を書き込みframeを返す
def drawFrame(now_nobjlist,now_uobjlist,frame):
    for nobj in now_nobjlist:
        rect = nobj.rectlist[-1]
        cv2.putText(frame, str(nobj.name), (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, 8)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 0), 2)
    for uobj in now_uobjlist:
        rect = uobj.rectlist[-1]
        cv2.putText(frame, str("Unknown"), (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2, 8)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 0), 2)
    return frame

#新しくできた領域に対しtrackerを設定し、新たなNlnobjを生成
def appearUObj(frame,frame_no,now_uobjlist,rect_list):
    for rect in rect_list:
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(rect))
        uobj = UnknownObject()
        uobj.setTracker(tracker)
        uobj.tracking(frame,frame_no)
        now_uobjlist.append(uobj)

#nloが画面上から外に出た時の処理
def disappearUObj(uobj,now_uobjlist,past_uobjlist):
    print("金魚消失")
    uobj.rmTracker()
    now_uobjlist.remove(uobj)
    past_uobjlist.append(uobj)

def disappearNObj(nobj,now_nobjlist,past_nobjlist):
    print(nobj.getName()+"消失")
    nobj.rmTracker()
    now_nobjlist.remove(nobj)
    past_nobjlist.append(nobj)

#frame_noとprintから対象のNLOを特定し、nameとともにnobj生成,past_uobjはpast_opjにnowはnowに分類される
def naming(frame_no,now_nobjlist,now_uobjlist,past_nobjlist,past_uobjlist,rect_list):
    print("Naming")
    print("Enter new Name:")
    name = input()

    #print("Enter Point:")
    #point =list(map(int,input().split()))	#point = [x,y]
    point = [rect_list[0][0]+10,rect_list[0][1]+10]#テスト用に座標を固定よってrect_listは引数に不必要
    all_uobjlist = list()
    all_uobjlist.extend(now_uobjlist)
    all_uobjlist.extend(past_uobjlist)
    print(len(all_uobjlist))
    for uobj in all_uobjlist:
        if uobj.check(frame_no-1,point):#frame-1はよくない
            new_nobj = NamedObject(name,uobj.imagelist)
            if uobj in now_uobjlist:#new_nobjが現在画面内か、画面外かでappedn先が変わる
                new_nobj.setTracker(uobj.tracker)
                now_nobjlist.append(new_nobj)
            else:
                past_nobjlist.append(new_nobj)
            break
    else:
        print("座標がずれています。もう一度指定しなおしてください")

#初回の学習
def createCNN(cnn,optimizer,all_nobjlist):
    batch_size = 1#バッチサイズ
    n_epoch = 2#エポック数

    x_data = np.zeros((1,28*28), np.float32)#入力値
    for nobj in all_nobjlist:
        x_data = np.append(x_data, nobj.getImagelist(), 0)#怪しい
    x_data = np.delete(x_data,0,0)
    x_data = x_data.reshape((len(x_data), 1, 28, 28))
    print("x_data"+str(np.shape(x_data)))

    t_data = np.zeros((1,1), np.int32)#目標値
    for nobj in all_nobjlist:
        for i in range(len(nobj.getImagelist())):
            t_data = np.append(t_data, nobj.getId())
    t_data = np.delete(t_data,0,0)
    print("t_data"+str(np.shape(t_data)))

    perm = np.random.permutation(len(x_data))

    print("CNN生成開始")
    for epoch in range(n_epoch):
        for i in range(0, len(x_data), batch_size):
            x = Variable(x_data[perm[i:i+batch_size]])
            t = Variable(t_data[perm[i:i+batch_size]])
            y = cnn.forward(x)
            cnn.zerograds()
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            loss.backward()
            optimizer.update()
        print(str(epoch+1)+"epoch目 完了:acc="+str(acc.data))
#CNNの更新
def updateCNN(cnn,optimizer,train_img,id):
    print("CNN更新開始")
    batch_size = 1
    n_epoch= 2
    x_data = np.array(train_img,np.float32)
    print(np.shape(x_data))
    x_data = x_data.reshape((len(x_data), 1, 28, 28))
    print(np.shape(x_data))
    t_data = np.full(len(train_img),id, np.int32)
    perm = np.random.permutation(len(x_data))
    for epoch in range(n_epoch):
        for i in range(0, len(x_data), batch_size):
            x = Variable(x_data[perm[i:i+batch_size]])
            t = Variable(t_data[perm[i:i+batch_size]])
            y = cnn.forward(x)
            cnn.zerograds()
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            loss.backward()
            optimizer.update()
        print(str(epoch+1)+"epoch目 完了:acc="+str(acc.data))


if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # 0はカメラのデバイス番号
    now_nobjlist = list()#画面内のnobj
    past_nobjlist = list()#画面外のnobj
    now_uobjlist = list()#画面内のuobj
    past_uobjlist = list()#画面外のnl
    adding_mode="OFF"
    last_num = 0#前回フレームの金魚数
    frame_no = 0

    cnn = None #CNN
    optimizer = None #optimizer

    while True:
        ret, frame = cap.read()#frame読み込み
        color_frame = copy.deepcopy(frame)
        rect_list = getRectList(frame)#画像内の矩形領域リスト
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#グレースケール　（後でRGBにすること）

        #####キーボード入力#######################
        k = cv2.waitKey(1) # 1msec待つ
        if k == 13: #enterキーで追加モード切り替え
            if adding_mode == "OFF":
                for nobj in now_nobjlist:#nownobjを全部pastnobjにする
                    past_nobjlist.append(nobj)
                now_nobjlist.clear()
                adding_mode = "ON"
                print("adding_mode=ON")
            elif adding_mode == "ON":
                past_uobjlist.clear()#uobjは初期化
                now_uobjlist.clear()
                adding_mode = "OFF"
                print("adding_mode=OFF")

        if k == ord("g") and adding_mode == "ON":#gキーで名前登録
            naming(frame_no,now_nobjlist,now_uobjlist,past_nobjlist,past_uobjlist,rect_list)#名前からnobject生成
            all_nobjlist = list()
            all_nobjlist.extend(now_nobjlist)
            all_nobjlist.extend(past_nobjlist)
            if len(all_nobjlist)>1:
                cnn = CNN(len(all_nobjlist))
                optimizer = chainer.optimizers.Adam()
                optimizer.setup(cnn)
                createCNN(cnn,optimizer,all_nobjlist)
            adding_mode = "OFF"
            now_uobjlist = list()#初期化
            past_uobjlist = list()#初期化
        if k == 27: # ESCキーで終了
            break

        ######UObj################
        if adding_mode == "ON":
            for uobj in now_uobjlist:
                success = uobj.tracking(frame,frame_no)#trackerを更新
                if success == False:#金魚消滅してたら
                    disappearUObj(uobj,now_uobjlist,past_uobjlist)
                else:
                    for rect in rect_list:#rect_listからtrackerで追跡済みの物を削除
                        if uobj.checkLatestFrame(rect) == True:#一致するrectがあったつまり認識可能
                            rect_list.remove(rect)
                            break
                    else:#一致するrectがない⇒画面端にいるときつまり認識不可能
                        disappearUObj(uobj,now_uobjlist,past_uobjlist)
            if 0 < len(rect_list):#認識されていない金魚がいるとき
                print("金魚出現")
                appearUObj(frame,frame_no,now_uobjlist,rect_list)#UObj生成


        ######NObj#################
        elif adding_mode == "OFF":
            for nobj in now_nobjlist:#すべてのtrackerを更新する
                success = nobj.tracking(frame)
                if success == False:#消失
                    disappearNObj(nobj,now_nobjlist,past_nobjlist)
                    updateCNN(cnn,optimizer,nobj.getImageTemp(),nobj.getId())
                    nobj.saveImageTemp()
                else:
                    for rect in rect_list:
                        if nobj.checkLatestFrame(rect) == True:#一致するrectがあったつまり認識可能
                            rect_list.remove(rect)
                            break
                    else:#一致するrectがない⇒画面端にいるときつまり認識不可能
                        disappearNObj(nobj,now_nobjlist,past_nobjlist)
                        if len(past_nobjlist)+len(now_nobjlist) > 1: #1個以上nobjがあったら
                            updateCNN(cnn,optimizer,nobj.getImageTemp(),nobj.getId())
                            nobj.saveImageTemp()

            if 0 < len(rect_list):#金魚が入ってきた
                print("金魚出現")
                if len(past_nobjlist)+len(now_nobjlist) == 0:#ojbがないときに金魚は入ってこない
                    yet = "二値化エラー"
                elif (len(past_nobjlist)+len(now_nobjlist)) == 1:#金魚が一匹しかおらん時（微妙）
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(rect_list[0]))#一匹なら一匹と決めつけしているので微妙
                    now_nobjlist.append(past_nobjlist[0])
                    past_nobjlist.remove(past_nobjlist[0])
                    nobj.setTracker(tracker)
                else:
                    for rect in rect_list:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, tuple(rect))
                        x_data = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                        x_data = cv2.resize(x_data, (28,28),0,0 ,cv2.INTER_NEAREST)
                        x_data = x_data.reshape(1,1, 28, 28)
                        x_data = np.array(x_data, np.float32)
                        #学習
                        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                            prolist = cnn(x_data)
                        print("prolist"+str(prolist.data))
                        max = -10000
                        maxproID = None
                        for nobj in past_nobjlist:#past_nobjlistの中から最も可能性の高いIDを見つける（deleteが入ったときに怪しい）
                            if max < prolist.data[0][nobj.getId()]:
                                max = prolist.data[0][nobj.getId()]
                                maxproID = nobj.getId()
                        print("PROID:"+str(maxproID))
                        past_nobjlist.remove(nobj)
                        now_nobjlist.append(nobj)
                        nobj.setTracker(tracker)


        ####描画処理#########################
        drawFrame(now_nobjlist,now_uobjlist,color_frame)#認識結果描画
        cv2.imshow('camera capture', color_frame)#表示


        frame_no += 1
    cap.release()
    cv2.destroyAllWindows()
