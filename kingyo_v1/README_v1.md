# 金魚個体識別プログラム
## 概要
tracking,CNNを用いて、水槽内に設置されたwebカメラから水槽内の金魚の個体識別を行うプログラムを作成する。  
本プログラムは最終的にサーバプログラムから呼び出されることを想定しているが現在は暫定的にキーボード操作で各動作を行っている。

## デモ
[https://www.youtube.com/watch?v=42_qHQsF7MA](https://www.youtube.com/watch?v=42_qHQsF7MA)

## 要求機能
- 金魚に対してユーザが名前を付けることができる  

- webカメラの映像を受け取り、映像内の金魚に対して、名前情報を付与した動画を出力する。  

## フォルダ構成
- kingyo.py ソースコード
- StatementDiagram.drawio 状態遷移図（drawioで書いた）
- opencv ~~~~ opencvのファイル（うまくインストールできなかった）

## キー入力
- エンターキー：addingmode 切り替え
- gキー : 金魚に命名を行う。コンソール上に名前を入力する。画面に一匹だけいる状態で行うこと。


## 制約
- 複数同時に新しい金魚を登録することはできない

- 名前付けは1匹ずつで行い必ず異なる金魚に対して行う。（金魚の数と、本プログラム内の金魚の数がずれたら終わり）

- 金魚追加モードがoffの時認識されていない金魚は存在してはならない

- 金魚は重なってはいけない



## クラス
クラスとそのクラスが持つ主なメンバ変数、メンバ関数について述べる。

## 1. Objectクラス

### - 概要
UnknownObjectとNamedObject用の抽象クラス

### - 属性

### - ふるまい
- tracking(frame,frame_no) : bool  
フレーム画像とフレーム番号を受け取ってtrackerを更新、トラッキングできたかどうかを返す。抽象メソッド。

- setTracker(tracker) : void   
trackerをセットする

- rmTracker() : void  
trackerを削除する  

- checkLatestFrame(rect) : bool  
矩形座標を受け取り、最新の矩形座標と一致するかどうかを返す。


## 2. NamedObjectクラス

### - 概要
ユーザによって名前が付けられたオブジェクト

### - 属性
- tracker : cv2.tracker  
トラッカー

- id : int  
CNNの目標値となるID、クラス変数を用いて連番で定義

- name : str
  ユーザによって入力される金魚の名前

- imagelist : list
trackingによって収集された過去の画像群。cnnを作成するときに使う。

- imagelist_temp : list
trackingによって収集された出現期間中の画像群。cnnを更新するときに使う。

- rectlist : list  
trackingで収集した矩形領域のログ（現状配列の最後の参照しかしていない）


### - ふるまい

- init(name,imagelist) : void  
初期化

- tracking(frame): bool  
Objectクラスのオーバーライド


- saveImageTemp()：void  
imagelist_tempをimagelistに移す



## 2. UnknownObjectクラス

### - 概要

名前が付けられる前のobject。addingmode = "OFF"の時のみ生成される。

### - 属性

- frame_nolist : list  
画面に入ったときから出るときまでのframe番号

- rectlist : list  
矩形座標リスト。frame_nolistと同期

- imagelist: list  
矩形画像リスト。frame_nolistと同期

- tracker : cv2.tracker
トラッカー

### - ふるまい

- tracking(frame, frame_no) : Objectクラスのオーバーライド

- check(frame_no, point ) : frame_noのフレームのpoint座標に自信が該当しているかどうか
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkwNzkyMzA5MCwxMzU4MzYyMDQ4LDE2Nj
I5OTE0NDcsLTE5NTE3NDA5NzIsLTE2NjQwOTI1MzAsMTYyODEx
NDE4OCwtNTQ4NTQwNDQ0LC0xMzgyNTk4NjA2LC02ODQ4MjA1ND
AsLTEzMjU0ODc2NTgsLTg1MTQ1MTMxNSwtMTgwNTI2MDU2NCw3
MzA5OTgxMTZdfQ==
-->