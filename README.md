# 金魚の個体認識
## 概要
CNNによって金魚の個体認識を行うpythonモジュールを作成した。  
複数回名前を付けることで精度向上を行うことができる。
金魚以外でも動く。  
https://www.youtube.com/watch?v=OMtZ7q4T6pw&t=6s  
https://www.youtube.com/watch?v=4what_o0AHA

## バージョン
python 3.7.1  
chainer==6.5.0  
scikit-learn==0.20.2  
numpy==1.18.1  
opencv-contrib-python==4.1.2.30  

## ファイル構成
- kingyo_v2.py:個体認識モジュール
- tester_v2.py:とりあえずうごかすためのプログラム
- kingyo_cnn.net:cnnを保存しているファイル

## モジュール機能
- learnFrame(frame,frame_no):  
フレームとフレーム番号を投げると学習結果を書き込んだフレームを返す。
- nameNewKingyo(name,frame_no,point):  
名前と対象の金魚が出現したフレーム番号、座標を投げると対象の金魚に対して命名、学習を行う。  
結果をBoolean型で返す。
- renameKingyo(name,frame_no,point):  
名前と対象の金魚が出現したフレーム番号、座標を投げると対象の金魚に対して再度命名、再度学習を行う。  
結果をBoolean型で返す。

## 制約
- 金魚学習中(NowLoading文字が画面上に出ている間)、nameNewKingyo()、renameKingyo()を行うことはできない。グローバル変数learningで学習中か判断できる。


## デモ
tester_v2.pyを起動する。（カメラ番号と、座標(point)を調整すること)  
エンターキーで金魚に対して命名が、スペースキーで金魚に対して再度命名ができる。  
ただし、命名は指定した座標にいる金魚に対して行われる。
