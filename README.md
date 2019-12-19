# 金魚個体識別プログラム


#  入出力
1. adding_mode():金魚追加モードを移行する
2. adding_fish(frame_no,name,rect):金魚を追加する
3. delete_fish(fish_id):対象の金魚情報を削除する
4. get_latest_frame():最新フレームを取り出す



# 状態遷移

## Tracking状態(adding_mode="NO")

- trackingを行い追跡物の画像を対象のObjectクラスにaddImageTemp()する。

- 画像データにtrackingによって求められた矩形領域並びに、対応しているobject名を書き込み、ストリーミングを行う。



## Tracking状態(adding_mode="YES")

- trackingを行い追跡物の画像,フレーム番号、矩形座標を対象のNameLessObjectクラスにaddInfo()する。



## 新規金魚追加タイミング

- adding_modeフラグを"YES"にする
##  金魚名付けタイミング(adding_mode="YES")

- 受け取った座標位置、flame番号が合致するNameLessObjectListを探索し、合致するNamelessObjectと受け取ったNameから新しいObjectクラスを生成する

- adding_mode = "DONE"

- 強制的に認識物消失タイミングに遷移



## 認識物出現タイミング (adding_mode="NO")

- CNNに出現物のデータを投げ、判別されたIDから出現したObjectを特定する。

- trackerを出現物を対象として生成し、そのtrackerと特定したObjectを紐づける。

- trackerによって切り取られた矩形画像を対応したObjectにaddImageTemp()する



## 認識物出現タイミング (adding_mode="YES")

- 新しいNameLessObjectを生成しそれとTrackerを紐づける




## 認識物消失タイミング (adding_mode="YES")

- 消失したNameLessObjectのTrackerを削除する



## 認識物消失タイミング（adding_mode="DONE")

- すべてのObjectに対してsaveImageTemp()する

- 現在のCNNを削除し、新しくCNNをObject数に合わせて作成する

- すべてのObjectに対してgetLearnAll()を行い取得したデータでCNNを学習させる

- adding_mode="NO"



## 認識物消失タイミング(adding_mode="NO")

- 全オブジェクトに対してgetLearnTemp()を行い取得したデータでCNNを学習させる。

- 全オブジェクトに対してsaveImageTemp()を行うことで学習した画像をimagelistに移す



# 制約

- 初期状態では金魚は水槽内にいないものとする

- 複数同時に新しい金魚を登録することはできない

- 名前付けは1匹ずつで行い必ず異なる金魚に対して行う。（金魚の数と、本プログラム内の金魚の数がずれたら終わり）

- 認識されていない金魚は金魚追加モードがoffの時は0匹,onの時は1匹となっていなければならない



# クラス
クラスとそのクラスが持つ主なメンバ変数、メンバ関数について述べる。

## 1. Objectクラス

### - 概要
UnknownObjectとNamedObjectの抽象クラス

### - 属性

### - ふるまい
- tracking(frame,frame_no) : bool   
フレーム画像とフレーム番号を受け取ってtrackerを更新、トラッキンできたかどうかを返す。抽象メソッド。
- setTracker(tracker) : void  
trackerをセットする
- rmTracker() : void
trackerを削除する
- checkLatestFrame(rect) : bool
矩形座標を受け取り、最新の矩形座標と一致するかどうかを返す。



## 2. Objectクラス

### - 概要
UnknownObjectとNamedObjectの抽象クラス

### - 属性
- tracker : トラッカー

- id : CNNの目標値となるID

- name : ユーザによって入力される正解

- imagelist : trackingによって収集された過去の画像群

- imagelist_temp : trackingによって収集された出現期間中の画像群



### - ふるまい

- init(NameLessObject,name) : NameLessObjectとnameを受け取ってname,imagelist,を更新する。
- getLearnTemp() : ImageTempとIDを学習用に成形して渡す
- getLearnAll()：Imagelistを学習用に成形して渡す

- addImageTemp(image) :imagelist_tempに画像を追加する

- saveImageTemp()：imagelist_tempをimagelistに移す



## 2. NameLessObjectクラス

### - 概要

認識が完了していないオブジェクト（各リストはindexを共通させておく）

### - 属性

- frame_nolist : 画面に入ったときから出るときまでのframe番号

- rectlist : 矩形座標リスト

- imagelist: 矩形画像リスト

- tracker : トラッカー

### - ふるまい

- addInfo(frame_no, rect, image) : それぞれのパラメータを追加する

- addTracker(tracker) : トラッカーを追加する

- tracking(frame, frame_no) :
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU0ODU0MDQ0NCwtMTM4MjU5ODYwNiwtNj
g0ODIwNTQwLC0xMzI1NDg3NjU4LC04NTE0NTEzMTUsLTE4MDUy
NjA1NjQsNzMwOTk4MTE2XX0=
-->