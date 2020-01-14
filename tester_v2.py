import kingyo_v2 as K
import cv2
#kingyo_v2のテスト用コード


cap = cv2.VideoCapture(1) # カメラのデバイス番号
frame_no = 0
while True:

    ret, frame = cap.read()#frame読み込み
    output_frame = K.learnFrame(frame,frame_no)#フレームとフレームナンバーを投げると学習したフレームが返ってくる
    #####キーボード入力#######################
    k = cv2.waitKey(1) # 1msec待つ
    if k == 13:#エンターキー
        print("名前付け")
        print("Enter new Name:")
        name = input()
        point = [30,120]#画面右上らへんの座標
        K.nameNewKingyo(name,frame_no,point)
    elif k == 32:#スペースキー
        print("名前の更新")
        print("Enter new Name:")
        name = input()
        point = [30,120]#画面右上らへんの座標
        K.renameKingyo(name,frame_no,point)

    elif k == 27: # ESCキーで終了
        break

    ####描画処理#########################
    cv2.imshow('camera capture', output_frame)#表示
    frame_no += 1
cap.release()
cv2.destroyAllWindows()
