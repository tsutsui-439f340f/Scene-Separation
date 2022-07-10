# scene_separation
## 概要
このプログラムは動画内の各シーンを分離する際に、利用できるプログラムになっています。
なおシステム概要として、まず動画の各フレームをVGG16を使って特徴量に変換します。\
そして現在のフレームの特徴量と過去のフレームの特長量の類似度を測定し、閾値判定で現在のフレームが過去のフレームのシーンの延長なのかを決定します。
また各シーンのキーフレームを取得後、細かく分割してしまっているシーンに関しては再度、特長量照合を行い、できるだけ大きな塊に分割するようにしています。\
この後処理によって、分離しすぎてしまうミスを減らしています。\
しかし類似部分が多いシーン間ではそれらの結合を誘発する可能性があります。\
また、現状、変化が激しいシーン(輝度が瞬間的に変化するようなシーンなど)ではシーンの分割ミスが起きやすくなっています。
## 動作確認環境
windows10\
Python3.8.8
```
主なライブラリ
torchvision==0.10.0
opencv-python==4.6.0.66
scipy==1.6.2
```
## 使い方
``` python scene_separation.py "動画ファイル.mp4"```

## サンプル
この動画はBlenderを用いて作成したもので権利は作者にあります。

https://user-images.githubusercontent.com/55880071/178159094-64ba36de-eb38-4ce2-8e8c-1695786f9dd6.mp4


https://user-images.githubusercontent.com/55880071/178159234-6caa0e49-7584-42b0-a77d-6d41aa6a9ca0.mp4


https://user-images.githubusercontent.com/55880071/178159251-f709169e-24df-4051-b3e0-d47e5c2b25c2.mp4


https://user-images.githubusercontent.com/55880071/178159254-073b4364-c464-4117-aa05-c9b69bfbd7e3.mp4


https://user-images.githubusercontent.com/55880071/178159259-49534b0b-e964-49ba-a2df-7a0b4d920b46.mp4




## 備考


今後、このプログラムの改良として、より激しいシーンにも対応出来るニューラルシーン分離機を作成する予定です。


