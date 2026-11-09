# 出典

- 元のモデル: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) の `Simulation/SO101`（URDF・MJCF・STL メッシュ。Onshape の CAD から onshape-to-robot で生成されたもの）。Apache License 2.0
- `README.md` は元のリポジトリのものをそのまま置いています（このディレクトリにないファイルにも触れています）

## このリポジトリで変えたところ

- `so101_new_calib.xml`: 固定ジョーの衝突形状をメッシュの凸包から箱 3 つに変更（凸包が指とモーターの筐体をつないでしまい、ブロックに当たっていたため）
- `pick_place.xml`: 机・トレイ・ブロック・カメラを足したピック＆プレースのシーン（新規）
