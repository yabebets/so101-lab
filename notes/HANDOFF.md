# SO-ARM101 模倣学習 引き継ぎメモ

最終更新：2026-10-13 07:00 JST（作業を中断した時点）

再開するときは、このファイルを最初に読んでください。記事は `articles/so101-imitation-learning-act.md` です（作業の経緯と技術解説はこちらに詳しく書いてあります）。

---

## 1. 今どこまで進んでいるか

| ステップ | 状態 |
|---|---|
| テレオペ（リーダー → フォロワー） | ✅ 動作確認済み |
| フォロワーの再キャリブレーション | ✅ 2026-10-12 20:57 に実施（グリッパー過負荷の対策） |
| データ収録 | ✅ 50 回（最初の 30 回 + 今のカメラ位置で追加 20 回） |
| 学習（ACT、Colab A100） | ✅ 30 回版・50 回版とも完了 |
| 実機評価 | ✅ **50 回版で初成功**（2026-10-13 06:46、`eval_so101_block_place_50ep`） |
| **成功率の測定（10 回）** | ⏸ **1 回目のみ完了（失敗）。残り 9 回が未実施** ← ここから再開 |
| ブログ記事 | 📝 8-8「初成功」まで記述済み。成功率の結果を 8-9 として追記する |

---

## 2. 再開の手順（成功率測定の残り 9 回）

### 2-1. 事前準備（毎回）

1. **フォロワーの電源を入れ直す**（腕を支えてから）。グリッパーが過負荷のままだと ID 6 が応答せず、`Missing motor IDs: 6` で起動できない。電源アダプタ側を抜いて 10 秒待つ
2. **サーボ確認**：`.venv/bin/python tools/health.py`（10 秒読み取り＋電圧。エラー 0・12.3〜12.5V なら正常）
3. **iPhone の位置合わせ**：一晩で動いていたことがある。基準は **追加収録した 20 回のうち episode 40** の映像
   ```bash
   .venv/bin/python tools/align_live.py 4 /tmp/align_final.jpg 40
   ```
   ウィンドウに `OK!`（ずれ ±3%、大きさ ±5% 以内）が出るまで合わせて `q` で閉じる。前回は dx +3% / dy −2% / scale 0.98 で評価した
4. **リーダーとフォロワーを休止姿勢（折りたたみ）にする**。ブロックを録画時によく置いた範囲（トレイの左奥あたり、遠すぎない位置）に置く

### 2-2. 実行コマンド（残り 9 回を既存の評価データセットに追記）

```bash
cd <このリポジトリ>   # outputs/ と .venv/ はここに置く（gitignore）
# iPhone は開いた直後だけ 16:9 で届くことがあるので、先に一度開いて慣らす
.venv/bin/python -c "
import cv2,time
for t in range(5):
    c=cv2.VideoCapture(4); c.set(3,640); c.set(4,480)
    for _ in range(10): ok,f=c.read(); time.sleep(0.03)
    c.release()
    if ok and f.shape[0]==480: print('warm ok', t); break
    time.sleep(0.5)
"
# record_relaxed.py = lerobot-record と同じだが、カメラの「最新フレームが古すぎる」判定を 0.5 秒 → 2 秒に緩めたもの
.venv/bin/python tools/record_relaxed.py \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5C821074811 --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5C821066341 --teleop.id=my_leader \
  --policy.path=outputs/train/act_so101_block_place_50ep/030000/pretrained_model --policy.device=mps \
  --dataset.repo_id=local/eval_so101_block_place_50ep_x10 \
  --dataset.root=$HOME/.cache/huggingface/lerobot/local/eval_so101_block_place_50ep_x10 --resume=true \
  --dataset.single_task="Pick up the block and place it in the target area" \
  --dataset.num_episodes=9 --dataset.episode_time_s=20 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false --display_data=true
```

- エピソード中は学習済みモデルが動かし、準備時間（Reset the environment）はリーダーで操作できる（policy と teleop を両方渡している）
- 準備時間に入った瞬間、フォロワーはリーダーの姿勢に一気に動く。リーダーは休止姿勢で待つ
- `--resume=true` のときは `--dataset.root` が必須。`num_episodes` は「追加で録る回数」
- `lerobot-record` は準備時間のあとに直前の回を保存する。途中で落ちたら、保存済みの回数を `meta/info.json` の `total_episodes` で確認して、残りを同じコマンドで追記する

### 2-3. 成功判定

各回の最後（19 秒前後）のフレームで、ブロックがトレイの上にあれば成功。フレームの切り出し例：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/eval_so101_block_place_50ep_x10")
for ep in range(ds.meta.total_episodes):
    end = int(ds.meta.episodes[ep]["dataset_to_index"]) - 1
    img = ds[end]["observation.images.front"]  # C,H,W, 0-1
```

| 回 | 結果 | メモ |
|---|---|---|
| 1 | ❌ 失敗 | ブロックがトレイの奥（録画時より遠い位置）。アームは動くがつかめず |
| 2〜10 | 未実施 | |

---

## 3. 環境・データの場所

### ハードウェア

| 項目 | 値 |
|---|---|
| フォロワー | `/dev/tty.usbmodem5C821074811`、id `my_follower` |
| リーダー | `/dev/tty.usbmodem5C821066341`、id `my_leader` |
| 俯瞰カメラ | iPhone（Continuity カメラ）、OpenCV index **4**、640×480・30fps。キー名 `front` |
| 手首カメラ | index 0（icspring camera）。**現在は未使用**（マグネットマウントがグリッパーに干渉したため外した） |
| ほかのカメラ | 1・3 は Insta360（会議用）、2 は FaceTime HD（取得不可） |

### キャリブレーション（`~/.cache/huggingface/lerobot/calibration/`）

- `robots/so_follower/my_follower.json`：2026-10-12 20:57 に取り直したもの（グリッパー 1083〜2837）
- `robots/so_follower/my_follower.backup-20261012.json`：取り直す前（9/25 版）
- `teleoperators/so_leader/my_leader.json`：9/25 のまま

### データセット（`~/.cache/huggingface/lerobot/local/`）

| 名前 | 中身 |
|---|---|
| `so101_block_place` | 学習用 50 回（ep 0〜29：最初のカメラ位置、ep 30〜49：今のカメラ位置） |
| `so101_block_place.backup-30ep` | 追記前の 30 回版のバックアップ |
| `eval_so101_block_place` | 30 回版モデルの評価（失敗） |
| `eval_so101_block_place_nas25` | 30 回版モデル、`n_action_steps=25` の評価（失敗） |
| `eval_so101_block_place_50ep` | 50 回版モデルの評価 1 回（**初成功**） |
| `eval_so101_block_place_50ep_x10` | 成功率測定用。現在 1 回分（失敗） |

### モデル

| 場所 | 中身 |
|---|---|
| `outputs/train/act_so101_block_place/030000/pretrained_model` | 30 回版（ローカルコピー） |
| `outputs/train/act_so101_block_place_50ep/030000/pretrained_model` | **50 回版（評価に使っているもの）** |
| Google ドライブ `マイドライブ/lerobot/outputs/act_so101_block_place[_50ep]/checkpoints/` | 5k〜30k の全チェックポイント |

### Colab

- ノートブック：`マイドライブ/lerobot/train_act_colab.ipynb`（Colab Pro / A100）
- データ：`マイドライブ/lerobot/so101_block_place_50ep_parts/`（10MB 単位の 97 個 + `SHA256SUMS`）。ノートブックのセル 3 が結合・検証・展開する
- ノートブックに入っている修正：`pip uninstall -y transformers`（GR00T の import エラー回避）、`._*` ファイルの削除、分割ファイルの結合

### 自作スクリプト（`tools/`）

| ファイル | 用途 |
|---|---|
| `align_live.py` | iPhone の位置合わせ（録画時の輪郭を重ねて、ずれと直す方向を表示）。引数：カメラ番号、保存先、基準エピソード |
| `record_relaxed.py` | カメラのフレーム鮮度判定を 2 秒に緩めた `lerobot-record` |
| `calib_steps.py` | Enter の代わりに合図ファイル（`step1` / `step2`）で進めるキャリブレーション。引数：合図ファイルを置くフォルダ |
| `grip_probe2.py` | グリッパーのトルクを切って、生の位置を 75 秒記録 |
| `move_test.py` | 関節を 1 つずつ動かしながらの通信テスト（音声で合図） |
| `health.py` | 10 秒間の通信チェック + 全サーボの電圧 |
| `loss_chart.py` | 学習ログの CSV から loss のグラフ（PNG）を作る |

---

## 4. よく起きるトラブルと対処

| 症状 | 対処 |
|---|---|
| `Missing motor IDs: 6` / 終了時に `Overload error` | グリッパーの過負荷保護。フォロワーの電源を入れ直す（腕を支えてから） |
| `frame width=640 or height=360 do not match` | iPhone が開いた直後だけ 16:9 になる。起動前にカメラを一度開いて閉じる |
| `latest frame is too old: 500 ms` | iPhone の映像が一瞬止まった。`record_relaxed.py` を使う。iPhone は給電・自動ロックオフ |
| 評価でアームが的外れな場所へ向かう | カメラ位置のずれ。`align_live.py` で合わせ直す |
| バックグラウンド起動で `EOFError` | キャリブレーション確認の Enter 待ち。`printf '\n' \| ...` で渡すか `< /dev/null`（キャリブレーションが一致していれば出ない） |
| ドライブに大きいファイルが上がらない | 10MB に分割 + `SHA256SUMS`。ドライブのアプリを再起動 |

---

## 5. その後の候補

- 成功率が低ければ：ブロックを不透明で色のはっきりしたものに変える／手首カメラを干渉しない位置に付け直して 2 カメラにする／`n_action_steps` を 25〜50 で比較する
- ブログ：8-9 に成功率、付録にコマンド集を追加。公開前に、画像に個人情報（顔・書類など）が写っていないか最終確認する
