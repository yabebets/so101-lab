# so101-lab

SO-101 ロボットアーム（Hiwonder 版 SO-ARM101）を使って、模倣学習・VLM / VLA・強化学習・ロボティクス、つまりフィジカル AI を手を動かしながら学んでいく記録です。うまくいかなかったことも含めて、作業のログをそのまま公開しています。

*A public learning log: physical AI (imitation learning, VLM/VLA, RL) on a Hiwonder SO-ARM101 with LeRobot. Posts are in Japanese.*

## 学習ログ

記事は [articles/](articles/) にあり、Zenn にも同じものを載せます。

| 日付 | 記事 | 結果 |
|---|---|---|
| 2026-10-12 | [実機で模倣学習：テレオペから ACT 学習まで](articles/so101-imitation-learning-act.md) | 50 回のデモで学習したモデルが実機で初成功。成功率の測定は途中 |
| 2026-10-18 | MetaWorld（push-v3）で ACT を試す（Colab、記事は未作成） | 未知の初期配置で 20/20 |
| 2026-10-25 | [SO-ARM101 の MuJoCo モデルでシミュレーション上の模倣学習](articles/so101-mujoco-sim-act.md) | スクリプト教師は 20/20、ACT は最大 5/10 |

これから何をやるかは [ROADMAP.md](ROADMAP.md) にあります。実機作業の引き継ぎメモは [notes/HANDOFF.md](notes/HANDOFF.md) です。

## 環境

| | |
|---|---|
| ロボット | Hiwonder SO-ARM101（リーダーとフォロワーの 2 本、サーボは Hiwonder HX-30HM） |
| ソフト | LeRobot 0.5.1（[Hiwonder のフォーク](https://github.com/Hiwonder-official/hiwonder-SoArm-101)）、Python 3.13、MuJoCo |
| カメラ | iPhone（Continuity Camera）を俯瞰カメラとして使用 |
| 計算 | 手元は Apple M1 Max（MPS）。重い学習は Google Colab の A100 |

## 動かし方

[uv](https://docs.astral.sh/uv/) で環境を作ります。LeRobot は記事と同じ版（Hiwonder のフォークのコミット）に固定してあります。

```bash
uv sync
# シミュレーションのスクリプト教師を眺める（macOS は mjpython が必要）
.venv/bin/mjpython sim/view_expert.py --loop --speed 0.5
# sim でデモを集める → LeRobot のデータセットになる
.venv/bin/python sim/collect_dataset.py --episodes 50 --root sim/data/so101_sim_pick_place
```

学習には LeRobot の `lerobot-train` を使います。評価は `sim/eval_policy.py` です（使い方はファイルの先頭）。

## リポジトリの中身

```
articles/          記事（Zenn の形式）
images/<記事>/     記事の図
sim/               SO-ARM101 の MuJoCo 環境・スクリプト教師・データ収集・評価
tools/             実機用の小さな道具（サーボの確認、カメラの位置合わせなど）と公開前チェック
notes/             実機作業の引き継ぎメモ、学習ログの生データ
runs/              実験の記録（設定と結果の数字）
ROADMAP.md         学ぶ順番
research.yaml      進み具合の台帳（タスク・作業の記録）
```

## データとモデル（Hugging Face Hub）

| 名前 | 中身 |
|---|---|
| [yabebets/so101_block_place](https://huggingface.co/datasets/yabebets/so101_block_place) | 実機でテレオペしたデモ 50 回（LeRobot v3.0 形式） |
| [yabebets/act_so101_block_place_50ep](https://huggingface.co/yabebets/act_so101_block_place_50ep) | 上のデータで学習した ACT |
| [yabebets/so101_sim_pick_place_v2](https://huggingface.co/datasets/yabebets/so101_sim_pick_place_v2) | MuJoCo のスクリプト教師のデモ 100 回 |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("yabebets/so101_block_place")
```

## 出典とライセンス

- 記事と図（`articles/`・`images/`）は [CC BY 4.0](LICENSE-CC-BY-4.0)、コード（`sim/`・`tools/` など）は [Apache License 2.0](LICENSE) です。
  記事や図を使うときは、出典として「so101-lab（yabebets）」とこのリポジトリへのリンクを書いてください。
- LeRobot 本体は [Hugging Face の LeRobot](https://github.com/huggingface/lerobot) と、それを Hiwonder のサーボに対応させた [Hiwonder-official/hiwonder-SoArm-101](https://github.com/Hiwonder-official/hiwonder-SoArm-101) を依存として使っています（Apache License 2.0）。
- `sim/so101/` のロボットモデルは [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) の SO-101 モデルをもとに手を加えたものです（Apache License 2.0、[sim/so101/SOURCE.md](sim/so101/SOURCE.md)）。
