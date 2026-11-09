#!/usr/bin/env bash
# 公開リポジトリに push する前の確認。個人情報・秘密が混ざっていないかを機械的に調べる。
#   tools/check_public.sh            これから push するコミット（origin/main..HEAD。origin/main がなければ全履歴）
#   tools/check_public.sh <base>     <base>..HEAD
#   tools/check_public.sh --staged   コミット前の差分
# 見つかったら 1 で終わる。画像・動画は中身を機械では判定できないので、一覧を出して人の確認に回す。
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
empty_tree=$(git hash-object -t tree /dev/null)
authors=""
if [[ "${1:-}" == "--staged" ]]; then
  desc="staged"; diff=(git diff --cached); names=(git diff --cached --name-only --diff-filter=AM)
else
  base="${1:-origin/main}"
  if git rev-parse -q --verify "$base" >/dev/null; then
    desc="$base..HEAD"; range="$base..HEAD"; from="$base"
  else
    desc="全履歴（$base がない）"; range="HEAD"; from="$empty_tree"
  fi
  diff=(git diff "$from" HEAD); names=(git diff --name-only --diff-filter=AM "$from" HEAD)
  authors="$(git log --format='%ae%n%ce' $range | sort -u)"
fi
fail=0
bad="$(printf '%s\n' "$authors" | grep -v -E '^$|users\.noreply\.github\.com$' || true)"
[[ -n "$bad" ]] && { echo "✗ noreply 以外のコミットのメール: $bad"; fail=1; }
pat='[A-Za-z0-9._%+-]+@(gmail|yahoo|icloud|outlook|hotmail)\.[a-z.]+|/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+/|/private/tmp/|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY'
hits="$("${diff[@]}" -- . ':!uv.lock' ':!tools/check_public.sh' | grep -E '^\+' | grep -v -E '^\+\+\+' | grep -E "$pat" || true)"
[[ -n "$hits" ]] && { echo "✗ 個人のパス・メール・鍵らしい文字列:"; printf '%s\n' "$hits" | cut -c1-200; fail=1; }
media="$("${names[@]}" | grep -i -E '\.(jpe?g|png|gif|webp|heic|mp4|mov)$' || true)"
[[ -n "$media" ]] && { echo "△ 画像・動画（顔・画面の文字・部屋の様子が写っていないか目で確認）:"; printf '  %s\n' $media; }
while read -r f; do
  [[ -f "$f" ]] || continue
  s=$(wc -c <"$f")
  (( s > 20000000 )) && { echo "✗ 20 MB を超えるファイル（データや重みは Hugging Face Hub へ）: $f"; fail=1; }
  # Zenn は images/ の下の画像しか読まず、1 枚 3 MB まで
  if [[ "$f" == images/* ]]; then
    [[ "$f" =~ \.(png|jpe?g|gif|webp)$ ]] || { echo "✗ images/ に画像以外のファイル（Zenn の同期が失敗する）: $f"; fail=1; }
    (( s > 3000000 )) && { echo "✗ 3 MB を超える画像（Zenn に載らない）: $f"; fail=1; }
  fi
done < <("${names[@]}")
# Zenn の記事: フロントマターと画像の参照先
for a in articles/*.md; do
  [[ -f "$a" ]] || continue
  head -1 "$a" | grep -q '^---$' || { echo "✗ フロントマターがない: $a"; fail=1; }
  grep -q '^published: ' "$a" || { echo "✗ published: がない: $a"; fail=1; }
  while read -r img; do
    [[ -f ".${img}" ]] || { echo "✗ $a が参照する画像がない: $img"; fail=1; }
  done < <(grep -o -E '\]\(/images/[^)]+\)' "$a" | sed -E 's/^\]\(//; s/\)$//')
done
[[ $fail -eq 0 ]] && echo "✓ 公開前チェック OK（${desc}）"
exit $fail
