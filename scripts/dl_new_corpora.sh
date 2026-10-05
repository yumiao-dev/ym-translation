#!/bin/bash
# ym-Translation 补训数据下载（在 CPU 实例上跑）
# 语料源：OPUS 官方 object.pouta.csc.fi（可直连，支持断点续传）
set -u
DEST=/mnt/workspace/ym-Translation/data/new
mkdir -p "$DEST/zip" "$DEST/raw"
cd "$DEST/zip" || exit 1

BASE=https://object.pouta.csc.fi

# 格式： 短名|URL
CORPORA=(
  # ---- P0 口语 ----
  "OpenSubtitles_v2016|$BASE/OPUS-OpenSubtitles/v2016/moses/en-zh.txt.zip"
  "TED2020|$BASE/OPUS-TED2020/v1/moses/en-zh.txt.zip"
  "TED2013|$BASE/OPUS-TED2013/v1.1/moses/en-zh.txt.zip"
  "NeuLab-TedTalks|$BASE/OPUS-NeuLab-TedTalks/v1/moses/en-zh.txt.zip"
  # ---- P0 高质人工译 ----
  "UNPC|$BASE/OPUS-UNPC/v1.0/moses/en-zh.txt.zip"
  "News-Commentary_v16|$BASE/OPUS-News-Commentary/v16/moses/en-zh.txt.zip"
  # ---- P1 正式/长句 ----
  "MultiUN|$BASE/OPUS-MultiUN/v1/moses/en-zh.txt.zip"
  "WikiMatrix|$BASE/OPUS-WikiMatrix/v1/moses/en-zh.txt.zip"
  "XLEnt|$BASE/OPUS-XLEnt/v1.2/moses/en-zh.txt.zip"
  "wikimedia|$BASE/OPUS-wikimedia/v20260327/moses/en-zh.txt.zip"
  "ALT|$BASE/OPUS-ALT/v20191206/moses/en-zh.txt.zip"
  "QED|$BASE/OPUS-QED/v2.0a/moses/en-zh.txt.zip"
  "Tanzil|$BASE/OPUS-Tanzil/v1/moses/en-zh.txt.zip"
  "PHP|$BASE/OPUS-PHP/v1/moses/en-zh.txt.zip"
  "infopankki|$BASE/OPUS-infopankki/v1/moses/en-zh.txt.zip"
  "WMT-News|$BASE/OPUS-WMT-News/v2019/moses/en-zh.txt.zip"
  "tico-19|$BASE/OPUS-tico-19/v2020-10-28/moses/en-zh.txt.zip"
  "tldr-pages|$BASE/OPUS-tldr-pages/v2026-07-07/moses/en-zh.txt.zip"
)

echo "=== START $(date '+%F %T') ==="
OK=0; FAIL=0
for item in "${CORPORA[@]}"; do
  name="${item%%|*}"; url="${item#*|}"
  out="$name.txt.zip"
  if [ -s "$out" ] && unzip -tq "$out" >/dev/null 2>&1; then
    sz=$(du -h "$out" | cut -f1)
    echo "[SKIP] $name ($sz)  已存在且完整"
    OK=$((OK+1)); continue
  fi
  printf '[GET ] %-22s ' "$name"
  if curl -sSL --retry 5 --retry-delay 3 --retry-all-errors \
        -C - -o "$out" --max-time 900 "$url"; then
    sz=$(du -h "$out" | cut -f1)
    if unzip -tq "$out" >/dev/null 2>&1; then
      echo "OK   $sz"; OK=$((OK+1))
    else
      echo "BAD  $sz (zip 损坏)"; FAIL=$((FAIL+1))
    fi
  else
    echo "FAIL (curl $?)"; FAIL=$((FAIL+1))
  fi
done

echo "=== 下载完成 OK=$OK FAIL=$FAIL $(date '+%F %T') ==="

# ---- 解压到 raw/ ----
cd "$DEST/raw" || exit 1
for z in ../zip/*.txt.zip; do
  [ -s "$z" ] || continue
  n=$(basename "$z" .txt.zip)
  if [ -s "$n.en" ] && [ -s "$n.zh" ]; then echo "[SKIP-UNZIP] $n"; continue; fi
  unzip -o -q "$z" -d /tmp/_uz_$$ 2>/dev/null || { echo "[UNZIP-FAIL] $n"; continue; }
  # OPUS moses zip 里是 <name>.en / <name>.zh
  mv /tmp/_uz_$$/*.en "$n.en" 2>/dev/null
  mv /tmp/_uz_$$/*.zh "$n.zh" 2>/dev/null
  rm -rf /tmp/_uz_$$
  if [ -s "$n.en" ] && [ -s "$n.zh" ]; then
    echo "[UNZIP] $n  lines=$(wc -l < "$n.en")"
  else
    echo "[UNZIP-FAIL] $n"
  fi
done

echo "=== ALL DONE $(date '+%F %T') ==="
du -sh "$DEST"
