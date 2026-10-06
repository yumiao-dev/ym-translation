#!/bin/bash
# ym-Translation Stage2 续训 —— 一键启动（在 AMD MI300X 实例的终端里跑）
# 用法：bash /mnt/workspace/ym-Translation/scripts/start_stage2.sh
set -u
cd /mnt/workspace/ym-Translation || exit 1

echo "=============================================="
echo " 1) 设备检查"
echo "=============================================="
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("mem GB", round(p.total_memory/1024**3, 1))
    print("count", torch.cuda.device_count())
else:
    print("!! 没有可用 GPU —— 请确认切到了 AMD MI300X 实例")
PY

echo
echo "=============================================="
echo " 2) 数据检查"
echo "=============================================="
echo -n "原有 data/tokens/   : "; ls data/tokens/part_*.bin 2>/dev/null | wc -l
echo -n "新增 data/tokens2/  : "; ls data/tokens2/*.bin 2>/dev/null | wc -l
du -sh data/tokens data/tokens2 2>/dev/null
echo -n "旧检查点 output/ckpt/: "; ls output/ckpt/*.pt 2>/dev/null | wc -l
ls -lh output/ckpt/ 2>/dev/null | tail -5

echo
echo "=============================================="
echo " 3) 启动训练（tmux 会话名 t2）"
echo "=============================================="
tmux kill-session -t t2 2>/dev/null
rm -f /tmp/train2.log
tmux new-session -d -s t2 \
  "cd /mnt/workspace/ym-Translation && TARGET_TOKENS=4200000000 LR_PEAK=3e-4 LR_MIN=3e-5 WARMUP=60 NEW_DATA_REPEAT=2 python3 scripts/train_trans2.py > /tmp/train2.log 2>&1"

echo "已启动。等 45 秒看日志..."
sleep 45
echo
echo "--- /tmp/train2.log ---"
tail -30 /tmp/train2.log
echo
echo "=============================================="
echo " 常用命令："
echo "  看进度:  tail -20 /tmp/train2.log"
echo "  实时看:  tail -f /tmp/train2.log"
echo "  进会话:  tmux attach -t t2   (退出: Ctrl+B 然后 D)"
echo "  看GPU:   rocm-smi"
echo "=============================================="
