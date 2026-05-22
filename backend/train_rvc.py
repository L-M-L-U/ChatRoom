"""
RVC 训练数据准备与指引脚本

处理流程：
  1. 扫描音频目录，校验格式与总时长
  2. 静音切除（silence removal）
  3. 输出训练指引

用法：
  python backend/train_rvc.py --input_dir ./my_audio --output_dir ./dataset
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EXPECTED_SR = {22050, 24000}
MIN_DURATION_SEC = 300  # 5 分钟


def check_audio_dir(input_dir: str) -> list:
    """扫描目录，返回有效的 WAV 文件路径列表。"""
    valid = []
    for f in sorted(Path(input_dir).iterdir()):
        if f.suffix.lower() not in (".wav",):
            continue
        valid.append(str(f))

    if not valid:
        print("❌ 未找到任何 WAV 文件。")
        sys.exit(1)

    print(f"✅ 找到 {len(valid)} 个 WAV 文件。")
    return valid


def validate_audio(files: list) -> tuple:
    """
    快速验证采样率和总时长。
    返回 (total_seconds, errors)。
    """
    try:
        import soundfile as sf
    except ImportError:
        print("⚠️  未安装 soundfile，尝试使用 pydub…")
        return _validate_fallback(files)

    total = 0.0
    errors = []
    for fp in files:
        try:
            info = sf.info(fp)
            if info.samplerate not in EXPECTED_SR:
                errors.append(f"  ⚠ {fp}: 采样率 {info.samplerate} Hz，建议 {EXPECTED_SR}")
            total += info.duration
        except Exception as e:
            errors.append(f"  ⚠ {fp}: 读取失败 ({e})")

    return total, errors


def _validate_fallback(files: list) -> tuple:
    """使用 pydub 作为降级方案。"""
    from pydub import AudioSegment

    total = 0.0
    errors = []
    for fp in files:
        try:
            seg = AudioSegment.from_file(fp)
            sr = seg.frame_rate
            dur = seg.duration_seconds
            if sr not in EXPECTED_SR:
                errors.append(f"  ⚠ {fp}: 采样率 {sr} Hz，建议 {EXPECTED_SR}")
            total += dur
        except Exception as e:
            errors.append(f"  ⚠ {fp}: 读取失败 ({e})")

    return total, errors


def trim_silence(input_dir: str, output_dir: str):
    """对每个 WAV 执行静音切除，输出到 output_dir。"""
    from pydub import AudioSegment, silence

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for fp in sorted(Path(input_dir).iterdir()):
        if fp.suffix.lower() != ".wav":
            continue
        try:
            audio = AudioSegment.from_file(str(fp))
            chunks = silence.split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=-40,
                keep_silence=200,
            )
            if not chunks:
                print(f"  ⚠ {fp.name}: 全静音，跳过")
                continue

            merged = sum(chunks, AudioSegment.empty())
            out_path = os.path.join(output_dir, fp.name)
            merged.export(out_path, format="wav", parameters=["-ac", "1", "-ar", "24000"])
            count += 1
            print(f"  ✓ {fp.name} ({len(chunks)} 段 -> {merged.duration_seconds:.1f}s)")
        except Exception as e:
            print(f"  ✗ {fp.name}: 处理失败 ({e})")

    print(f"\n✅ 已处理 {count} 个文件，输出至: {output_dir}")


def print_training_guide(data_dir: str):
    """输出训练指引。"""
    print("\n" + "=" * 60)
    print("  RVC 训练指引")
    print("=" * 60)
    print(f"""
数据集目录: {os.path.abspath(data_dir)}

步骤：
  1. 下载并解压 RVC-WebUI: https://github.com/RVC-Project/RVC-WebUI/releases
  2. 运行 go-webui.bat 启动 Web 界面
  3. 在浏览器中打开 http://127.0.0.1:7865
  4. 进入"训练"页面 -> "数据预处理"
     - 训练数据路径: {os.path.abspath(data_dir)}
     - 点击"数据预处理"
  5. 处理完成后点击"特征提取"
  6. 最后点击"训练模型"
     - 推荐参数：Total Epoch = 300, Batch Size = 4/6

注意：
  - GPU (NVIDIA) 可大幅加速训练
  - 训练完成后，.pth 模型文件在 logs/ 目录下
  - 在 backend/main.py 中设置 RVC_MODEL_PATH 指向该文件
""")


def main():
    parser = argparse.ArgumentParser(description="RVC 训练数据准备")
    parser.add_argument("--input_dir", required=True, help="原始 WAV 音频文件夹")
    parser.add_argument("--output_dir", default="./dataset", help="处理后数据集输出路径")
    args = parser.parse_args()

    # 1. 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    print(f"📂 输入目录: {os.path.abspath(args.input_dir)}")

    # 2. 扫描并验证
    files = check_audio_dir(args.input_dir)
    total_sec, errors = validate_audio(files)

    for err in errors:
        print(err)

    if total_sec < MIN_DURATION_SEC:
        print(f"\n❌ 有效音频总时长 {total_sec / 60:.1f} 分钟，不足 {MIN_DURATION_SEC // 60} 分钟。")
        sys.exit(1)
    print(f"✅ 音频总时长 {total_sec / 60:.1f} 分钟，满足要求。")

    # 3. 静音切除
    print("\n--- 开始静音切除 ---")
    trim_silence(args.input_dir, args.output_dir)

    # 4. 训练指引
    print_training_guide(args.output_dir)


if __name__ == "__main__":
    main()