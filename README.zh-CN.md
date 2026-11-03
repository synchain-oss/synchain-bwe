# synchain-bwe

在本机给频宽不够的人声补上缺的高频，比如以 16、22.05 或 24 kHz 生成的 AI 人声。神经网络模型
（[LavaSR](https://github.com/ysharma3501/LavaSR)）猜截止频率以上应该有什么；synchain-bwe 只借用它猜的频谱形状，
音量逐帧跟着人声自己走，还可以用真人声音给亮度封顶。

> **状态：** 早期版本（0.1.0）。目前只在一种声音上测过：IndexTTS 2.0 以 22.05 kHz 生成的 77 句中文。用之前请先听结果。

[English](README.md)

## 怎么做的

分频点以下，输出就是输入，只是重采样到 48 kHz，其余不动。以上补一个频段：

1. **猜。** LavaSR 的补频宽模型按 LavaSR 自己的流程听输入：先降到 16 kHz 再送进去，然后猜到 24 kHz 的完整频段。
2. **定音量。** 每一帧（10.7 毫秒长，每 2.7 毫秒一帧）把猜的结果缩放到这样：它在边缘频段（分频点下面那 1.5 kHz，输入真的有）
   的音量和输入自己在这里的音量一样。所以补上的频段跟着人声走：人声这一刻多亮就多亮，人声没声音的地方也没有。最多抬 24 dB。
3. **滚降。** 16 kHz 起慢慢减弱，到 20 kHz 为零，去掉模型在上面的那层持续的嘶声。
4. **封顶**（给了 `--reference` 时）。取一段真人的全频段录音：分频点以上，每一段频谱相对同一帧的边缘频段，不超过参考录音里
   只有一成语音帧才超过的那个值。各帧拼回去以后再量一次、再压一次，一共四遍。
5. **分频。** 补的频段经过 500 Hz 宽的平滑分频，替换输入在分频点以上原有的内容。分频点放在输入开始往下掉的地方（截止频率）
   稍低一点，按每个文件自动找（`detect` 两个都会显示），也可以用 `--crossover` 指定。每个文件首尾 20 毫秒渐入渐出。

为什么不直接用模型的输出？在测试的声音上，它猜得比截止频率下面人声真实存在的内容低约 15 dB，16 kHz 以上主要是一层不跟着说话走的
嘶声，停顿里也有。`--mode raw` 输出的就是这个版本，可以拿来对比。

## 在一种声音上的测量结果

IndexTTS 2.0 以 22.05 kHz 生成的 77 句（约 10.3 kHz 以下完整，11 kHz 以上没有内容）。参考录音是这个 AI 声音所模仿的那个人的
19 秒 48 kHz 录音，经本人同意使用。

| | 输入 | `--mode raw` | 按原声电平，带参考 |
|---|---|---|---|
| 截止处的台阶（9.5–10 kHz 对 10.5–11 kHz） | 约 22 dB | 17.1 dB | 5.0 dB（各句 3.2–6.9） |
| 停顿里补的频段 | 没有 | 持续的嘶声 | 比 raw 低约 26 dB |

和真人比（每帧都相对自己的 8.5–10 kHz）：齿音处补的频段和真人差 3 dB 以内，最亮的一成齿音比真人暗 2–8 dB；其余语音（主要是元音）
12–18 kHz 比真人亮 3.5–5.5 dB。按整句平均，10.5–16 kHz 比真人暗 3–4 dB，16–20 kHz 和真人持平。

已知的限制：

- 再压几遍之后，约 1% 的语音帧仍超过上限 3 dB 以上，最多约 13 dB，都在齿音突然起头前的那一帧：模型补的频段早起约 2 毫秒
  （测试脉冲上中位数 1.8 毫秒），很轻，比紧跟着的齿音低 15–30 dB。作为对照，参考录音本身约 11% 的语音帧超过它自己的上限 3 dB 以上。
- 不给参考录音就没有封顶，只有最多抬 24 dB 和滚降。
- synchain-bwe 没有在 7 kHz 以下就截止的输入上测过（LavaSR 自己说它接受 8 kHz 及以上采样率的输入）。

## 安装

需要 Python 3.10 以上和 git。

```sh
python -m venv .venv
# Windows：.venv\Scripts\activate    macOS / Linux：source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # 可选：体积小一些的 CPU 版
pip install "synchain-bwe @ git+https://github.com/synchain-oss/synchain-bwe"
```

第一次运行会从 Hugging Face 下载两个文件（约 56 MB），是 [YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)
固定版本里的 `enhancer_v2`，加载前用 SHA-256 核对两个文件。`synchain-bwe fetch` 可以提前下载；已有副本用 `--model-dir` 指过去。

## 用法

```sh
synchain-bwe detect voice.wav                                   # 输入在哪里截止，分频放在哪里
synchain-bwe process voice.wav -o voice_48k.wav -r real_voice_48k.wav
synchain-bwe process "takes/*.wav" -d out/ -r real_voice_48k.wav
synchain-bwe ceiling real_voice_48k.wav --like voice.wav -o voice_ceiling.json
synchain-bwe process "takes/*.wav" -d out/ -c voice_ceiling.json  # 用存好的上限代替录音
synchain-bwe process voice.wav -o band.wav --band-only          # 只要补的频段，在 DAW 里垫在 48 kHz 的原声下面
synchain-bwe process voice.wav -o raw.wav --mode raw            # 模型原样的猜测，用来对比
```

| 选项 | |
|---|---|
| `-o`、`-d` | 输出文件（一个输入）或输出文件夹（任意个，文件名不变，`--band-only` 时是 `.band.wav`） |
| `-r`、`-c` | 参考录音，或者用 `synchain-bwe ceiling` 存好的上限文件 |
| `--crossover HZ` | 分频从哪里开始；默认按每个文件自动找 |
| `--mode raw` | 不重新定音量，直接用模型的猜测 |
| `--band-only` | 只输出补的频段 |
| `--float` | 输出 32 位浮点，默认 24 位 |
| `--model-dir DIR` | 用这个文件夹里的模型文件（照样核对 SHA-256） |
| `--threads N` | 模型用几个 CPU 线程 |
| `--report FILE.json` | 每个文件一条 JSON 记录：分频点、前后台阶、抬到 24 dB 上限的帧、被封顶的帧，以及用到的上限 |
| `--force` | 覆盖已有的输出和报告（输入、参考录音和上限文件永远不会被覆盖） |

输出一律是 48 kHz WAV，声道数和输入相同，每个声道分开处理。找不到明显截止频率的文件会跳过并报告。长文件按 30 秒一段处理，拼接处
和整个文件一次处理的结果相差远低于 24 位精度（实测：用模型处理测试声音时 −175 dB，测试里约 −150 dB）。在一台台式机 CPU（AMD Ryzen 9 9950X）上用 8 个线程，速度约是实时的 15 倍。

### 参考录音

用一段干净的真人录音，声音和 AI 人声越接近越好，最好就是这个 AI 声音所模仿的那个人，并经本人同意：44.1 或 48 kHz，
频段完整（20 kHz 以下没有被切掉），10–30 秒说话（至少 4 秒），没有音乐和噪声，没有做过去齿音或 EQ。提前截止、或者分频点以上几乎没有内容的参考录音会被拒绝。上限是相对分频点下面那一段算的，所以一个上限文件只适用于分频点相同的输入（`--like` 会从某个输入取分频点）。

## 隐私

所有处理都在你的电脑上。唯一的联网是第一次从 Hugging Face 下载模型。音频不会发到任何地方。上限文件是描述参考声音的几个数字，请按对待
那个声音的方式对待它。`--report` 输出的报告里也有这些数字。

## 致谢和许可证

- synchain-bwe：MIT，见 [LICENSE](LICENSE)。
- 模型和权重来自 Yatharth Sharma 的 [LavaSR](https://github.com/ysharma3501/LavaSR)
  （[YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)）。它的 LICENSE 文件和模型卡写的是 Apache-2.0，包元数据里写的是 MIT。
  synchain-bwe 不包含也不安装 LavaSR 的代码：它只下载权重，用自己的加载代码运行，其中照做了 LavaSR 对 Vocos 输出头的唯一改动
  （幅度上限 1e3）。
- [Vocos](https://github.com/charactr-platform/vocos)（MIT，Charactr Inc.）以 LavaSR 所用的
  [langtech-bsc 分支](https://github.com/langtech-bsc/vocos)、固定提交作为依赖安装。

synchain-bwe 与 LavaSR、Vocos 的作者没有关联，也未获其认可。
