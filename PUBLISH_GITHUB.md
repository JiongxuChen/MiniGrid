# 发布到 GitHub

此目录是baseline v1的公开导出，仅seed0、1、2；原研究目录保持不变。发布前先检查仓库内容和权利归属。GitHub仓库设为Public并不自动授予开源许可；请由代码所有者决定LICENSE，并保留第三方归属说明。

## 1. 本地检查

```bash
cd ~/MiniGrid
python baseline.py verify
python baseline.py summarize
python baseline.py plot
git status --short
git diff -- README.md
```

确认仅包含代码、配置、三个seed的记录和模型；不要加入个人凭据、环境目录或新的长训练输出。已有`.git`保留，未替你提交或推送。若远端已有历史，先查看并协调，勿使用force push：

```bash
git remote -v
git branch --show-current
git log --oneline -5
```

## 2. 发布范围：只上传六个权重

**轻量实验记录全部上传，模型ZIP只上传以下六个文件：**

| seed | Final主模型 | Best补充模型 |
|---|---|---|
| 0 | `results/seed_0/final_model.zip` | `results/seed_0/checkpoints/step_000950000.zip` |
| 1 | `results/seed_1/final_model.zip` | `results/seed_1/checkpoints/step_000750000.zip` |
| 2 | `results/seed_2/final_model.zip` | `results/seed_2/checkpoints/step_000850000.zip` |

Best仍沿用原checkpoint文件名，避免复制和改变结果引用。其余57个周期／初始化权重已移出待发布目录，原研究资料保留。所有周期验证、选择依据、学习曲线、日志、测试CSV和轻量JSON仍在，JSON中未分发模型的哈希仅为历史记录，不代表权重已提供。

`.gitignore`采用ZIP默认忽略、上述六个文件明确放行的白名单。不要使用`git add -f`绕过它，也不要手动放宽白名单上传其他checkpoint。新的训练产物在`runs/`，整个目录仍被忽略。

上传前核对Git可见的权重列表：

```bash
git ls-files --cached --others --exclude-standard -- '*.zip'
```

列表应恰好对应表中六个文件。`python baseline.py verify`也会核验实际六个模型、校验清单和final／best测试引用。

默认直接通过普通Git提交这六个权重即可。若选择Git LFS，在首次加入模型前安装并配置：

```bash
git lfs install
git lfs track 'results/seed_*/final_model.zip'
git lfs track 'results/seed_*/checkpoints/step_*.zip'
```

这不会放行其他被忽略的ZIP。使用LFS时，克隆后先执行`git lfs pull`再做模型校验，`ARTIFACTS.json`校验的是真实文件而非LFS指针。Git LFS是否启用不改变仅六个权重的发布范围。

## 3. 提交

为自己的Git身份设置真实信息，不要复制他人身份。检查暂存区再提交：

```bash
git add README.md PUBLISH_GITHUB.md requirements.txt .gitignore .gitattributes baseline.py train_minigrid.py minigrid_baseline configs docs results ARTIFACTS.json
git diff --cached --stat
git status --short
git commit -m "Publish MiniGrid PPO baseline with three-seed final and supplementary best results"
```

若已决定许可证，也加入LICENSE。

## 4. 创建远端并推送

在GitHub网页新建空Public仓库（不要自动初始化README，以免与本地历史冲突）。若当前仓库尚无origin，再运行以下命令，把占位符替换为真实账号/仓库名：

```bash
git remote add origin https://github.com/YOUR_USER/YOUR_REPOSITORY.git
git push -u origin HEAD
```

若已有origin，确认它确实是目标仓库后使用`git push -u origin HEAD`，不要盲目覆盖远端配置。认证使用GitHub支持的凭据管理器、SSH或token，不将token写入仓库或命令文档。

## 5. 发布后核验

从新目录克隆仓库，安装依赖，执行`python baseline.py verify`。使用LFS则先拉取真实权重。确认README、三个seed的轻量记录，以及表中六个模型可访问。其他周期权重不属于公开包，重放它们需要按文档自行训练。

最终主结果始终是final；best来自6×6验证集，仅为补充。仓库中不包含其他baseline版本或改进方向。不要用重新挑选seed或按测试表现挑checkpoint修改已发表主结果。
