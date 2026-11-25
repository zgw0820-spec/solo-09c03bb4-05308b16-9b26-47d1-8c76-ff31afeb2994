# 产线缺陷分数推理 · 单帧缺口补算（卷持久化、崩溃续跑）

FastAPI 推理网页：内置固定权重的 NumPy 递推模型和两路样例。按相机 + 帧号提交
8 维特征向量，每路允许暂缺一帧；补交原帧后从缺口前的状态按帧序补算，补算追上
本路最新已出分帧后，才把修正状态交给后续推理，已发分数不改。所有状态写入 Docker
命名卷上的 JSON 文件（临时文件 + `fsync` + 原子改名），容器意外退出后重开网页
即可看到两路原分/修正分，并从各自下一帧继续。

## 启动

```bash
docker compose up --build
```

打开 <http://localhost:8000> 。网页里可以：

- 「试跑：边补帧边进新帧」——一键跑完整演示（A 路缺第 4 帧、B 路缺第 8 帧，
  补算 A 的同时 B 和 A 自己都继续进新帧）；
- 每个相机卡片里单步「送入下一帧 / 标记缺失 / 补交原帧 / 补算故障演练」；
- 「手动提交一帧」表单按相机和帧号提交任意 8 维特征向量；
- 「操作记录」里看到每次出分、暂估、补算范围和修正量。

不用 Docker 也可以本地跑（Python ≥ 3.11）：

```bash
python -m pip install -r requirements.txt
# 可选：不设 QUALITY_DATA_DIR 时退化为纯内存模式（重启即丢，仅用于快速调试）
QUALITY_DATA_DIR=./.data python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## 规则怎么落实的

| 需求 | 实现 |
| --- | --- |
| 每路允许暂缺一帧 | 每相机维护 `last_frame`；下一帧跳过一个帧号即打开缺口，缺口未补时不允许再缺第二帧（`409`） |
| 用零向量代入、受影响分数标暂估 | 缺口帧以零向量推进，`estimated=true`；缺口未补期间后续帧从临时状态继续出分，同样标暂估 |
| 按相机和帧号提交特征 | `POST /api/cameras/{camera}/frames`，也可 `POST /api/frames` 平铺提交 |
| 补交原帧后从缺口前状态按帧序补算 | 开缺口时保存 `state_before`，补交时从它开始逐帧重放缺口帧到最新已出分帧 |
| 两路仍照常接帧、出分 | 每相机独立锁，补算在锁外做模型运算；其它相机和本相机新帧都不被阻塞 |
| 追上本路最新已出分帧才切换状态 | 重放目标取当前 `last_output_frame`，每帧检查是否又长了；追平后才一次性提交修正状态 |
| 已发分数不改 | `score` 字段只写一次；修正值写入 `revised_score/delta/revision`，界面两列并列 |
| 补算失败按原状态出分、可重试 | 重放期间不提交任何状态；失败返回 `422 {"retryable": true, "state_preserved": true}`，缺口和临时状态原样保留 |
| 固定权重 NumPy 递推模型 | `app/model.py` 里的权重字面量，`GET /api/model` 给出权重指纹，运行时不训练 |
| 容器退出后状态不丢 | `QUALITY_DATA_DIR`（Compose 里是命名卷 `quality-inference-data` 挂到 `/data`）下每相机一个状态文件，启动时整体恢复 |
| 整笔写入成功才生效 | 送帧/标缺/补算提交/重置都先在暂存对象上构造新状态，原子写盘（同目录临时文件 `fsync` 后 `os.replace`）成功才采纳到内存；写失败返回 `503 {"retryable": true, "state_preserved": true}`，内存与磁盘都停在上一笔已生效状态，原请求可原样重试 |
| 补算提交前退出，复机可续 | 补算开始时先把缺口帧特征、缺口前状态写入 `pending_fill`；进程退出后重开，缺口、暂估状态、其间已收帧全部恢复，重新 `POST /fill` 同一帧即从缺口前状态续算（响应里 `replay.resumed=true`） |
| 补算已提交，修正分与递推状态一起恢复 | 提交是一次整笔原子写：窗口内每帧的 `revised_score/delta/revision` 与修正后的隐状态同生共死，重启后下一帧直接用修正状态出分 |
| 送帧只保存本次变化、重启不重算旧帧 | 每次写整份单相机文档（只含追加的新帧/新修正），重启直接读回完整文档，不回放历史帧 |

## 卷上的文件

```
/data（容器内；宿主机上是命名卷 quality-inference-data）
├── manifest.json          格式版本、固定权重指纹、相机清单（启动时校验，权重变了拒绝混读）
├── camera-0.state.json    A 路整份状态：隐状态、帧特征、已发分/修正分、缺口、pending_fill、事件
├── camera-1.state.json    B 路同上，两路互不影响
└── journal.jsonl          追加式审计流水（仅记录真正生效的事件；状态文件才是事实来源）
```

写盘顺序固定为「同目录临时文件 → flush/fsync → `os.replace` 原子改名 → 目录
fsync」，所以断电/被杀也只会留下上一份完整文件或新文件，不会出现半截 JSON。

模型：`h_t = tanh(W x_t + U h_{t-1} + b)`，`s_t = sigmoid(v·h_t + bias)`，
8 维输入、6 维隐状态、分数 0–1。样例数据在 `app/data/samples.json`，两路各 18 帧，
A 路第 6/7 帧、B 路第 5/10 帧带明显缺陷特征，方便观察分数跳变。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health`、`/api/model`、`/api/samples` | 健康检查（含 `storage` 持久化模式与卷目录）、模型信息、样例数据 |
| GET | `/api/status` | 两路汇总（缺口、暂估数、补算数、`pending_fill`、状态指纹、存储信息） |
| GET | `/api/cameras/{camera}/state` | 状态、分数、修正值、`pending_fill`、事件 |
| POST | `/api/cameras/{camera}/frames` | `{frame_no, features[8]}` 提交一帧（整笔写盘成功才生效） |
| POST | `/api/cameras/{camera}/missing` | `{frame_no}` 声明该帧缺失 |
| POST | `/api/cameras/{camera}/fill` | `{frame_no, features[8], pace_ms?, simulate_failure?}` 补交并补算；中断后重发同一帧即续算 |
| POST | `/api/reset` | `{confirm: true}` 清空（先删卷上状态文件，再清内存；补算中的任务会被取消且不提交） |

所有写接口在持久化故障时返回 `503`，载荷为
`{"code":"persistence_failed","retryable":true,"state_preserved":true}`，
此前已生效的结果和状态不变，客户端按相同请求体重试即可。

`pace_ms` 只在演示时放慢补算便于观察；`simulate_failure` 是演示用故障注入，
在重放第一帧后失败，用来验证“失败仍按原状态出分、可重试”。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                 # 43 个单元/接口/持久化测试
python scripts/check_interleaved.py   # 边补帧边进新帧的端到端场景
python scripts/check_restart.py       # 交付核对：SIGKILL 重启 + 补算中断续跑
```

`scripts/check_interleaved.py` 自己拉起一个本地 uvicorn，逐条打印 PASS/FAIL：

1. 缺口期间照常出分、标暂估；
2. 补算期间另一路和本路新帧都继续出分；
3. 补算目标从 #8 长到 #9，追平后才提交；
4. 已发分数逐帧不变，修正值与“无缺口重算”逐位一致；
5. 失败重试：状态指纹、缺口、已发分数都不变，之后仍能出分，重试追平。

`scripts/check_restart.py` 用临时目录模拟命名卷，两次真实重启 uvicorn 进程：

1. 两路各开缺口、A 路补算失败落盘并继续进帧后 `SIGKILL`；
2. 同卷重启：原分、暂估递推状态、缺口、`pending_fill`（含失败标记）逐帧恢复；
3. 复机后两路从各自下一帧继续，补算期间另一路照常出分；
4. 重新补交缺口帧：标记为续算（`resumed=true`），追上复机后最新帧，修正分与
   无缺口参照逐位一致，下一帧使用恢复后的修正递推状态；
5. 再次重启：窗口内修正分仍在、缺口已清空，B 路缺口可独立补交。

## 边界与未完成

- 持久化是单实例本地卷方案（JSON 整文件原子替换），没有多副本共享存储或
  分布式锁；水平扩容需要换成带事务的数据库，不在本次范围内。
- 每相机每次状态变化写一份完整文档，按演示规模（每路 ≤ 1 万帧）足够；真实
  产线长周期运行应再加分片/滚动归档。
- 相机只支持两路（0/1），是配置常量而不是数据库配置；`manifest.json` 会在
  相机集合或模型权重指纹变化时拒绝启动，避免把旧状态混进新模型。
- 每路同一时间只允许一个缺口；缺口窗口内再出现第二个零向量帧时，补算会明确
  拒绝（`422`）而不是猜，需先人工处理数据。
- `simulate_failure` 是演示开关，生产部署前应移除或加权限。
- 没有鉴权、限流；审计流水 `journal.jsonl` 只追加不轮转，事件在状态文件里
  保留最近 500 条。
- 没有做真实产线帧率/吞吐压测，只有并发行为与崩溃恢复测试；本环境没有
  Docker 守护进程，Compose 配置已按卷挂载/只读根文件系统/非 root 用户核对，
  但镜像构建与 `docker compose up` 未在本机实测。
