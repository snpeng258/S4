# WSL2 资源分配（建库共享模式）

建库期间 Windows 与 WSL 共用机器时，建议限制 WSL 占用约一半 CPU/内存。

## 1. 编辑 Windows 侧配置

在 **Windows** 用户目录创建或编辑：

`%UserProfile%\.wslconfig`

内容：

```ini
[wsl2]
processors=56
memory=60GB
swap=8GB
localhostForwarding=true
```

说明：

- `processors=56`：WSL 最多使用 56 个逻辑核（宿主机约 112 核时取一半）
- `memory=60GB`：WSL 内存上限；建库 NPZ 峰值约 25GB，足够
- 修改后必须重启 WSL 才生效

## 2. 应用配置

**PowerShell（管理员）**：

```powershell
wsl --shutdown
```

重新打开 WSL 终端。

## 3. 在 WSL 内验证

```bash
nproc          # 期望 56
free -h        # total 约 60Gi
```

## 4. 拉满模式（几乎不用 Windows 时）

将 `processors=112`、`memory=120GB`（按宿主机实际核数/内存调整），同样 `wsl --shutdown` 后生效。

## 5. 建库期间建议

- Windows 上避免 CPU 密集型任务
- 用 `htop` / `pgrep -c S4` 观察并发 S4 进程数
- 建库输出：`../data/inverse/`（npz 谱库）、`../runs/inverse/`（日志与图）
