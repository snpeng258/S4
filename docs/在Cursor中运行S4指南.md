# 在Cursor中运行S4代码指南

## 当前状态

根据检查，您的Windows系统上**目前还没有安装**编译S4所需的工具：
- ❌ GCC编译器（gcc/g++）
- ❌ Lua解释器

## 运行S4的三种方式

### 方式1：直接运行已编译的二进制文件（最简单）⭐

如果您有已编译好的S4可执行文件，可以直接运行Lua脚本：

```bash
# 假设S4.exe在项目根目录
.\S4.exe examples\simple\simple.lua
```

**优点**：无需编译，立即可用  
**缺点**：需要先获得编译好的二进制文件

### 方式2：在Cursor中编译S4（推荐用于开发）

#### 步骤1：安装MSYS2（Windows上的Unix环境）

1. 下载并安装 [MSYS2](https://www.msys2.org/)
2. 打开MSYS2终端，安装必要的工具：

```bash
# 更新包管理器
pacman -Syu

# 安装编译工具
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-g++
pacman -S mingw-w64-x86_64-make

# 安装Lua
pacman -S mingw-w64-x86_64-lua

# 安装数学库（推荐）
pacman -S mingw-w64-x86_64-openblas
pacman -S mingw-w64-x86_64-lapack
```

#### 步骤2：配置环境变量

在Windows PowerShell中，将MSYS2的bin目录添加到PATH：

```powershell
# 临时添加（当前会话有效）
$env:Path += ";C:\msys64\mingw64\bin"

# 或者永久添加（需要管理员权限）
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\msys64\mingw64\bin", "User")
```

#### 步骤3：编译S4

在Cursor的终端中（或MSYS2终端）：

```bash
# 进入项目目录
cd C:\Users\PSN\Desktop\s4-github\S4

# 编译源码在 upstream/（原版 S4）
cd upstream
make

# 编译完成后，可执行文件在 upstream/build/S4.exe
# 示例脚本也在 upstream/examples/
```

#### 步骤4：运行示例

```bash
# 运行简单示例
.\build\S4.exe examples\simple\simple.lua

# 或者进入交互模式
.\build\S4.exe
```

### 方式3：使用WSL（Windows Subsystem for Linux）

如果您安装了WSL，可以在Linux环境中编译和运行：

```bash
# 在WSL中安装依赖
sudo apt-get update
sudo apt-get install -y lua5.2 liblua5.2-dev gcc g++ make liblapack-dev libblas-dev

# 编译
cd upstream
make

# 运行
./build/S4 examples/simple/simple.lua
```

## 快速测试（无需编译）

即使没有编译S4，您也可以：

### 1. 查看和理解Lua脚本

所有示例都在 `examples/` 目录下，可以直接查看：

```bash
# 查看简单示例
cat examples\simple\simple.lua

# 查看其他示例
dir examples\1d
dir examples\2d
```

### 2. 使用Python接口（如果已编译Python扩展）

如果有Python扩展，可以直接在Python中使用：

```python
import S4

# 创建仿真
S = S4.NewSimulation()
# ... 使用S4 API
```

## 在Cursor中运行Lua脚本的步骤

### 如果已编译S4：

1. **打开终端**：在Cursor中按 `` Ctrl+` `` 打开终端

2. **运行脚本**：
```powershell
# 进入项目目录
cd C:\Users\PSN\Desktop\s4-github\S4

# 运行示例（假设S4.exe在build目录）
.\build\S4.exe examples\simple\simple.lua > output.txt

# 查看输出
cat output.txt
```

3. **交互式运行**：
```powershell
.\build\S4.exe
# 然后可以输入Lua命令
```

### 如果没有编译S4：

您可以：
1. **阅读和理解代码**：所有源代码都可以直接查看
2. **修改Lua脚本**：可以编辑示例脚本，等有编译环境后再运行
3. **学习算法**：研究RCWA和FMM的实现

## 推荐的开发流程

### 对于代码理解：
✅ **可以直接进行** - 所有源代码都可以在Cursor中查看和编辑

### 对于运行测试：
1. **选项A**：安装MSYS2并编译（适合长期开发）
2. **选项B**：使用WSL（如果您熟悉Linux）
3. **选项C**：获取预编译的二进制文件

## 检查编译环境

运行以下命令检查是否已准备好编译环境：

```powershell
# 检查GCC
gcc --version

# 检查Lua
lua -v

# 检查Make
make --version
```

如果这些命令都能正常执行，就可以开始编译了！

## 常见问题

### Q: 为什么需要编译？
A: S4是C/C++程序，需要编译成可执行文件才能运行。

### Q: 可以直接运行Lua脚本吗？
A: 不可以。Lua脚本需要通过S4程序来执行，S4提供了Lua API。

### Q: 能否在Cursor中直接运行？
A: 可以！只要：
1. 有编译好的S4.exe，或
2. 安装了编译工具并编译成功

### Q: 编译很复杂吗？
A: 在Windows上需要MSYS2环境，但安装后编译过程很简单（`make`命令）。

## 下一步

1. **如果您想立即运行**：安装MSYS2并按照方式2操作
2. **如果您想先理解代码**：直接查看源代码和示例脚本
3. **如果您有已编译版本**：直接运行 `.\S4.exe examples\simple\simple.lua`

需要我帮您安装编译环境或检查其他配置吗？
