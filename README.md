
### A Python Local Dependency Analysis Tool  
**Features**:  

- 🔍 **Precise Parsing**: Analyzes only `import` statements  
- ♻️ **Circular Reference Compatibility**: Supports projects with circular references (common in real-world development)  
- 💾 **Backup Archive**: Preserved here to prevent code loss  

> *"If anyone requires this project name, please notify me—I will rename it immediately."*  

---

### Key Specifications:  
1. Designed specifically for **local dependency analysis** in Python projects.  
2. Created because **I needed an analysis tool** when downloading projects and not knowing where to start reading.  
3. Shows:  
   - Python files imported by File A  
   - File hierarchy levels (starting from 0) after comprehensive evaluation  
   - **Lower levels indicate more foundational modules**  
4. Handles circular references by:  
   - Treating circularly-referenced files as a single **macro-node** (strongly connected component)  
   - Allowing internal circular references within macro-nodes  
   - Ensuring **no circular dependencies exist between macro-nodes** for level analysis  
5. **Graph algorithm implementation**: AI-generated.  
6. **Usage**: Place this file in your project's root directory and execute it.  

---
### 一个 Python 本地依赖分析工具
功能特性:

- 🔍 精准解析：仅分析 import 语句
- ♻️ 循环引用兼容：支持存在循环引用的项目（现实开发中普遍存在）
- 💾 备份存档：为防止代码丢失特此保存
>*"若有人需要此项目名称，请告知我——我将立即重命名。"*  

### 核心说明：
1. 专为 Python 项目的本地依赖分析设计。
2. 设计这个是因为我下项目后，不懂从哪里开始看，想要个分析工具
3. 能说明a文件import的python文件，和总体评估后，给各文件的分级，从0开始，等级越低越基础。
4. 能处理循环引用问题，思想就是：假设存在循环引用，那么将存在循环引用的项目视作一个大的结点（强连通图），大结点内各文件可以循环引用，大结点间不存在循环引用，所以可以用来分析等级。
5. 图论算法是ai写的。
6. 使用方法：把这个文件扔到要分析的项目的根目录，然后运行就行

---
