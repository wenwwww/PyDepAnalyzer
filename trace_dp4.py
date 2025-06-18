# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path as path
import ast


@dataclass(frozen=True)
class DirNode:
    root:str|path=None
    local_py:dict=None
    def __post_init__(self):
        if self.root is None:
            root = path(__file__).resolve().parent
        elif isinstance(self.root, str):
            root = path(self.root).resolve()
        else:
            root = self.root.resolve()

        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'directory',{})
        if self.local_py is None:
            object.__setattr__(self, 'local_py',{})
        
        for item in self.root.iterdir():
            item_path = str(item)
            if item.is_dir():
                dir = DirNode(item,self.local_py)
                self.directory[item_path]=ItemInfo(Info=dir)
            elif item.suffix == '.py':
                Item = ItemInfo()
                self.directory[item_path] = Item
                self.local_py[item_path] = Item
                
#统一使用绝对路径 / Use absolute paths uniformly
    def items(self):
        return self.directory.items()
    
    def __getitem__(self, item):
        return self.directory[item] if item in self.directory else None
    
    def __len__(self):
        return len(self.directory)
    
    def __contains__(self, item):
        return item in self.directory

@dataclass
class ItemInfo:
    Info: 'DirNode|None'= None  # 子文件或目录 / Child file or directory
    Level:int =-1  # 级别，初始为-1，后续会根据依赖关系更新 / Level, initialized to -1, will be updated based on dependencies
    __match__args__ = ('Info', 'Level') 
    def __str__(self):
        return f"ItemInfo(Info={self.Info}, level={self.Level})"
    


    

class LocalDependencyAnalyzer:
    subdir_prefix = '├───'
    last_subdir_prefix = '└───'
    dep_prefix = '│   '
    last_dep_prefix = '    '
    import_module = '└──>>> '
    
    def __init__(self, root: path | str=path(__file__).resolve().parent):
        if isinstance(root, str):
            root = path(root)
        self.root = root.resolve()
        self.dir=DirNode(self.root)  # Initialize the directory structure
        self.need_imports: dict[str, set[str]] = {}
        self.provide_imports: dict[str, set[str]] = {}
        
    @property
    def local_py(self) -> dict[str, ItemInfo]:
        return self.dir.local_py  
    
    @property
    def local_dict(self) -> dict[str, ItemInfo]:
        return self.dir.directory

    
    def find_import_file(self, import_path: path,) -> str | None:# flag=False) -> str | None:
        import_py = import_path.with_suffix('.py')
        flag_py = import_py.exists()
        flag_dir = import_path.exists()
        
        while (not (flag_py | flag_dir)) and str(import_path) != str(self.root):
            import_path = import_path.parent
            import_py = import_path.with_suffix('.py')
            flag_py = import_py.exists()
            flag_dir = import_path.exists()
           
        
        if flag_dir:  # 如果是目录，则添加目录下的__init__.py文件 / If it's a directory, add the __init__.py file from that directory
            if import_path == self.root:  # 如果是根目录，则返回None，默认为外部库 / If it's the root directory, return None, assuming it's an external library
                return None
            init_file = import_path/'__init__.py'
            if init_file.exists():
                return str(init_file)
        elif flag_py:  # 如果是.py文件，则添加到导入列表 / If it's a .py file, add it to the import list
            return str(import_py)        
        else:
            return None
            
    def get_imports(self,file:str)->set[str]:
    #file是一个.py文件的绝对路径 / file is the absolute path of a .py file
        file= path(file)
        dir_path=file.parents
        imports = set()
        with open(file, "r",encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:#node.names是所有导入的模块别名列表 / node.names is a list of all imported module aliases
                        module_path_parts = alias.name.split('.')
                        import_path = self.root.joinpath(*module_path_parts)
                        import_path = self.find_import_file(import_path)
                        if import_path is not None:
                            imports.add(import_path)
                            
                elif isinstance(node, ast.ImportFrom):
                    current_dir=None
                    if node.level==0:
                        current_dir=self.root
                    else:
                        current_dir=dir_path[node.level-1]
                
                    if node.module is None:
                        for alias in node.names:
                            module_path_parts = alias.name.split('.')
                            import_path = current_dir.joinpath(*module_path_parts)
                            import_path = self.find_import_file(import_path)
                            if import_path is not None:
                                imports.add(import_path)  
                                
                    else: #node.module is not None:
                        module_path_parts1 = node.module.split('.')
                        for alias in node.names:
                            module_path_parts2 = alias.name.split('.')
                            import_path = current_dir.joinpath(*module_path_parts1, *module_path_parts2)
                            import_path = self.find_import_file(import_path)
                            if import_path is not None:
                                imports.add(import_path)
                        
        return imports
    
    def current_get_imports(self, py_path: str) -> set[str]:
        imports = self.get_imports(py_path)
        return py_path, imports
    
        
    def analyse_local_imports(self) -> dict:
        py_paths = list(self.local_py.keys())
        with ProcessPoolExecutor(max_workers=6) as executor:
            results = executor.map(self.current_get_imports, py_paths)
            for py_path, imports in results:
                if len(imports) == 0: #只依赖外部库 / Only depends on external libraries
                    continue 
                
                # 模块依赖，这里因为每个py文件作为一个key独一无二，所以可以字典直接update，不用担心key重复而导致value覆盖问题
                # Module dependency. Since each .py file is a unique key, we can directly update the dictionary without worrying about overwriting values due to duplicate keys.
                self.need_imports[py_path] = imports
                
                for import_item in imports:
                # 因为provide_imports中的key在别处也可能出现，所以这里要进入循环
                # Since the same file can be imported by multiple other files, we add the dependent file to a set for that key.
                    self.provide_imports.setdefault(import_item, set()).add(py_path)
                    
        return self.need_imports, self.provide_imports




    def tarjan_scc(self) -> list[set[str]]:
        """
        使用 Tarjan 算法找出图中所有的强连通分量 (SCCs)。
        此算法更高效，但理解起来稍复杂。
        
        Finds all strongly connected components (SCCs) in a graph using Tarjan's algorithm.
        This algorithm is more efficient, but slightly more complex to understand.
        """
        graph = self.need_imports
        nodes = list(self.local_py.keys())

        # 初始化 / Initialization
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        ids = [-1] * n
        low_links = [-1] * n
        on_stack = [False] * n
        stack = []
        time = 0
        sccs = []

        def dfs(at_idx):
            nonlocal time
            stack.append(at_idx)
            on_stack[at_idx] = True
            ids[at_idx] = low_links[at_idx] = time
            time += 1

            at_node = nodes[at_idx]
            for to_node in graph.get(at_node, set()):
                if to_node not in node_to_idx:
                    continue  # 邻居可能不在分析范围内 / The neighbor might not be within the analysis scope
                to_idx = node_to_idx[to_node]

                if ids[to_idx] == -1:  # 如果邻居未被访问 / If the neighbor has not been visited
                    dfs(to_idx)
                    low_links[at_idx] = min(
                        low_links[at_idx], low_links[to_idx])
                elif on_stack[to_idx]:  # 如果邻居在栈上，说明是回边 / If the neighbor is on the stack, it's a back edge
                    low_links[at_idx] = min(low_links[at_idx], ids[to_idx])

            # 如果当前节点是SCC的根 / If the current node is the root of an SCC
            if ids[at_idx] == low_links[at_idx]:
                current_scc = set()
                while stack:
                    node_idx = stack.pop()
                    on_stack[node_idx] = False
                    current_scc.add(nodes[node_idx])
                    if node_idx == at_idx:
                        break
                sccs.append(current_scc)

        for i in range(n):
            if ids[i] == -1:
                dfs(i)

        return sccs
        
    def analyze_scc_and_level(self):
        """
        新的主分析方法：找出SCCs("团")并为它们分级。(修正版)
        New main analysis method: Find SCCs ("cliques") and assign levels to them. (Revised version)
        """
        # 步骤 1: 构建基础依赖图 / Step 1: Build the basic dependency graph
        self.analyse_local_imports()

        # 步骤 2: 使用 Tarjan 算法找出所有“团” / Step 2: Use Tarjan's algorithm to find all "cliques" (SCCs)
        sccs = self.tarjan_scc()
        
        # 创建一个从 文件 -> "团"ID 的映射 / Create a mapping from file -> SCC ID
        file_to_scc_id = {}
        for i, scc in enumerate(sccs):
            for file_node in scc:
                file_to_scc_id[file_node] = i
        
        # 步骤 3: 构建“团”间的【反向】关系图 ("谁依赖我" 图) / Step 3: Build the inverted graph between SCCs (a "who depends on me" graph)
        num_sccs = len(sccs)
        # scc_dependents_graph[i] 存储的是所有依赖于 SCC i 的 SCC 集合 / scc_dependents_graph[i] stores the set of all SCCs that depend on SCC i
        scc_dependents_graph = {i: set() for i in range(num_sccs)}
        scc_in_degree = {i: 0 for i in range(num_sccs)}

        for u, deps in self.need_imports.items():
            u_scc_id = file_to_scc_id.get(u)
            if u_scc_id is None: continue
            
            for v in deps:
                v_scc_id = file_to_scc_id.get(v)
                if v_scc_id is None or u_scc_id == v_scc_id:
                    continue
                
                # 原始依赖是 u -> v (u 依赖 v) / The original dependency is u -> v (u depends on v)
                # 这意味着 v_scc_id 是 u_scc_id 的一个前置依赖。 / This means v_scc_id is a prerequisite for u_scc_id.
                # 【修正点】我们要在反向图中添加一条 v_scc_id -> u_scc_id 的边 / [Correction] We add an edge v_scc_id -> u_scc_id in the inverted graph
                # 表示 u_scc_id 是 v_scc_id 的一个后继（依赖者） / This indicates that u_scc_id is a successor (dependent) of v_scc_id
                if u_scc_id not in scc_dependents_graph[v_scc_id]:
                    scc_dependents_graph[v_scc_id].add(u_scc_id)
                    # 同时，u_scc_id 的入度加一 / At the same time, the in-degree of u_scc_id is incremented
                    scc_in_degree[u_scc_id] += 1
        
        # 步骤 4: 对“团”进行拓扑排序分级 (现在逻辑是正确的) / Step 4: Perform topological sorting and leveling on the SCCs (the logic is now correct)
        scc_levels = [-1] * num_sccs
        queue = [i for i, degree in scc_in_degree.items() if degree == 0]
        level = 0
        
        while queue:
            next_queue = []
            for scc_id in queue:
                scc_levels[scc_id] = level
                
                # 【修正点】现在我们遍历的是“依赖我的邻居” / [Correction] Now we are iterating through the "neighbors that depend on me"
                # 当我处理完后，这些邻居的一个前置依赖就完成了 / After I am processed, one of the prerequisites for these neighbors is fulfilled
                for dependent_scc_id in scc_dependents_graph.get(scc_id, set()):
                    scc_in_degree[dependent_scc_id] -= 1
                    if scc_in_degree[dependent_scc_id] == 0:
                        next_queue.append(dependent_scc_id)
            queue = next_queue
            level += 1

        # 步骤 5: 将计算出的等级赋给每个文件 / Step 5: Assign the calculated levels to each file
        for i, scc in enumerate(sccs):
            scc_level = scc_levels[i]
            # 如果一个SCC因为在环中且不被外部依赖，可能没有被正确分级 / If an SCC is in a cycle and not depended on by anything external, it might not have been correctly leveled
            if scc_level == -1:
                # 可以给一个特殊值，或者根据需求处理 / We can assign a special value or handle it as needed
                # 这里暂时也设为0级，表示它们是自包含的顶级环 / Here, we temporarily set it to level 0, indicating they are self-contained top-level cycles
                scc_level = 0
            
            for file_node in scc:
                # 确保 self.local_py 中有这个文件条目 / Ensure this file entry exists in self.local_py
                if file_node in self.local_py:
                    self.local_py[file_node].Level = scc_level    
        
    
    
    
    def display(self):
       
        self.trace_local_dependency(self.local_dict, prefix="")

    def trace_local_dependency(self,dir:DirNode,prefix: str = "",):
        
        # Define the characters for the tree structure
        items = list(dir.items())
        for i, (item_full_path, item_info) in enumerate(items):
            subtree, level = item_info.Info, item_info.Level
            is_current_item_last = (i == len(items) - 1)
            connector = self.last_subdir_prefix if is_current_item_last else self.subdir_prefix

            # Ensure the path is absolute
            path_item = path(item_full_path)
            new_prefix = prefix + \
                (self.last_dep_prefix if is_current_item_last else self.dep_prefix)  # 提前计算下一行 / Pre-calculate for the next line

            '''这个是输出各文件的依赖 / This part outputs the dependencies of each file
            if subtree is None:  # It's a Python file
                # 输出为一行了 / Print on one line
                print(f"{prefix}{connector}{path_item.name}:{level}")
                dependencies = self.need_imports.get(item_full_path)  # 获取该模块的依赖 / Get the dependencies of this module
                if dependencies:
                    for dep in dependencies:
                        dep = path(dep).resolve()  # 确保依赖是绝对路径 / Ensure the dependency path is absolute
                        
                    
                # Show dependency path relative to the project root for readability
                    if str(dep.parent) == str(path_item.parent):
                    # If it's in the same directory, just show the file name
                        relative_dep = str(dep.name)
                    else:
                        relative_dep = dep.relative_to(self.root)  # 获取相对根目录的相对路径 / Get the path relative to the root
                
                        print(f"{new_prefix}{self.import_module}{relative_dep}:{self.local_py[str(dep)].Level}")  # 打印依赖 / Print dependency
            
            else:  # It's a directory
                # Recurse into the subdirectory
                print(f"{prefix}{connector}{path_item.name}")
                self.trace_local_dependency(dir[item_full_path].Info,new_prefix)
            '''
            if subtree is None:
                print(f"{prefix}{connector}{path_item.name}:{level}")
            else:
                print(f"{prefix}{connector}{path_item.name}")
                self.trace_local_dependency(dir[item_full_path].Info,new_prefix)

    
    




if __name__ == "__main__":
    directory_path=path(__file__).resolve().parent  # 默认当前目录 / Default to the current directory
    analyzer = LocalDependencyAnalyzer(directory_path)
    analyzer.analyze_scc_and_level()
    analyzer.display()