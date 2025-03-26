import sys
import inspect
from functools import wraps
import torch
import os
import threading
from icecream import ic 
import time
import copy
import numpy as np

import logging
import os
import threading
import inspect
from functools import wraps
import os
import threading
import logging

class ThreadFileHandler(logging.Handler):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    def emit(self, record):
        # 动态生成当前进程和线程的文件名
        pid = os.getpid()
        tid = threading.get_ident()
        filename = os.path.join(self.base_path, f"process_{pid}_thread_{tid}.log")
        # 确保日志目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 写入日志
        with open(filename, 'a', encoding='utf-8') as f:
            msg = self.formatter.format(record)
            f.write(msg + '\n')

def h4logger():
    # 创建日志目录（以主进程启动时间命名）
    if "LTAG" in os.environ:
        LTAG = os.environ["LTAG"]
    else:
        LTAG = "NOTAG"
    log_dir = os.path.join("./logs", LTAG)
    os.makedirs(log_dir, exist_ok=True)

    handler = ThreadFileHandler(log_dir)
    return handler

def setup_logger():
    # 配置日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # 移除已有的Handler，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(h4logger())
    return logger

logger = setup_logger()
logger.info("Main process started.")

def pskip():
    # import torch.distributed as dist
    # return dist.get_rank() == 0
    return True

def log2file(tmp):
    # if "LOG2F" in os.environ:
    logger.info(tmp)
    # else:
    # if pskip():
    print(tmp, flush=True)


# 线程局部存储保证多线程安全
_local = threading.local()

class IndentManager:
    @staticmethod
    def get_indent():
        if not hasattr(_local, 'indent_level'):
            _local.indent_level = 0
        return '│   ' * _local.indent_level  # 使用树状缩进符号

    @staticmethod
    def increase():
        _local.indent_level += 1

    @staticmethod
    def decrease():
        _local.indent_level = max(0, _local.indent_level - 1)

def get_call_stack():
    stack = inspect.stack()
    # 跳过装饰器本身的堆栈帧
    relevant_frames = [
        f"{os.path.basename(frame.filename)}:{frame.lineno}"
        for frame in stack[2:]  # 跳过wrapper和装饰器层
    ]
    return " <- ".join(relevant_frames[:3])  # 显示最近3层调用

def auto_diff(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取类名（如果是类方法）
        class_name = ""
        if '.' in func.__qualname__:  # 类方法会有 MyClass.method 的qualname
            class_name = func.__qualname__.split('.')[0] + '.'
        
        # 获取函数标识
        func_identity = f"{class_name}{func.__name__}"

        # 获取定义位置信息
        def_file = os.path.basename(func.__code__.co_filename)
        def_line = func.__code__.co_firstlineno
        def_info = f"{def_file}:{def_line}"

        # 获取调用位置信息
        caller_frame = inspect.currentframe().f_back
        call_info = f"{caller_frame.f_code.co_name}:{caller_frame.f_lineno}"
        
        # 生成缩进前缀
        indent = IndentManager.get_indent()
        IndentManager.increase()
        
        try:
            # 打印带类名的调用入口
            log2file(f"{indent}┌── [{def_info}] 函数 [{func_identity}] 被调用于 {call_info} ({get_call_stack()})")
            # 参数追踪
            sig = inspect.signature(func)
            bound_params = sig.bind(*args, **kwargs)
            for name, value in bound_params.arguments.items():
                _log_change("├─ 输入", name, value, indent)
            
            # 执行函数
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()


            # 结果追踪
            if isinstance(result, torch.Tensor):
                _log_change("└─ 输出", "output", result, indent)
            elif isinstance(result, (tuple, list)):
                for i, item in enumerate(result):
                    _log_change("└─ 输出", i, item, indent)
            log2file(f"{indent}└─ 耗时 {func_identity} 执行耗时: {end_time - start_time:.4f} 秒")
            return result
        finally:
            IndentManager.decrease()
    return wrapper

def extract_tensor_info(data, max_elements=25):
    """递归提取数据结构的元信息，小张量直接打印数值"""
    # 处理 PyTorch 张量
    if isinstance(data, torch.Tensor):
        if data.numel() <= max_elements:  # 元素总数 <= 25 (5x5)
            # 将张量转为 Python 列表（确保在 CPU 上）
            return data.detach().cpu().tolist()
        else:
            return {
                "type": "Tensor",
                "shape": tuple(data.shape),
                "dtype": str(data.dtype),
                "sum": data.sum(),
                "mean": data.mean(),
                "min": data.min(),
                "max": data.max()
            }
    
    # 处理 NumPy 数组
    if isinstance(data, np.ndarray):
        if data.size <= max_elements:
            return data.tolist()
        else:
            return {
                "type": "ndarray",
                "shape": data.shape,
                "dtype": str(data.dtype)
            }
    
    # 处理列表、元组（递归处理每个元素）
    if isinstance(data, (list, tuple)):
        return [extract_tensor_info(item, max_elements) for item in data]
    
    # 处理字典（递归处理每个键值对）
    if isinstance(data, dict):
        return {
            key: extract_tensor_info(value, max_elements)
            for key, value in data.items()
        }
    
    # 处理集合（转换为列表后递归处理）
    if isinstance(data, set):
        return [extract_tensor_info(item, max_elements) for item in data]
    
    # 其他类型直接返回
    return data

# prestr ├─
def _log_change(prestr, name, value, indent, lineno=0):
    """类型感知的日志输出"""
    prefix_str=f"{indent}{prestr} [Var] {name}@{lineno} "
    if isinstance(value, (torch.Tensor, np.ndarray)):
        info = f"\t shape={value.shape} | dtype={value.dtype}  | sum={value.sum()}  | mean={value.mean()}  | min={value.min()}  | max={value.max()}"
        log2file(f"{indent}{prestr} [Tensor] {name}@{lineno} → {info}")
    elif isinstance(value, (list, dict, set)):
        # print(f"{indent}{prestr} [Var] {name}@{lineno} → {len(value)}")
        # 生成包含张量元信息的结构
        simplified = extract_tensor_info(value)
        # 使用 ic 打印结构化数据
        # if pskip():
            # ic.configureOutput(prefix=prefix_str)
            # ic(simplified)  # 显示嵌套结构的元信息
            # ic.configureOutput(prefix="   ic: ")
        log2file(prefix_str+str(simplified))
    elif isinstance(value, (int, float, str)):
        # print(f"{indent}{prestr} [Var] {name}@{lineno} → {value}")
        # if pskip():
            # ic.configureOutput(prefix=prefix_str)
            # ic(value)  # 使用icecream打印结构化数据
            # ic.configureOutput(prefix=f"   ic: ")
        log2file(prefix_str+str(value))

def is_content_changed(last, current):
    # 类型不同直接认为变化
    if type(last) != type(current):
        return True

    # 处理张量（以 PyTorch 为例）
    if isinstance(last, torch.Tensor):
        return not torch.all(torch.eq(last, current)).item()

    # 处理列表、元组等可迭代对象（递归比较元素）
    if isinstance(last, (list, tuple, tuple)):
        if len(last) != len(current):
            return True
        return any(is_content_changed(l, c) for l, c in zip(last, current))

    # 处理字典（递归比较键和值）
    if isinstance(last, dict):
        if last.keys() != current.keys():
            return True
        return any(is_content_changed(last[k], current[k]) for k in last)

    # 处理集合（转换为有序结构后比较）
    if isinstance(last, set):
        return set(sorted(last)) != set(sorted(current))

    # 其他类型（如基本类型、字符串等）
    return last != current

def var_tracker(*track_vars):
    """支持同时追踪多个变量的装饰器工厂"""
    class Tracer:
        def __init__(self, func_code):
            self.target_code = func_code
            self.track_list = track_vars  # 要追踪的变量名列表
            self.snapshots = {}  # 保存变量历史值
            self.in_target = False  # 是否在目标函数作用域内

        def trace_changes(self, frame, event, arg):
            # 生成缩进前缀
            indent = IndentManager.get_indent()
            if event == 'call':
                if frame.f_code == self.target_code:
                    self.in_target = True
            elif event == 'return':
                self.in_target = False
            if frame.f_code == self.target_code:
                self.in_target = True
            if event == 'line' and self.in_target:
                current_locals = frame.f_locals
                for var in self.track_list:
                    if var in current_locals:
                        current_val = current_locals[var]
                        if self._is_modified(var, current_val):
                            _log_change("├─", var, current_val, indent, frame.f_lineno)
            return self.trace_changes

        def _is_modified(self, var, current):
            """智能类型感知对比方法"""
            # 获取历史记录
            history = self.snapshots.get(var, [])
            last = history[-1] if history else None
            
            # 类型检查优先
            current_type = type(current)
            if last is None:
                self.snapshots[var] = [self._create_snapshot(current)]
                return True
            elif not isinstance(last, current_type):
                # self.snapshots[var].append(self._create_snapshot(current))
                self.snapshots[var] = [self._create_snapshot(current)]
                return True
            # 类型相同则具体对比
            elif self._compare_by_type(last, current):
                # self.snapshots[var].append(self._create_snapshot(current))
                self.snapshots[var] = [self._create_snapshot(current)]
                return True
            return False

        def _create_snapshot(self, value):
            """创建安全快照（防止引用变化）"""
            return copy.deepcopy(value) if not isinstance(value, torch.Tensor) else value.clone()

        def _compare_by_type(self, last, current):
            """类型敏感对比逻辑"""
            if isinstance(current, (torch.Tensor, np.ndarray)):
                # 张量对比逻辑
                # 形状检查优先
                if last.shape != current.shape:
                    return True
                
                # dtype兼容处理
                if last.dtype != current.dtype:
                    # 记录类型差异日志
                    log2file(f"[WARN] 数据类型变化 {last.dtype} → {current.dtype}，自动转换后比较")
                    return True
                
                # 数值比较（保留原始精度）
                return not torch.allclose(last, current, atol=1e-5)
            elif isinstance(current, (list, dict, set, tuple)):
                # 复杂结构深对比
                return is_content_changed(copy.deepcopy(last), copy.deepcopy(current))
            else:
                # 基础类型直接对比
                # ic(last)
                # ic(current)
                # ic(type(current))
                # ic(type(last))
                return last != current


    def decorator(func):
        @auto_diff
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = Tracer(func.__code__)
            sys.settrace(tracer.trace_changes)
            result = func(*args,**kwargs)
            sys.settrace(None)
            return result
        return wrapper
    return decorator

import sys
import inspect
from functools import wraps

def safe_getsourcelines(code):
    """带异常处理的源码获取函数"""
    try:
        return inspect.getsourcelines(code)
    except (OSError, TypeError):
        return None, 0

def trace_lines(frame, event, arg):
    indent = IndentManager.get_indent()
    if event == 'line':
        code = frame.f_code
        lines, first_line = safe_getsourcelines(code)
        
        if lines and frame.f_lineno - first_line < len(lines):
            line_content = lines[frame.f_lineno - first_line].strip()
            log2file(f"{indent}├── [行 {frame.f_lineno}] {line_content}")

        else:
            line_content = "[动态生成代码]"
            
    return trace_lines


def trace_calls(frame, event, arg):
    indent = IndentManager.get_indent()
    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name != '<module>':
            log2file(f"{indent}┌── 进入函数: {func_name}()")
            IndentManager.increase()
            sys.settrace(trace_lines)
    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name != '<module>':
            IndentManager.decrease()
            log2file(f"{indent}└── 退出函数: {func_name}()")
    return trace_calls

def debug_trace(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        log2file(f"【debug_trace for {func.__name__}】")
        sys.settrace(trace_calls)
        result = func(*args, **kwargs)
        sys.settrace(None)
        return result
    return wrapper

def fstack(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stack = inspect.stack()
        log2file(f"【Call Stack for {func.__name__}】")
        for frame in reversed(stack[1:]):  # 排除装饰器自身调用层
            log2file(f"-> {frame.function}() at {frame.filename}:{frame.lineno}")
        return func(*args,**kwargs)
    return wrapper
# 使用示例：同时监控变量A和B的变化
# @var_tracker('A', 'B')
# @auto_diff
# def test_func(x):
#     A = x * 2
#     B = A.sum()
#     A = A.reshape(-1, 4)
#     return A

# test_func(torch.randn(1,4,2))
