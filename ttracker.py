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
            print(f"{indent}┌── [{def_info}] 函数 [{func_identity}] 被调用于 {call_info} ({get_call_stack()})")
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
                print(f"{indent}└─ 输出 shape={result.shape}")
            elif isinstance(result, (tuple, list)):
                for i, item in enumerate(result):
                    _log_change("└─ 输出", i, item, indent)
            print(f"{indent}└─ 耗时 {func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
            return result
        finally:
            IndentManager.decrease()
    return wrapper

# prestr ├─
def _log_change(prestr, name, value, indent, lineno=0):
    """类型感知的日志输出"""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        info = f"shape={value.shape} | dtype={value.dtype}"
        print(f"{indent}{prestr} [Tensor] {name}@{lineno} → {info}")
    elif isinstance(value, (list, dict, set)):
        ic.configureOutput(prefix=f"{indent}{prestr} [Var] {name}@{lineno} ")
        ic(len(value))  # 使用icecream打印结构化数据
        ic.configureOutput(prefix=f"   ic: ")
    elif isinstance(value, (int, float, str)):
        ic.configureOutput(prefix=f"{indent}{prestr} [Var] {name}@{lineno} ")
        ic(value)  # 使用icecream打印结构化数据
        ic.configureOutput(prefix=f"   ic: ")

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
            # 类型相同则具体对比
            elif not isinstance(last, current_type) or self._compare_by_type(last, current):
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
                    print(f"[WARN] 数据类型变化 {last.dtype} → {current.dtype}，自动转换后比较")
                    return True
                
                # 数值比较（保留原始精度）
                return not torch.allclose(last, current, atol=1e-5)
            elif isinstance(current, (list, dict, set)):
                # 复杂结构深对比
                return copy.deepcopy(last) != copy.deepcopy(current)
            else:
                # 基础类型直接对比
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

# 使用示例：同时监控变量A和B的变化
# @var_tracker('A', 'B')
# @auto_diff
# def test_func(x):
#     A = x * 2
#     B = A.sum()
#     A = A.reshape(-1, 4)
#     return A

# test_func(torch.randn(1,4,2))
