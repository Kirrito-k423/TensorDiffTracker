import sys
import inspect
from functools import wraps
import torch
import os

def auto_diff(func):
    # 获取函数定义位置信息
    def_info = f"{os.path.basename(func.__code__.co_filename)}:{func.__code__.co_firstlineno}"
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取调用位置信息
        caller_frame = inspect.currentframe().f_back
        call_info = f"{os.path.basename(caller_frame.f_code.co_filename)}:{caller_frame.f_lineno}"
        
        # 打印函数标识
        print(f"[{def_info}] 函数 {func.__name__} 被调用 (触发于 {call_info})")
        
        # 参数追踪
        sig = inspect.signature(func)
        bound_params = sig.bind(*args, **kwargs)
        for name, value in bound_params.arguments.items():
            if isinstance(value, torch.Tensor):
                print(f"  ↪ 输入参数 [{name}] shape: {value.shape} | dtype: {value.dtype}")

        # 执行原函数
        result = func(*args, **kwargs)
        
        # 结果追踪
        if isinstance(result, torch.Tensor):
            print(f"  ↦ 输出结果 shape: {result.shape} | dtype: {result.dtype}")
        elif isinstance(result, (tuple, list)):
            for i, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    print(f"  ↦ 输出项[{i}] shape: {item.shape} | dtype: {item.dtype}")
        return result
    return wrapper

def var_tracker(*track_vars):
    """支持同时追踪多个变量的装饰器工厂"""
    class Tracer:
        def __init__(self, track_list):
            self.track_list = track_list  # 要追踪的变量名列表
            self.snapshots = {}  # 保存变量历史值

        def trace_changes(self, frame, event, arg):
            if event == 'line':
                current_locals = frame.f_locals
                for var in self.track_list:
                    if var in current_locals:
                        current_val = current_locals[var]
                        if self._is_modified(var, current_val):
                            self.snapshots[var].append(current_val.clone())
                            print(f"    [追踪] {var} 在行 {frame.f_lineno} 变化为 {current_val.shape}")
            return self.trace_changes

        def _is_modified(self, var, current):
            if var not in self.snapshots:
                self.snapshots[var] = []
                return True  # 首次记录
            last = self.snapshots[var][-1] if self.snapshots[var] else None
            if last.shape != current.shape:  # 先检查形状变化
                return True
            return not torch.allclose(last, current, atol=1e-5) if last is not None else True

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = Tracer(track_vars)
            sys.settrace(tracer.trace_changes)
            result = func(*args,**kwargs)
            sys.settrace(None)
            return result
        return wrapper
    return decorator

# 使用示例：同时监控变量A和B的变化
# @auto_diff
# @var_tracker('A', 'B')
# def test_func(x):
#     A = x * 2
#     B = A.sum()
#     A = A.reshape(-1, 4)
#     return A

# test_func(torch.randn(1,4,2))
