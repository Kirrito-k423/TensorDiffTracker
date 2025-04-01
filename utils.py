
import torch
import torch_npu, torchair

import threading

# 定义线程本地存储
_thread_local = threading.local()

def inlog(tmpstr, tmpt, interval=1, save2file=False):
    # 初始化线程本地计数器
    if not hasattr(_thread_local, 'counter'):
        _thread_local.counter = 0
    
    # 更新计数器
    _thread_local.counter += 1
    
    # 达到间隔次数时打印
    if _thread_local.counter % interval == 0:
        if VLLM_ENABLE_GRAPH_MODE == '1':
            torchair.ops.npu_print(f"{tmpstr} ", tmpt)
        else:
            result =f"{tmpstr} {tmpt}"
            print(result)
            if save2file:
                cur_rank = torch.distributed.get_rank()
                os.makedirs("logs/ranklogs", exist_ok=True)
                with open(f"logs/ranklogs/rank_{cur_rank}.log", "a", encoding="utf-8") as file:
                    file.write(f"{result} \n")

def safe_mean(tensor):
    if tensor.dtype not in (torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128):
        tensor = tensor.float()  # 转换为默认的浮点类型
    return tensor.mean()

def dlog(tmpstr, tmpt, record_all=False, interval=1 ):
    cur_rank = torch.distributed.get_rank()
    if cur_rank == 0 or record_all :
        if tmpt is None:
            h2_mean=torch.tensor(1)
        elif isinstance(tmpt, (int, float)):
            h2_mean=torch.tensor(tmpt)
        elif isinstance(tmpt, torch.Tensor):
            h2_mean=safe_mean(tmpt)
            inlog(f"{tmpstr} rank{cur_rank} {str(tmpt.shape)}", tmpt.sum(),interval, record_all)
        else:
            h2_mean=tmpt
        inlog(f"{tmpstr} rank{cur_rank}", h2_mean,interval, record_all)

def dump_data(tmpstr, ttensor):
    # 初始化线程本地计数器
    if not hasattr(_thread_local, 'dump_counter'):
        _thread_local.dump_counter = 0
    
    # 更新计数器
    _thread_local.dump_counter += 1

    cur_rank = torch.distributed.get_rank()
    os.makedirs("dump_data", exist_ok=True)
    torch.save(ttensor, f"dump_data/tensor_{tmpstr}_{_thread_local.dump_counter}_{cur_rank}.pt")
