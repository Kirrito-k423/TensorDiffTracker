
# Tensor Tracker  

Track and visualize tensor changes across your code with ease.  

**Tensor Tracker** is a lightweight debugging tool designed to monitor the evolution of target tensors (PyTorch/NumPy) throughout your pipeline. It provides clear, hierarchical logs showing tensor transformations, variable values, and function call traces – helping you pinpoint shape, dtype, or value changes effortlessly.  


## Install

```
pip install .
```

## Quick Start  

### `auto_diff`: Automatic Input/Output Tracing  
Decorate any function with `@auto_diff` to log its inputs, outputs, and execution flow in an intuitive tree format:  

```python  
from ttracker import auto_diff  

@auto_diff  
def process_data(inputs):  
    # ... your tensor operations ...  
    return outputs  
```  

**Sample Output:**  
```bash  
┌── [ttracker.py:176] 函数 [T2VDataset.get_merge_data] 被调用于 getitem:283 (t2v_dataset.py:283 <- thread.py:57 <- thread.py:80)
├─ 输入 [Var] examples@0 len(value): 4
├─ 输入 [Var] index@0 value: 113
│   ├─ [Var] sample@335 len(value): 10
│   │   │   ├─ [Tensor] video@575 → shape=torch.Size([73, 3, 720, 720]) | dtype=torch.uint8
│   ├─ [Var] frame_indice@341 len(value): 73
│   │   │   ├─ [Tensor] video@588 → shape=torch.Size([3, 73, 384, 544]) | dtype=torch.float32
│   │   └─ 输出 shape=torch.Size([3, 73, 384, 544])
│   └─ 输出 shape=torch.Size([3, 73, 384, 544])
```  

### `var_tracker`: Targeted Variable Monitoring  
Use `@var_tracker('target_var')` to automatically track specific variables across function calls. It enables `auto_diff` by default and highlights changes to the target variable:  

```python  
from ttracker import var_tracker  

@var_tracker('video')  
def transform_video(video_tensor):  
    # ... video processing steps ...  
    return transformed_video  
```  

**Sample Output:**  
```bash  
│   │   ├─ 输入 [Var] clip_total_frames@0 value: 222
│   │   ├─ 输入 [Var] start_frame_idx@0 value: 0
│   │   ├─ 输入 [Var] clip_total_frames@0 value: 1770
│   │   │   ├─ [Tensor] video@572 → shape=(73, 640, 960, 3) | dtype=uint8
│   │   │   ├─ [Tensor] video@574 → shape=torch.Size([73, 640, 960, 3]) | dtype=torch.uint8
│   │   │   ├─ [Tensor] video@575 → shape=torch.Size([73, 3, 640, 960]) | dtype=torch.uint8
│   │   │   ├─ [Tensor] video@572 → shape=(73, 720, 960, 3) | dtype=uint8
│   │   │   ├─ [Tensor] video@574 → shape=torch.Size([73, 720, 960, 3]) | dtype=torch.uint8
│   │   │   ├─ [Tensor] video@575 → shape=torch.Size([73, 3, 720, 960]) | dtype=torch.uint8
│   │   │   ├─ [Tensor] video@572 → shape=(73, 720, 960, 3) | dtype=uint8
```

### Log2file

```py
logger.addHandler(h4logger())

log2file(f"xx is {xxx}")
```

## Key Features  
• **Hierarchical Logging:** See function call chains and tensor evolution in a tree structure.  
• **Precision Tracking:** Monitor changes in tensor shape, dtype, device (GPU/CPU), and more.  
• **Targeted Debugging:** Focus on specific variables with `var_tracker` while maintaining context.  
• **Lightweight:** Minimal overhead, designed for interactive debugging during development.  

---

**Debug smarter, not harder** – Tensor Tracker helps you see the unseen in complex tensor pipelines. 🚀

## Example 

### Accuracy monitor

```py
def check_hook(module, input, output):
    from ttracker import _log_change
    _log_change(f"{module.__class__.__name__}", "input", input, "", 0)
    _log_change(f"{module.__class__.__name__}", "output", output, "", 0)
        
print(model_executable)
for name, layer in model_executable.named_modules():
    layer.register_forward_hook(check_hook)
```
