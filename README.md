# Tensor tracker

trace the diff or change of target tensor

## Quick  Start


### auto_diff
```py
@auto_diff
def target(xxx):
  xxx
```

`auto_diff` will print the input and output of target func in elegent way such like:

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

### var_tracker

var_tracker will turn on auto_diff in default and tracker the target vars change

```py
@var_tracker('video')
def target(xxx):
  xxx
```

log like, show the var  `video` changed in line 572、574、575 :

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
