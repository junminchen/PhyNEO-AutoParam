import torch
import time
import sys

def occupy_gpu_memory(buffer_mb=1024):
    """
    占满所有可用显存，每个 GPU 保留 buffer_mb 左右的空余。
    """
    if not torch.cuda.is_available():
        print("未检测到 GPU")
        return

    device_count = torch.cuda.device_count()
    tensors = []

    print(f"检测到 {device_count} 个 GPU，正在进行显存锁定...")

    for i in range(device_count):
        # 获取当前显存状态
        total_memory = torch.cuda.get_device_properties(i).total_memory
        
        # 为了更准确地填满，我们先清空当前进程可能的缓存
        torch.cuda.empty_cache()
        
        # 获取当前已分配显存（由其他进程占用）
        # 注意：PyTorch 只能看到自己进程的，所以我们需要参考总剩余
        # 这里采用尝试性分配策略
        
        print(f"\n[GPU {i}] 正在计算可分配空间...")
        
        try:
            # 尝试分配一个极小的 tensor 来初始化上下文
            torch.zeros((1,), device=f'cuda:{i}')
            
            # 剩余显存 = 总量 - 其它进程占用 - 预留缓冲区
            # 由于无法直接通过 torch 获取其他进程精确占用，我们采用循环尝试法
            
            # 获取当前 free 显存 (Bytes)
            free_mem, _ = torch.cuda.mem_get_info(i)
            
            to_allocate = free_mem - (buffer_mb * 1024 * 1024)
            
            if to_allocate > 0:
                # 使用 float32，每个元素 4 字节
                num_elements = to_allocate // 4
                t = torch.zeros((num_elements,), device=f'cuda:{i}', dtype=torch.float32)
                tensors.append(t)
                print(f"GPU {i}: 成功锁定约 {to_allocate / 1024**2:.2f} MB 显存 (保留 {buffer_mb}MB)")
            else:
                print(f"GPU {i}: 当前剩余显存 ({free_mem / 1024**2:.2f}MB) 已小于预留阈值，跳过分配。")
                
        except RuntimeError as e:
            print(f"GPU {i}: 分配过程中出现错误 - {e}")

    print("\n" + "="*40)
    print("[状态] 显存已锁定。脚本进入静默状态，不占用计算资源。")
    print("[操作] 按 Ctrl+C 释放显存并退出。")
    print("="*40)
    
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        print("\n[清理] 正在释放显存引用...")
        del tensors
        torch.cuda.empty_cache()
        print("[退出] 显存已归还系统。")

if __name__ == "__main__":
    # 默认给每个 GPU 留出 1GB 缓冲区
    buffer = 1024 
    if len(sys.argv) > 1:
        try:
            buffer = int(sys.argv[1])
        except ValueError:
            print("错误: 缓冲区大小必须是整数（单位 MB）。")
            sys.exit(1)
    
    occupy_gpu_memory(buffer)
