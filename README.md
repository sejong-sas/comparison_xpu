# SKT & Rebellion Project

- Comparison between GPU & NPU

### Install
``` bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```
### Start

- You want to know about performance of your GPU

``` bash
vllm serve skt/A.X-4.0-Light\
        --port 8000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 4096 \
        --disable-log-requests
```

``` bash
python benchmarks/benchmark_serving.py     \
    --backend vllm     \
    --model skt/A.X-4.0-Light     \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json    \
    --request-rate 3.0
```

- You want to check state of your GPU

``` bash
pip install GPUtil
```

``` python
import GPUtil

def get_gpu_info_gputil():
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for gpu in gpus:
            gpu_info.append({
                'index': gpu.id,
                'name': gpu.name,
                'temperature': gpu.temperature,
                'utilization': gpu.load * 100, 
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_free': gpu.memoryFree
            })
        
        return gpu_info
    
    except Exception as e:
        print(f"GPU 정보를 가져오는 중 오류 발생: {e}")
        return None

# 사용 예시
gpu_info = get_gpu_info_gputil()
if gpu_info:
    for gpu in gpu_info:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  온도: {gpu['temperature']}°C")
        print(f"  사용률: {gpu['utilization']:.1f}%")
        print(f"  메모리: {gpu['memory_used']}/{gpu['memory_total']} MB")
        print()

```