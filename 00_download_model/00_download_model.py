from modelscope.hub.snapshot_download import snapshot_download
import time
from concurrent.futures import ThreadPoolExecutor

# qwen/Qwen2.5-Coder-7B
def download_snapshot(*, model_id):
    print('Downloading snapshot {}'.format(model_id))
    snapshot_download(f'{model_id}', cache_dir='E:/Model/bge-large-zh-v1.5')

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        model_ids = ['BAAI/bge-large-zh-v1.5']
        start = time.time()
        for model_id in model_ids:
            executor.submit(download_snapshot, model_id=model_id)
        end = time.time()
        print('Total elapsed time: {}'.format(end - start))

if __name__ == '__main__':
    main()