import os
import tarfile
import urllib.request
import scipy.io
import numpy as np
import shutil
from PIL import Image

def is_valid_tar(filename):
    """检查是否为合法压缩包"""
    return tarfile.is_tarfile(filename)

def download_file(urls, filename):
    if os.path.exists(filename):
        if is_valid_tar(filename):
            print(f"{filename} already exists and is valid")
            return
        else:
            print(f"{filename} is corrupted, re-downloading...")
            os.remove(filename)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:
        try:
            print(f"Trying {url} ...")
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req) as response, open(filename, "wb") as f:
                f.write(response.read())

            # ✅ 下载后立即校验
            if not is_valid_tar(filename):
                print(f"{filename} is not a valid tar file, removing...")
                os.remove(filename)
                continue

            print(f"Downloaded {filename} from {url}")
            return

        except Exception as e:
            print(f"Failed: {url} -> {e}")

    raise RuntimeError(f"All download sources failed for {filename}")

def extract_tar(filename, extract_path):
    print(f"Extracting {filename}...")
    with tarfile.open(filename) as tar:
        tar.extractall(extract_path)

def main():
    base_dir = "./dataset/VOCAug"
    os.makedirs(base_dir, exist_ok=True)

    # ✅ VOC 官方
    voc_urls = [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    ]

    # ✅ SBD（GitHub Release 稳定源）
    sbd_urls = [
        # ✅ HuggingFace（当前最稳，强烈推荐）
        "https://huggingface.co/datasets/zhanghang1989/ResNet-101-DeepLab-v2-PASCAL/resolve/main/benchmark.tgz",

        # ✅ GitHub镜像（备用）
        "https://github.com/zhanghang1989/PyTorch-Encoding/releases/download/v1.0/benchmark.tgz"  
    ]

    voc_tar = "VOCtrainval_11-May-2012.tar"
    sbd_tar = "benchmark.tgz"

    download_file(voc_urls, voc_tar)
    download_file(sbd_urls, sbd_tar)

    extract_tar(voc_tar, "./dataset")
    extract_tar(sbd_tar, "./dataset")

    # =========================
    # 构建 VOCAug
    # =========================
    print("Organizing VOCAug...")

    voc2012_dir = "./dataset/VOCdevkit/VOC2012"

    os.makedirs(os.path.join(base_dir, "JPEGImages"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "SegmentationClass"), exist_ok=True)

    os.system(f"cp -r {voc2012_dir}/JPEGImages/* {base_dir}/JPEGImages/")
    os.system(f"cp -r {voc2012_dir}/SegmentationClass/* {base_dir}/SegmentationClass/")

    # =========================
    # 处理 SBD
    # =========================
    print("Processing SBD...")

    sbd_dir = "./dataset/benchmark_RELEASE/dataset/cls"
    sbd_img_dir = "./dataset/benchmark_RELEASE/dataset/img"

    aug_cls_dir = os.path.join(base_dir, "SegmentationClassAug")
    os.makedirs(aug_cls_dir, exist_ok=True)

    # 复制 VOC mask
    os.system(f"cp -r {voc2012_dir}/SegmentationClass/* {aug_cls_dir}/")

    import glob
    mat_files = glob.glob(os.path.join(sbd_dir, "*.mat"))

    for mat_file in mat_files:
        name = os.path.basename(mat_file).replace('.mat', '')

        mat = scipy.io.loadmat(mat_file)
        lbl = mat['GTcls'][0]['Segmentation'][0]

        img = Image.fromarray(lbl.astype(np.uint8))
        img.save(os.path.join(aug_cls_dir, f"{name}.png"))

        # 补图像
        img_src = os.path.join(sbd_img_dir, f"{name}.jpg")
        img_dst = os.path.join(base_dir, f"JPEGImages/{name}.jpg")

        if not os.path.exists(img_dst) and os.path.exists(img_src):
            shutil.copy(img_src, img_dst)

    # =========================
    # 清理
    # =========================
    print("Cleaning up...")
    shutil.rmtree("./dataset/VOCdevkit", ignore_errors=True)
    shutil.rmtree("./dataset/benchmark_RELEASE", ignore_errors=True)

    print("Done! VOCAug is at ./dataset/VOCAug")

if __name__ == "__main__":
    main()