import os
import shutil
import random
from sklearn.model_selection import train_test_split

# ===================== 配置参数（根据你的数据集修改） =====================
# 原始训练集路径（包含所有植物种类的子文件夹）
original_train_dir = "C:/Users/32992/Desktop/spacemit_houseplant/house_plant/pre"  # 结构示例：original_train/月季/xxx.jpg、original_train/玫瑰/xxx.jpg...
# 划分后的数据保存路径
new_train_dir ="C:/Users/32992/Desktop/spacemit_houseplant/house_plant/train"   # 新训练集保存路径
new_val_dir = "C:/Users/32992/Desktop/spacemit_houseplant/house_plant/valid"   # 验证集保存路径
split_ratio = 0.2  # 验证集占比（8:2划分）
random_seed = 42  # 随机种子，保证结果可复现

'''
# ===================== 核心函数：创建文件夹 =====================
def create_dir(path):
    """创建文件夹，若已存在则不报错"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建文件夹：{path}")
    else:
        print(f"文件夹已存在：{path}")


# ===================== 核心逻辑：分层抽样并划分数据集 =====================
def split_dataset():
    # 1. 创建新的训练集和验证集根目录
    create_dir(new_train_dir)
    create_dir(new_val_dir)

    # 2. 获取所有植物种类（原始训练集的子文件夹名即类别名）
    classes = os.listdir(original_train_dir)
    print(f"共检测到 {len(classes)} 种植物：{classes}")

    # 3. 遍历每个类别，分层抽样划分
    for cls in classes:
        # 类别原始路径
        cls_original_path = os.path.join(original_train_dir, cls)
        # 跳过非文件夹（避免误判）
        if not os.path.isdir(cls_original_path):
            continue

        # 获取该类别下所有图片路径
        img_names = os.listdir(cls_original_path)
        img_paths = [os.path.join(cls_original_path, name) for name in img_names]
        total = len(img_paths)
        print(f"\n处理类别：{cls}，共 {total} 张图片")

        # 跳过样本量不足的类别（至少需要5张，否则划分意义不大）
        if total < 5:
            print(f"警告：{cls} 样本量不足5张，不进行划分，全部放入训练集")
            # 创建该类别在新训练集的文件夹
            cls_train_path = os.path.join(new_train_dir, cls)
            create_dir(cls_train_path)
            # 复制所有图片到新训练集
            for img_path in img_paths:
                shutil.copy(img_path, os.path.join(cls_train_path, os.path.basename(img_path)))
            continue

        # 分层抽样（按8:2划分，stratify参数保证类别内比例一致）
        # 由于单类别划分，stratify设为None即可（本质是对每个类别单独随机抽样）
        train_imgs, val_imgs = train_test_split(
            img_paths,
            test_size=split_ratio,
            random_state=random_seed
        )

        # 4. 创建该类别在新训练集和验证集的文件夹
        cls_train_path = os.path.join(new_train_dir, cls)
        cls_val_path = os.path.join(new_val_dir, cls)
        create_dir(cls_train_path)
        create_dir(cls_val_path)

        # 5. 复制图片到对应文件夹
        # 复制训练集图片
        for img_path in train_imgs:
            shutil.copy(img_path, os.path.join(cls_train_path, os.path.basename(img_path)))
        # 复制验证集图片
        for img_path in val_imgs:
            shutil.copy(img_path, os.path.join(cls_val_path, os.path.basename(img_path)))

        # 打印划分结果
        print(f"  训练集：{len(train_imgs)} 张 | 验证集：{len(val_imgs)} 张")

    print("\n===== 数据集划分完成 =====")
    print(f"新训练集路径：{new_train_dir}")
    print(f"验证集路径：{new_val_dir}")


# ===================== 运行划分 =====================
if __name__ == "__main__":
    # 固定随机种子，保证结果可复现
    random.seed(random_seed)
    split_dataset()
'''