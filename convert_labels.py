import cv2
import os
import numpy as np
import shutil


def convert_polygon_to_rectangle(polygon_points, image_width, image_height):
    pts = np.array(polygon_points, np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    x_center = (x + x + w) / 2 / image_width
    y_center = (y + y + h) / 2 / image_height
    width = w / image_width
    height = h / image_height

    return x_center, y_center, width, height


def draw_polygon(image, polygon_points, color):
    pts = np.array(polygon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)


def draw_rectangle(image, x_center, y_center, width, height, image_width, image_height, color):
    x1 = int((x_center - width / 2) * image_width)
    y1 = int((y_center - height / 2) * image_height)
    x2 = int((x_center + width / 2) * image_width)
    y2 = int((y_center + height / 2) * image_height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def process_labels(labels_dir, images_dir, output_labels_dir, output_images_dir):
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    # 1. 复制备份文件到目标文件夹
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_labels_dir, label_file))

    # 2. 处理标注数据并绘制结果
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')  # 假设图像文件为.jpg格式
            image_path = os.path.join(images_dir, image_file)
            original_label_path = os.path.join(labels_dir, label_file)
            output_label_path = os.path.join(output_labels_dir, label_file)
            output_image_path = os.path.join(output_images_dir, image_file)

            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image {image_path}.")
                continue

            h, w, _ = image.shape
            new_labels = []

            with open(original_label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                annotation = parts[1:]

                if len(annotation) == 4:
                    # YOLO 矩形标注
                    x_center, y_center, width, height = map(float, annotation)
                    draw_rectangle(image, x_center, y_center, width, height, w, h, (0, 0, 255))  # 原始标注（红色）
                    new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
                else:
                    # 多边形标注
                    points = [float(p) for p in annotation]
                    polygon_points = [(points[i] * w, points[i + 1] * h) for i in range(0, len(points), 2)]
                    draw_polygon(image, polygon_points, (0, 0, 255))  # 原始标注（红色）
                    x_center, y_center, width, height = convert_polygon_to_rectangle(polygon_points, w, h)
                    new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
                    draw_rectangle(image, x_center, y_center, width, height, w, h, (0, 255, 0))  # 转换后的标注（绿色）

            # 保存新标签
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(new_labels))

            # 保存绘制后的图像
            cv2.imwrite(output_image_path, image)
            print(f"Processed {label_file} and saved to {output_label_path} and {output_image_path}")


# 文件夹路径
labels_dir = r'ori_labels'
images_dir = r'ori_imgs'
output_labels_dir = r'after_convert_labels'
output_images_dir = r'show_result'

process_labels(labels_dir, images_dir, output_labels_dir, output_images_dir)
