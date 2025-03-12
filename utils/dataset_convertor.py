import json
import os
from os import PathLike
from typing import Union

import cv2
import numpy as np


def calc_letterbox_pad(src_size, target_size) -> (float, list[int]):
    """

    """

    src_width = src_size[0]
    src_height = src_size[1]

    target_width = target_size[0]
    target_height = target_size[1]

    ratio = min(target_width / src_width, target_height / src_height)

    dst_width = int(src_width * ratio)
    dst_height = int(src_height * ratio)

    pad_top = (target_height - dst_height) // 2
    pad_left = (target_width - dst_width) // 2
    pad_bottom = target_height - dst_height - pad_top
    pad_right = target_width - dst_height - pad_left

    return ratio, [pad_top, pad_bottom, pad_left, pad_right]


def letterbox(src: np.ndarray, target_size: Union[list[int], tuple[int]], padding_value: int = 0) -> np.ndarray:
    """
    
    """
    src_input = src
    if len(src.shape) ==3 and  src.shape[2] == 1:
        src_input = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    # src shape -> [h,w,c]
    src_width = src.shape[1]
    src_height = src.shape[0]

    ratio, [pad_top, pad_bottom, pad_left, pad_right] = calc_letterbox_pad([src_width, src_height], target_size)

    dst_width = int(src_width * ratio)
    dst_height = int(src_height * ratio)

    src_input = cv2.resize(src_input, (dst_width, dst_height))
    dst = cv2.copyMakeBorder(src_input, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                             None, [padding_value, padding_value, padding_value])
    return dst

def generate_masks_from_json(json_data:dict[str,object],category_list:list[str])->np.ndarray:
    image_height = json_data['imageHeight']
    image_width = json_data['imageWidth']
    shapes = json_data['shapes']
    # draw mask from background to foreground
    object_masks = [[] for _ in category_list]
    for shape in shapes:
        if shape['shape_type'] == "polygon" \
            or shape["shape_type"] == "rectangle":
            category = shape['label']
            shape_index = category_list.index(category)
            object_masks[shape_index].append(shape['points'])

    # _background value is 0
    mask_canvas = np.zeros([image_height,image_width],dtype=np.uint8)

    for index,masks in enumerate(object_masks) :
        for mask in masks:
            mask = [[int(p[0]),int(p[1])] for p in mask]
            pts = np.asarray([mask])
            cv2.fillPoly(mask_canvas,[pts],(index+1,index+1,index+1))

    return mask_canvas


def is_image_file(file: str) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif','.bmp']
    extension = file[file.rfind('.'):]
    if extension in image_extensions:
        return True
    else:
        return False

def is_label_file(file:str)->bool:
    label_extensions = ['.json']
    extension = file[file.rfind('.'):]
    if extension in label_extensions:
        return True
    else:
        return False

def file_name_without_extension(file_path):
    return os.path.basename(file_path)[:os.path.basename(file_path).rfind('.')]

def convert_coco_to_masks(src_images_path:Union[PathLike,str],
                          src_labels_path:Union[PathLike,str],
                          src_category_list: list[str],
                          dst_images_path: Union[PathLike,str],
                          dst_masks_path: Union[PathLike,str],
                          mask_suffix:str='_mask',
                          use_letterbox:bool=True,
                          target_size:Union[list[int],tuple[int]]=[640,640])->None:
    src_images_files = [file for file in os.listdir(src_images_path) if is_image_file(file)]
    src_labels_files = [file for file in os.listdir(src_labels_path) if is_label_file(file)]
    src_images_names = [file[:file.rfind('.')] for file in src_images_files]
    src_labels_names = [file[:file.rfind('.')] for file in src_labels_files]
    common_names = list(set(src_images_names) & set(src_labels_names))

    selected_images_files= [os.path.join(src_images_path,file) for file in src_images_files if file_name_without_extension(file) in common_names]
    selected_labels_files= [os.path.join(src_labels_path,file) for file in src_labels_files if file_name_without_extension(file) in common_names]

    for index,image_file in enumerate (selected_images_files):
        label_file = [file for file in selected_labels_files
                      if file_name_without_extension(file) == file_name_without_extension(image_file)][0]

        with open(label_file,'r') as f:

            save_image = cv2.imread(image_file,cv2.IMREAD_UNCHANGED)
            json_data = json.load(f)
            save_mask = generate_masks_from_json(json_data,category_list=src_category_list)

            if use_letterbox:
                save_image = letterbox(save_image, target_size)
                save_mask = letterbox(save_mask,target_size)

            save_image_name = os.path.basename(image_file)

            save_mask_name = save_image_name[:save_image_name.rfind('.')]+mask_suffix+save_image_name[save_image_name.rfind('.'):]

            save_image_full_path = os.path.join(dst_images_path,save_image_name)
            save_mask_full_path = os.path.join(dst_masks_path,save_mask_name)


            cv2.imwrite(save_image_full_path,save_image)
            cv2.imwrite(save_mask_full_path,save_mask)

            print(f'generate image-mask: {os.path.basename(image_file)}')



if __name__ == '__main__':
    src_images_path = r'D:\Code\XRayInspection\images\PCB\dev\PCBBubble1111\images\train'
    src_labels_path = r'D:\Code\XRayInspection\images\PCB\dev\PCBBubble1111\labels\train'
    src_category_list=['chip','bubble']
    dst_images_path=r'D:\Code\github_repo\Pytorch-UNet\data\imgs'
    dst_masks_path=r'D:\Code\github_repo\Pytorch-UNet\data\masks'
    convert_coco_to_masks(src_images_path,src_labels_path,src_category_list,dst_images_path,dst_masks_path)


