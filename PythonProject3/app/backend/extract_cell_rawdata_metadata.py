# 测试
# 打印字符的Unicode编码
# print([ord(c) for c in "标本编号"])

import os
import re
import csv
import pandas as pd

def scan_data_directory(root_dir):
    # 定义存储符合规范的目录信息的列表
    valid_data = []

    # 遍历根目录下的所有子目录
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)

        # 检查是否为目录
        if os.path.isdir(dir_path):
            # 分解目录名，提取四个字段：日期，设备编号，地点，镜头类别
            fields = dir_name.split('-')
            if len(fields) != 4:
                continue  # 如果字段不完整，跳过该目录
            date, device_id, location, lens_type = fields

            # 检查是否包含背景图像目录、4K相机数据目录、多光谱相机数据目录
            background_dir = None
            fourk_dirs = []
            multispectral_dirs = []

            for sub_dir_name in os.listdir(dir_path):
                sub_dir_path = os.path.join(dir_path, sub_dir_name)

                if os.path.isdir(sub_dir_path):
                    # 查找背景图像目录
                    if '背景' in sub_dir_name:
                        background_dir = sub_dir_path

                    # 查找4K相机数据目录
                    if '4k相机' in sub_dir_name:
                        fourk_dirs.append(sub_dir_path)

                    # 查找多光谱相机数据目录
                    if '-' in sub_dir_name and len(sub_dir_name.split('-')) in [4, 5, 6]:
                        multispectral_dirs.append(sub_dir_path)

            # 当4K相机和多光谱相机数据目录数量一致时才继续
            if len(fourk_dirs) == len(multispectral_dirs) and background_dir:
                # 提取各类图像的文件编号列表
                background_files = {
                    'data_dir': dir_name, 
                    'image_dir': background_dir.split('/')[-1],
                    'device_id': device_id,
                    'lens_type': lens_type,
                    'images': extract_image_numbers(background_dir, ['bmp', 'HDR', 'raw'])
                }
                fourk_files = []
                multispectral_files = []

                fourk_sample_ids = set()
                multispectral_sample_ids = set()

                # Process each 4K camera and multispectral folder
                for i in range(len(fourk_dirs)):
                    # Get the sample_id for the current directory
                    subcategory, cell_type, multispectral_sample_id, multispectral_lens_type, multispectral_device_id = extract_multispectral_metadata(multispectral_dirs[i])
                    if multispectral_device_id != device_id:
                        print(f"Warning: Device ID in parent and multispectral directories do not match. {device_id} vs {multispectral_device_id} in {dir_name}")

                    if len(fourk_dirs) == 1:
                        fourk_sample_id = multispectral_sample_id
                    else:
                        fourk_sample_id = extract_sample_id_from_fourk(fourk_dirs[i])
                    
                    fourk_sample_ids.add(fourk_sample_id)
                    multispectral_sample_ids.add(multispectral_sample_id)

                    # Create dictionary for multispectral files with metadata and images
                    multispectral_dir_info = {
                        'data_dir': dir_name,
                        'image_dir': multispectral_dirs[i].split('/')[-1],
                        'device_id': multispectral_device_id,
                        'lens_type': multispectral_lens_type,
                        'sample_id': multispectral_sample_id,
                        'subcategory': subcategory,
                        'cell_type': cell_type,
                        'images': extract_image_numbers(multispectral_dirs[i], ['bmp', 'HDR', 'raw'])
                    }

                    # Create dictionary for 4K files with sample_id and images
                    fourk_dir_info = {
                        'data_dir': dir_name,
                        'image_dir': fourk_dirs[i].split('/')[-1],
                        'sample_id': fourk_sample_id,
                        'images': extract_image_numbers(fourk_dirs[i], ['bmp', 'png'])
                    }

                    # Append the folder information to the respective lists
                    fourk_files.append(fourk_dir_info)
                    multispectral_files.append(multispectral_dir_info)

                # 检查4K相机和多光谱数据的标本编号是否一致
                if sorted(fourk_sample_ids) != sorted(multispectral_sample_ids):
                    print(f"Warning: Sample IDs in 4K and multispectral directories do not match. {fourk_sample_ids} vs {multispectral_sample_ids} in {dir_name}")
                else:
                    print(f"Directory checked: {dir_name}")

                # Now that all multispectral_files are populated, add subcategory and cell_type to fourk_files
                for fourk_dir_info in fourk_files:
                    fourk_sample_id = fourk_dir_info['sample_id']
                    
                    # Find the corresponding multispectral directory's subcategory and cell_type for the sample_id
                    for multispectral in multispectral_files:
                        if multispectral['sample_id'] == fourk_sample_id:
                            # Add subcategory and cell_type from multispectral to 4K file metadata
                            fourk_dir_info['subcategory'] = multispectral['subcategory']
                            fourk_dir_info['cell_type'] = multispectral['cell_type']
                            break  # Stop once we find the corresponding multispectral entry

                # 将符合规范的数据目录信息保存到列表
                valid_data.append([
                    dir_name, date, device_id, location, lens_type,
                    background_files, fourk_files, multispectral_files
                ])

    return valid_data

def extract_image_numbers(directory, valid_extensions):
    """从指定目录中提取符合扩展名要求的文件的编号"""
    file_numbers = set()  # Use a set to store numbers to avoid duplicates
    for file_name in os.listdir(directory):
        if any(file_name.endswith(ext) for ext in valid_extensions):
            match = re.match(r"(\d+)", file_name)
            if match:
                file_number = int(match.group(1))
                file_numbers.add(file_number)  # Add the file number to the set (avoids duplicates)
    return sorted(file_numbers)  # Convert the set back to a sorted list

import re

def extract_sample_id_from_fourk(directory):
    """从4K相机目录名中提取标本编号"""
    sample_id = None
    
    # Check if the directory name contains a hyphen
    if '-' in directory.split('/')[-1]:
        # Case: directory name like "231700-4k相机" or "231700-4K相机"
        match = re.search(r"(\d+)-?4?k?相机", directory, re.IGNORECASE)
        if match:
            sample_id = match.group(1)
    else:
        # Case: directory name like "4k相机231700" or "4K相机231700"
        match = re.search(r"4?k?相机(\d+)", directory, re.IGNORECASE)
        if match:
            sample_id = match.group(1)
    
    return sample_id

def extract_sample_id_from_multispectral(directory):
    """从多光谱目录名中提取标本编号"""
    sample_id = None
    # 在目录名中查找标本编号（注意：用负号或其他非数字字符分隔）
    match = re.search(r"标本编号(\d+)", directory)
    if match:
        sample_id = match.group(1)
    return sample_id

def extract_multispectral_metadata(directory):
    """从多光谱数据的目录名中提取标注信息"""
    # 默认值
    directory = directory.split('/')[-1]
    subcategory = ""
    cell_type = ""
    sample_id = None
    lens_type = ""
    device_id = ""

    # 按照"-"分割目录名
    segments = directory.split('-')

    # 处理5段和4段名称
    if len(segments) > 4:
        # 5段名称的提取
        subcategory = '-'.join(segments[:-4])  # 数据类别（细分类）
        cell_type = segments[-4]  # 细胞类型
        # 提取标本编号
        match = re.search(r"标本编号(\d+)", segments[-3])
        if match:
            sample_id = match.group(1)
        # 镜头类型
        lens_type = segments[-2]
        # 设备编号
        device_id = segments[-1]
    
    elif len(segments) == 4:
        # 4段名称的提取
        cell_type = segments[0]  # 细胞类型
        # 提取标本编号
        match = re.search(r"标本编号(\d+)", segments[1])
        if match:
            sample_id = match.group(1)
        # 镜头类型
        lens_type = segments[2]
        # 设备编号
        device_id = segments[3]
    else:
        print(f"Warning: Invalid directory name format: {directory}")
    
    return subcategory, cell_type, sample_id, lens_type, device_id


def save_to_csv(valid_data, filename):
    # Open the file in write mode
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Define the CSV writer
        writer = csv.writer(csvfile)

        # Write the header (column names)
        writer.writerow(['date', 'device_id', 'location', 'lens_type', 'background_files', 'fourk_files', 'multispectral_files'])

        # Write the data rows
        for data in valid_data:
            writer.writerow(data)


def save_to_excel_multiple_sheets(valid_data, filename):
    # Create an Excel writer object
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Create a sheet for the main data
        main_data = []
        for data in valid_data:
            dir_name, date, device_id, location, lens_type, background_files, fourk_files, multispectral_files = data

            # # Concatenate the 'image_dir' from fourk_files
            # fourk_image_dirs = [fourk_dir['image_dir'] for fourk_dir in fourk_files]
    
            # # Concatenate the 'image_dir' from multispectral_files
            # multispectral_image_dirs = [multispectral_dir['image_dir'] for multispectral_dir in multispectral_files]
    
            main_data.append([dir_name, date, device_id, location, lens_type, len(fourk_files), len(multispectral_files), len(background_files)])
        
        df_main = pd.DataFrame(main_data, columns=['data_dir', 'date', 'device_id', 'location', 'lens_type', 'num_4k_dirs', 'num_multispectral_dirs', 'num_background_dirs'])
        df_main.to_excel(writer, sheet_name='main', index=False)

        # Create a sheet for the background files
        background_data = []
        for data in valid_data:
            background_files = data[5]  # Background files
            background_data.append({**{k: v for k, v in background_files.items() if k != 'images'}, 
                                    'num_images': len(background_files['images']), 
                                    'images': ', '.join(map(str, background_files['images']))})  # Convert list to a string
        df_background = pd.DataFrame(background_data)
        df_background.to_excel(writer, sheet_name='background', index=False)

        # Create a sheet for the 4K files
        fourk_data = []
        for data in valid_data:
            fourk_dirs = data[6]  # 4K files
            for fourk_dir in fourk_dirs:
                # Concatenate the 'images' list into a comma-separated string
                image_str = ', '.join(map(str, fourk_dir['images']))
                fourk_data.append({**{k: v for k, v in fourk_dir.items() if k != 'images'}, 'num_images': len(fourk_dir['images']), 'images': image_str})
        df_fourk = pd.DataFrame(fourk_data)
        df_fourk.to_excel(writer, sheet_name='4k', index=False)

        # Create a sheet for the multispectral files
        multispectral_data = []
        for data in valid_data:
            multispectral_files = data[7]  # Multispectral files
            for multispectral_dir in multispectral_files:
                image_str = ', '.join(map(str, multispectral_dir['images']))
                multispectral_data.append({**{k: v for k, v in multispectral_dir.items() if k != 'images'}, 'num_images': len(multispectral_dir['images']), 'images': image_str})
        df_multispectral = pd.DataFrame(multispectral_data)
        df_multispectral.to_excel(writer, sheet_name='multispectral', index=False)

if __name__ == '__main__':
    
    # root_dir = '/media/ljm/全4/细胞/采样数据/'
    root_dir = '/mnt/truenas_datasets/Multispectral_Pathology/细胞/采样数据/'

    valid_data = scan_data_directory(root_dir)

    # for data in valid_data:
    #     print(data)

    print(len(valid_data))

    # Save the valid_data to an Excel file with multiple sheets
    save_to_excel_multiple_sheets(valid_data, 'valid_raw_data_multiple_sheets.xlsx')

    print("Done!")
