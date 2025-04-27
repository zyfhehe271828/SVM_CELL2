
import logging
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from joblib import load
import sys
import os
import json
from openpyxl import Workbook
from collections import Counter
from app.backend import ms_utils
from flask import send_file
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print("✅ NEW VERSION RUNNING ✅")
# 指定模板目录
template_dir = os.path.join(os.getcwd(), 'app', 'templates')
# 指定静态文件目录
static_dir = os.path.join(os.getcwd(), 'app', 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# 打印当前工作目录
logger.info(f"Current working directory: {os.getcwd()}")

# 定义大肠杆菌和金黄色葡萄球菌模型和参数保存的目录，使用调整后的相对路径
ecoli_model_dir = os.path.join('models', 'saved_Esh_Shi_models')
s_aureus_model_dir = os.path.join('models', 'saved_MRSA_MSSA_models')
num_folds = 5
feature_method = 'pixelwise'  # 可根据需要修改为 'average'


# 封装加载模型和参数的函数
def load_models_and_params(model_dir):
    models = []
    train_means = []
    train_stds = []
    for fold_num in range(1, num_folds + 1):
        model_filename = os.path.join(model_dir, f"svm_model_fold{fold_num}_{feature_method}.pkl")
        mean_filename = os.path.join(model_dir, f"train_mean_fold{fold_num}_{feature_method}.npy")
        std_filename = os.path.join(model_dir, f"train_std_fold{fold_num}_{feature_method}.npy")
        try:
            model = load(model_filename)
        except FileNotFoundError:
            logger.error(f"Error: Model file not found for fold {fold_num} in {model_dir}. File path: {model_filename}")
            continue
        try:
            train_mean = np.load(mean_filename)
        except FileNotFoundError:
            logger.error(f"Error: Mean file not found for fold {fold_num} in {model_dir}. File path: {mean_filename}")
            continue
        try:
            train_std = np.load(std_filename)
        except FileNotFoundError:
            logger.error(f"Error: Std file not found for fold {fold_num} in {model_dir}. File path: {std_filename}")
            continue

        models.append(model)
        train_means.append(train_mean)
        train_stds.append(train_std)
    return models, train_means, train_stds


# 加载大肠杆菌模型和标准化参数
ecoli_models, ecoli_train_means, ecoli_train_stds = load_models_and_params(ecoli_model_dir)

# 加载金黄色葡萄球菌模型和标准化参数
s_aureus_models, s_aureus_train_means, s_aureus_train_stds = load_models_and_params(s_aureus_model_dir)


def find_related_files(file_path, original_dir):
    logger.info(f"Original file directory: {original_dir}")  # 输出原文件所在地址
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    related_files = []

    if file_path.lower().endswith('.hdr'):
        if '-background' in base_name:
            # 如果文件名包含 -background，不进行额外的文件查找
            pass
        else:
            # 如果文件名不包含 -background，寻找对应的带 -background 的 HDR 文件
            background_base_name = base_name + '-background'
            background_file = os.path.join(original_dir, background_base_name + '.HDR')
            logger.info(f"Searching for background file: {background_file}")
            if os.path.exists(background_file):
                related_files.append(background_file)
            else:
                logger.warning(f"Related background file not found: {background_file}")



    return related_files



def predict_single_image( model_type,excel_path):
    # logger.info(f"Processing original file: {image_path} from directory: {original_dir}")  # 输出原文件信息
    # related_files = find_related_files(image_path, original_dir)
    # 读取 Excel 数据
    df = pd.read_excel(excel_path)
    num_rows = df.shape[0]
    row_all = df[['folder_path_new', 'hdr_filename_new', 'background_hdr_filename_new']].to_dict('records')
    results = []
    # data_root = os.path.join('app', 'temp')
    # logger.info(f"Data root directory: {data_root}")  # 输出数据根目录
    if model_type == "ecoli":
        models, means, stds = ecoli_models, ecoli_train_means, ecoli_train_stds
    elif model_type == "s_aureus":
        models, means, stds = s_aureus_models, s_aureus_train_means, s_aureus_train_stds
    else:
        raise ValueError("Invalid model type")

    for i in range(num_rows):
        row = row_all[i]
        background_subtraction = not pd.isna(row['background_hdr_filename_new'])

        sample = ms_utils._process_single_image_background_division(row,  background_subtraction, 7)
        if sample is None:
            results.append({"filename": row['hdr_filename_new'], "prediction": None, "probability": None})
            continue

        spectra = sample["spectra"]

        fold_votes = []

        for model, mean, std in zip(models, means, stds):
            # 标准化
            spectra_std = (spectra - mean) / std
            preds = model.predict(spectra_std)  # 每个 block 的预测

            # 每个模型的投票结果（对这张图像）
            vote = Counter(preds).most_common(1)[0][0]
            fold_votes.append(vote)

            # 计算总票数
            total_votes = len(fold_votes)

            # 使用 logger 记录总票数
            logger.info(f"Total votes: {total_votes}")


        # 最终多数投票分类
        final_vote, count = Counter(fold_votes).most_common(1)[0]
        probability = count / len(fold_votes)

        results.append({
            "filename": row['hdr_filename_new'],
            "prediction": int(final_vote),
            "probability": round(probability, 3)
        })

    return results


##预处理完成，下面为分类逻辑
    # if model_type == 'ecoli':
    #     models = ecoli_models
    #     train_means = ecoli_train_means
    #     train_stds = ecoli_train_stds
    # elif model_type == 's_aureus':
    #     models = s_aureus_models
    #     train_means = s_aureus_train_means
    #     train_stds = s_aureus_train_stds
    # else:
    #     logger.error(f"Invalid model type: {model_type}")
    #     continue
    #
    #     # 交叉预测
    # votes = []
    # for model, mean, std in zip(models, train_means, train_stds):
    #     # 标准化数据，避免除零错误
    #     standardized_data = (processed_data - mean) / (std + 1e-8)
    #     # 预测
    #     prediction = model.predict(standardized_data)
    #     votes.append(prediction)
    #
    # # 投票统计
    # positive_votes = np.sum(votes)
    # negative_votes = len(votes) - positive_votes
    # total_votes = len(votes)
    # positive_prob = positive_votes / total_votes
    # negative_prob = negative_votes / total_votes
    #
    # result = {
    #     'filename': os.path.basename(file),
    #     'probabilities': {
    #         'positive': positive_prob,
    #         'negative': negative_prob
    #     }
    # }
    # results.append(result)
    #
    # except Exception as e:
    # logger.error(f"Error processing file {file}: {e}")
    #
    # return results


@app.route('/')
def index():
    return render_template('index.html')


# 定义 temp 文件夹路径
temp_dir = os.path.join('app', 'temp')

# 创建必要的目录
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


# def save_to_excel(files, original_paths):
#     data = []
#     for file, original_path in zip(files, original_paths):
#         file_path = os.path.join(temp_dir, os.path.basename(file.filename))
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         dir_path = os.path.dirname(file_path)
#         background_filename = base_name + '.background.HDR'
#
#         data.append({
#             'folder_path_new': dir_path,
#             'hdr_filename_new': os.path.basename(file_path),
#             'background_hdr_filename_new': background_filename
#         })
#
#     related_files = []
#     for file, original_path in zip(files, original_paths):
#         file_path = os.path.join(temp_dir, os.path.basename(file.filename))
#         original_dir = os.path.dirname(original_path)
#         logger.info(f"For file {file.filename}, original directory: {original_dir}")
#         related_files.extend(find_related_files(file_path, original_dir))
#
#     for related_file in related_files:
#         related_base_name = os.path.splitext(os.path.basename(related_file))[0]
#         related_dir_path = os.path.dirname(related_file)
#         related_background_filename = related_base_name + '.background.HDR'
#
#         data.append({
#             'folder_path_new': related_dir_path,
#             'hdr_filename_new': os.path.basename(related_file),
#             'background_hdr_filename_new': related_background_filename
#         })
#
#     df = pd.DataFrame(data)
#     excel_path = os.path.join(temp_dir, 'file_info.xlsx')
#     df.to_excel(excel_path, index=False)
#     logger.info(f"Excel file saved to {excel_path}")

def save_to_excel(files, original_paths):
    data = []
    for file, original_path in zip(files, original_paths):
        file_path = os.path.join(temp_dir, os.path.basename(file.filename))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_path = os.path.dirname(file_path)

        # 构造对应背景图路径
        background_filename_candidate = base_name + '-background.HDR'
        background_file_path = os.path.join(dir_path, background_filename_candidate)

        # 检查是否存在该背景文件
        background_filename = background_filename_candidate if os.path.exists(background_file_path) else ""

        data.append({
            'folder_path_new': dir_path,
            'hdr_filename_new': os.path.basename(file_path),
            'background_hdr_filename_new': background_filename
        })

    # 相关文件处理逻辑可按需保留或去除，此处原样保留
    related_files = []
    for file, original_path in zip(files, original_paths):
        file_path = os.path.join(temp_dir, os.path.basename(file.filename))
        original_dir = os.path.dirname(original_path)
        logger.info(f"For file {file.filename}, original directory: {original_dir}")
        related_files.extend(find_related_files(file_path, original_dir))

    for related_file in related_files:
        related_base_name = os.path.splitext(os.path.basename(related_file))[0]
        related_dir_path = os.path.dirname(related_file)

        related_background_candidate = related_base_name + '-background.HDR'
        related_background_path = os.path.join(related_dir_path, related_background_candidate)
        related_background_filename = related_background_candidate if os.path.exists(related_background_path) else ""

        data.append({
            'folder_path_new': related_dir_path,
            'hdr_filename_new': os.path.basename(related_file),
            'background_hdr_filename_new': related_background_filename
        })

    df = pd.DataFrame(data)
    excel_path = os.path.join(temp_dir, 'file_info.xlsx')
    df.to_excel(excel_path, index=False)
    logger.info(f"Excel file saved to {excel_path}")

    # 重新读取刚刚保存的 Excel
    df = pd.read_excel(excel_path)

    # 拆分出主图和背景图
    main_df = df[df['hdr_filename_new'].str.endswith('.HDR') & ~df['hdr_filename_new'].str.contains('-background',
                                                                                                    case=False)].copy()
    background_df = df[df['hdr_filename_new'].str.contains('-background', case=False)].copy()

    # 提取文件名前缀（如 A）
    main_df['prefix'] = main_df['hdr_filename_new'].str.extract(r'^(.*)\.HDR$', expand=False)
    background_df['prefix'] = background_df['hdr_filename_new'].str.extract(r'^(.*)-background\.HDR$', expand=False)

    # 合并主图与对应背景图
    result_df = pd.merge(
        main_df.drop(columns=['background_hdr_filename_new'], errors='ignore'),
        background_df[['prefix', 'hdr_filename_new']].rename(
            columns={'hdr_filename_new': 'background_hdr_filename_new'}),
        on='prefix',
        how='left'
    )

    # 删除辅助列
    result_df.drop(columns=['prefix'], inplace=True)

    # 保存更新后的表格（可覆盖原文件或另存）
    result_df.to_excel(excel_path, index=False)
    logger.info(f"Cleaned Excel file saved to {excel_path}")
    return excel_path  # 返回保存的 Excel 文件路径


def save_results_to_excel(all_results):
    """将结果保存为Excel文件"""
    try:
        # 创建结果目录（如果不存在）
        result_dir = 'result'
        os.makedirs(result_dir, exist_ok=True)

        # 定义文件路径（使用相对路径）
        excel_filename = 'results.xlsx'
        excel_path = os.path.join(result_dir, excel_filename)

        # 将结果转换为DataFrame并保存
        df = pd.DataFrame(all_results)
        df.to_excel(excel_path, index=False)

        logger.info(f"Results successfully saved to: {os.path.abspath(excel_path)}")
        return excel_path

    except Exception as e:
        logger.error(f"Error saving results to Excel: {str(e)}", exc_info=True)
        raise  # 重新抛出异常让调用者处理


@app.route('/download_results', methods=['GET'])
def download_results():
    try:
        # 定义相对路径（推荐方式）
        result_dir = 'result'  # 相对于应用根目录的result文件夹
        excel_filename = 'results.xlsx'
        excel_path = os.path.join(result_dir, excel_filename)

        # 记录调试信息
        logger.info(f"Attempting to download file from: {os.path.abspath(excel_path)}")

        # 检查文件是否存在
        if not os.path.exists(excel_path):
            logger.error(f"File not found: {excel_path}")
            return "结果文件不存在，请先生成结果", 404

        # 确保目录存在（虽然检查文件存在时目录必然存在，但保留这行也无妨）
        os.makedirs(result_dir, exist_ok=True)

        # 发送文件
        return send_file(
            excel_path,
            as_attachment=True,
            download_name='analysis_results.xlsx'  # 建议指定下载时的文件名
        )

    except Exception as e:
        logger.error(f"Error downloading results: {str(e)}", exc_info=True)
        return f"下载结果时发生错误: {str(e)}", 500



@app.route('/classify_ecoli', methods=['POST'])
def classify_ecoli():
    try:
        clear_result_directory()
        files = request.files.getlist('file')
        original_paths_str = request.form.get('original_path')
        if original_paths_str is None:
            logger.error("Error: 'original_path' is not provided in the request.")
            return jsonify({"error": "Missing 'original_path' in the request."}), 400
        original_paths = original_paths_str.split(',')
        excel_path = save_to_excel(files, original_paths)  # 获取保存的 Excel 文件路径

        all_results = []
        for file, original_path in zip(files, original_paths):
            # 使用 os.path.basename() 函数来获取最内层地址
            file.filename = os.path.basename(file.filename)

            file_path = os.path.join(temp_dir, file.filename)
            # original_dir = os.path.dirname(original_path)
            try:
                file.save(file_path)

                results = predict_single_image( 'ecoli', excel_path)
                logger.info("results is" + str(results))
                #Escherichia_coli是0，SHigella是1
                if results:
                    all_results.extend(results)
                    excel_path = save_results_to_excel(all_results)
                clear_temp_directory()  # 添加清空temp文件夹的调用
            except Exception as e:
                # logger.info("file_path is" + file_path)
                # logger.info(f"File saved to {file_path} from original directory: {original_dir}")
                logger.error(f"Error saving or processing file {file.filename}: {e}")
                # 保存 all_results 到 Excel 文件
                # 返回结果
        return jsonify(all_results)
    except Exception as e:
        logger.error(f"Error in classify_ecoli route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/classify_s_aureus', methods=['POST'])
def classify_s_aureus():
    try:
        clear_result_directory()
        files = request.files.getlist('file')
        original_paths_str = request.form.get('original_path')
        if original_paths_str is None:
            logger.error("Error: 'original_path' is not provided in the request.")
            return jsonify({"error": "Missing 'original_path' in the request."}), 400
        original_paths = original_paths_str.split(',')
        excel_path = save_to_excel(files, original_paths)  # 获取保存的 Excel 文件路径

        all_results = []
        for file, original_path in zip(files, original_paths):
            # 使用 os.path.basename() 函数来获取最内层地址
            file.filename = os.path.basename(file.filename)

            file_path = os.path.join(temp_dir, file.filename)
            # original_dir = os.path.dirname(original_path)
            try:
                file.save(file_path)

                results = predict_single_image( 's_aureus', excel_path)
                logger.info("results is" + str(results))
                #MRSA是0，MSSA是1
                if results:
                    all_results.extend(results)
                    excel_path = save_results_to_excel(all_results)
                clear_temp_directory()  # 添加清空temp文件夹的调用
            except Exception as e:
                # logger.info("file_path is" + file_path)
                # logger.info(f"File saved to {file_path} from original directory: {original_dir}")
                logger.error(f"Error saving or processing file {file.filename}: {e}")

        return jsonify(all_results)
    except Exception as e:
        logger.error(f"Error in classify_s_aureus route: {e}")
        return jsonify({"error": str(e)}), 500

    # 清空temp文件夹的函数
def clear_temp_directory():
    import shutil
    temp_dir = os.path.join('app', 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

def clear_result_directory():
    import shutil
    temp_result_dir = os.path.join('app', 'result')
    if os.path.exists(temp_result_dir):
        shutil.rmtree(temp_result_dir)
        os.makedirs(temp_result_dir)
    # try:
    #     files = request.files.getlist('file')
    #     original_paths_str = request.form.get('original_path')
    #     if original_paths_str is None:
    #         logger.error("Error: 'original_path' is not provided in the request.")
    #         return jsonify({"error": "Missing 'original_path' in the request."}), 400
    #     original_paths = original_paths_str.split(',')
    #     save_to_excel(files, original_paths)
    #     all_results = []
    #     for file, original_path in zip(files, original_paths):
    #         file_path = os.path.join(temp_dir, file.filename)
    #         logger.info(f"temp_dir is {file_path} : {temp_dir}")
    #         logger.info(f"File_name is {file_path} : {file.filename}")
    #         original_dir = os.path.dirname(original_path)
    #         excel_path = save_to_excel(files, original_paths)  # 获取保存的 Excel 文件路径
    #         try:
    #             file.save(file_path)
    #             logger.info(f"File saved to {file_path} from original directory: {original_dir}")
    #             results = predict_single_image( 's_aureus', excel_path)
    #             if results:
    #                 all_results.extend(results)
    #         except Exception as e:
    #             logger.error(f"Error saving or processing file {file.filename}: {e}")
    #     return jsonify(all_results)
    # except Exception as e:
    #     logger.error(f"Error in classify_s_aureus route: {e}")
    #     return jsonify({"error": str(e)}), 500


@app.route('/hybridaction/zybTrackerStatisticsAction', methods=['GET'])
def zybTrackerStatisticsAction():
    data = request.args.get('data')
    callback = request.args.get('__callback__')
    # 这里可以添加具体的业务逻辑
    response = {
        "message": "This is a placeholder response",
        "data": data
    }
    if callback:
        return f"{callback}({json.dumps(response)})"
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
