# main.py

import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from app.scripts import predict   # 导入 predict.py 的函数

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__,
           template_folder=os.path.join('app', 'templates'),
           static_folder=os.path.join('app', 'static'))

# 确保必要的目录存在
os.makedirs(os.path.join('app', 'temp'), exist_ok=True)
os.makedirs(os.path.join('app', 'result'), exist_ok=True)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 手动注册 predict.py 的路由
app.route('/classify_ecoli', methods=['POST'])(predict.classify_ecoli)
app.route('/download_results', methods=['GET'])(predict.download_results)

if __name__ == '__main__':
    logger.info("✅ Application starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)