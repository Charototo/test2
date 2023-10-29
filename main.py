# 导入临时文件模块
import tempfile

# 导入必要的Flask模块
from flask import Flask, request, render_template

# 用于加载模型和处理图像
import tensorflow as tf

# 用于数值计算
import numpy as np

# 用于处理上传的文件内容为字节流
import io

# 导入os模块
import os

# 导入Keras相关的图像处理函数
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 定义数据增强生成器
image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

# 创建一个新的Flask web应用实例
app = Flask(__name__)

# 使用TensorFlow的load_model方法从指定路径加载预训练的模型
model = tf.keras.models.load_model("D:\Coding\GCPapp\my_model.keras")


# 定义路由
@app.route('/', methods=['GET', 'POST'])
def index():
    # 检查当前的请求是否为POST方法
    if request.method == 'POST':
        # 从请求中获取名为'file'的上传文件
        file = request.files['file']

        # 检查文件是否存在
        if file:
            # 读取文件内容，并转换为BytesIO对象
            bytes_io = io.BytesIO(file.read())

            # 使用TensorFlow的load_img函数，从BytesIO对象中加载图片，并将其大小调整为224x224
            image = tf.keras.preprocessing.image.load_img(bytes_io, target_size=(180, 180))

            # 将图片对象转换为numpy数组
            data = tf.keras.preprocessing.image.img_to_array(image)

            # 扩展维度，使其成为一个4D数组，因为ImageDataGenerator.flow()需要batch形式的数据
            data = np.expand_dims(data, axis=0)

            # 使用定义的image_generator对图片进行数据增强
            ig = image_generator.flow(data)

            # 使用加载的模型对输入的图片进行预测
            predictions = model.predict(ig[0])

            if predictions[0][0] <= 0.7:
                return f'预测结果: 无肺炎, 预测值: {predictions[0][0]}'
            else:
                return f'预测结果: 肺炎，预测值：{predictions[0][0]}'

        # 如果没有文件被上传或文件不符合要求，则返回一个错误消息
        else:
            return '无效的文件格式'

            # 对于GET请求，服务器将渲染并返回upload.html模板
    return render_template('upload.html')


# 如果这个脚本是直接运行的，则启动Flask应用
if __name__ == '__main__':
    app.run(debug=True)  