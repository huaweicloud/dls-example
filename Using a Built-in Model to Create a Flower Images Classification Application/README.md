# 使用预置模型实现花卉图像分类应用

本文介绍在华为云ModelArts平台如何使用flowers数据集对预置的ResNet_v1\_50模型进行重训练，快速构建花卉图像分类应用。操作步骤分为4部分，分别是：

1.	**准备数据**：下载flowers数据集，并上传至华为云对象存储服务器（OBS）中，并将数据集划分为训练集和验证集。
2.	**训练模型**：使用flowers训练集，对ResNet_v1\_50模型重训练，得到新模型。
3.	**部署模型**：将得到的模型，部署为在线预测服务。
4.	**发起预测请求**：下载客户端工程，发起预测请求获取请求结果。
### 1. 准备数据
下载flowers数据集并上传至华为云对象存储服务器（OBS）桶中，操作步骤如下：

**步骤 1** &#160; &#160; 下载并解压缩数据集压缩包“flower_photos.tgz”，flowers数据集的下载路径为：[http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

**步骤 2**&#160; &#160; 参考<a href="https://support.huaweicloud.com/usermanual-dls/dls_01_0040.html">“上传业务数据”</a>章节内容，将数据集上传至华为云OBS桶中（假设OBS桶路径为：“s3://obs-testdata/flowers_photos”）。

该路径下包含了用户训练模型需要使用的所有图像文件， 该目录下有5个子目录，代表5种类别，分别为：daisy, dandelion, roses, sunflowers, tulips。每个子目录的文件夹名称即代表该分类的label信息，每个子目录下存放对应该目录的所有图像文件，则目录结构为：

    s3://obs-testdata/flowers_photos
	    |- daisy
	       |- 01.jpg
	       |- ...
	    |- dandelion
	       |- 11.jpg
	       |- ...
	    |- roses
	       |- 21.jpg
	       |- ...
	    |- sunflowers
	       |- 31.jpg
	       |- ...
	    |- tuplis
	       |- 41.jpg
	       |- ...

**步骤 3**  &#160; &#160; 登录“ModelArts”管理控制台，在“全局配置”界面添加访问秘钥。

**步骤 4**&#160; &#160; 单击左侧导航栏的“开发环境”，在“开发环境”界面，单击“Notebook”，点击左上角的“创建”，在弹出框中，输入开发环境名称、描述、镜像类型（请选择TF-1.8.0-python27或者TF-1.8.0-python36）、实例规格、代码存储的OBS路径等参数，单击“立即创建”，完成创建操作。

**步骤 5**&#160; &#160; 在开发环境列表中，单击所创建开发环境右侧的“打开”，进入Jupyter Notebook文件目录界面。

**步骤 6**&#160; &#160; 单击右上角的“New”，选择“Python 2” ，进入代码开发界面。参见数据格式转换完整代码，在Cell中填写数据代码。

    import moxing.tensorflow as mox
    import os
    from moxing.tensorflow.datasets.raw.raw_dataset import split_image_classification_dataset

	_S3_ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID', None)                       
	_S3_SECRET_ACCESS_KEY = os.environ.get('SECRET_ACCESS_KEY', None)
	_endpoint = os.environ.get('ENDPOINT_URL', None)
	_S3_USE_HTTPS = os.environ.get('_S3_ACCESS_KEY_ID', True)
	_S3_VERIFY_SSL = os.environ.get('_S3_SECRET_ACCESS_KEY', False)
	mox.file.set_auth(ak=_S3_ACCESS_KEY_ID,sk=_S3_SECRET_ACCESS_KEY,server=_endpoint,port=None,
	                     is_secure=_S3_USE_HTTPS,ssl_verify=_S3_VERIFY_SSL)
	    
    split_image_classification_dataset(
          split_spec={'train': 0.9, 'eval': 0.1},
          src_dir='s3://obs-testdata/flower_photos',
          dst_dir='s3://obs-testdata/flowers_raw',
          overwrite=False)


**步骤 7**&#160; &#160; 单击Cell上方的运行按钮 ，运行代码。将数据集按9：1的比例划分为train和eval两部分，并输出到“s3://obs-testdata/flowers_raw”，目录结果如下所示：

    s3://obs-testdata/flowers_raw
	    |- train
		    |- daisy
		       |- 01.jpg
		       |- ...
		    |- dandelion
		       |- 11.jpg
		       |- ...
		    |- roses
		       |- 21.jpg
		       |- ...
		    |- sunflowers
		       |- 31.jpg
		       |- ...
		    |- tuplis
		       |- 41.jpg
		       |- ...
	    |- eval
		    |- daisy
		       |- 02.jpg
		       |- ...
		    |- dandelion
		       |- 12.jpg
		       |- ...
		    |- roses
		       |- 22.jpg
		       |- ...
		    |- sunflowers
		       |- 32.jpg
		       |- ...
		    |- tuplis
		       |- 42.jpg
		       |- ...

### 2. 训练模型
接下来将使用训练集对预置的ResNet_v1\_50模型进行重训练获取新的模型，操作步骤如下：

**步骤 1**&#160; &#160; 返回“ModelArts”管理控制台界面。单击左侧导航栏的“训练作业”，进入“训练作业”界面。

**步骤 2**&#160; &#160;填写参数。“名称”和“描述”可以随意填写，“数据来源”请选择“数据的存储位置”，即数据所在的父目录，在“算法/预置算法”列表中找到名称为“ResNet_v1\_50”的模型，“训练输出位置”请选择一个路径（建议新建一个文件夹）用于保存输出模型和预测文件，参数确认无误后，单击“立即创建”完成训练作业创建, 如图1。

图1 训练作业的参数配置

<img src="images/训练作业参数配置.PNG" width="800px" />

"数据集"请选择训练集和验证集所在的父目录（在本案例中，即s3://obs-testdata/flowers_raw）。

**步骤 3**&#160; &#160; 在模型训练的过程中或者完成后，通过创建TensorBoard作业查看一些参数的统计信息，如loss， accuracy等。

训练作业完成后，即完成了模型训练过程。如有问题，可点击作业名称，进入作业详情界面查看训练作业日志信息。

**步骤 4**&#160; &#160; 当训练作业运行成功后，可以在创建训练作业选择的训练输出位置OBS路径下看到新的模型文件。


### 3. 部署模型

模型训练完成后，可以创建预测作业，将模型部署为在线预测服务，操作步骤如下：

**步骤 1**  &#160; &#160; 在“模型管理”界面，单击左上角的“导入”，参考图2填写参数。名称可随意填写，“元模型来源”选择“指定元模型位置”，“选择元模型”的路径与训练模型中“训练输出位置”保持一致，“AI引擎”选择“TensorFlow”。

图2 导入模型参数配置

<img src="images/导入模型参数配置.PNG" width="800px" />


**步骤 2**  &#160; &#160; 参数确认无误后，单击“立即创建”，完成模型创建。当模型状态为“正常”时，表示创建成功。单击部署-在线服务，创建预测服务，参考图3填写参数。

当模型状态为“正常”时，表示创建成功。单击部署-在线服务，创建预测服务。

图3 部署在线服务参数配置

<img src="images/部署在线服务参数配置.PNG" width="800px" />

### 4. 发起预测请求

完成模型部署后，在部署上线-在线服务界面可以看到已上线的预测服务名称，点击进入可以进行在线预测，如图4。

图4 在线服务测试

<img src="images/在线服务测试.PNG" width="800px" />

