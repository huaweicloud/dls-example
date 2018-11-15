# 使用MoXing实现冰山图像分类应用

本文介绍如何在华为云深度学习服务平台上使用MoXing实现Kaggle竞赛中的冰山图像分类任务。实验所使用的图像为雷达图像，需要参赛者利用算法识别出图像中是冰山（iceberg）还是船（ship）。操作的流程分为4部分，分别是：

- **准备数据**：下载数据集并上传至华为云OBS桶中，编写代码将数据集格式转换成TFRecord。
- **训练模型**：使用MoXing API编写用实现冰山图像分类的网络模型，新建训练作业进行模型训练。
- **预测结果**：再次新建训练作业，对test数据集进行预测，并将结果保存到csv文件。
- **查看结果**：查看csv文件的预测结果。
### 1. 准备数据
下载冰山图像分类数据集，并上传至华为云OBS桶中，然后通过华为云深度学习服务在线IDE将数据集格式转换成TFRecord格式，操作步骤如下：


**步骤 1**  &#160; &#160; 下载数据集<a href = "https://dls-obs.obs.cn-north-1.myhwclouds.com/iceberg/iceberg_data/train.rar">train.rar</a>和<a href = "https://dls-obs.obs.cn-north-1.myhwclouds.com/iceberg/iceberg_data/test.rar">test.rar</a>，解压缩后得到train.json和test.json（该格式可以通过pandas.read_json进行读取）。

其中，训练集train.json包含4类数据：band\_1、band\_2、inc\_angle和is_iceberg（测试集），分别是：

- band\_1、band\_2：雷达图像的2个通道，分别是75x75的矩阵。
- inc_angle：雷达图拍摄角度，单位是角度。
- is_iceberg： 标注，冰山为1，船为0。

**步骤 2**  &#160; &#160; 参考<a href = "https://support.huaweicloud.com/usermanual-dls/dls_01_0040.html">“上传业务数据”</a>章节内容，将数据集train.json和test.json上传至华为云OBS桶 （假设OBS桶路径为：s3://dls-obs/iceberg/iceberg_data/）。

**步骤 3**  &#160; &#160; 参考<a href ="https://support.huaweicloud.com/usermanual-dls/dls_01_0006.html">“访问深度学习服务”</a>章节内容，登录“深度学习服务”管理控制台。

**步骤 4**  &#160; &#160; 在“开发环境管理”界面，单击“创建开发环境”，在弹出框中填写对应参数，如图1。单击“确定”，完成创建操作。

图1 创建开发环境

<img src="images/创建开发环境.PNG" width="600px" />
 

**步骤 5**  &#160; &#160; 在开发环境列表中，单击所创建开发环境右侧的“打开”，如图2，进入Jupyter Notebook文件目录界面。

图2 开发环境创建成功

<img src="images/开发环境创建成功.jpg" width="1000px" />

**步骤 6**  &#160; &#160; 单击右上角的“New”，选择“Python 2” ，进入代码开发界面。在Cell中填写数据转换代码，完整代码请参见<a href ="codes/data_format_conversion.py">data\_format_conversion.py</a>（请根据数据集实际存储位置，修改脚本代码，原代码如图3，修改后如图4）。

图3 原data_format_conversion代码

<img src="images/原data_format_conversion部分代码.jpg" width="800px" />

图4 更改后的data_format_conversion代码

<img src="images/更改后data_format_conversion部分代码.PNG" width="600px" />


**步骤 7**  &#160; &#160; 单击Cell上方的 ，运行代码。代码运行成功后，将在“s3://dls-obs/iceberg/iceberg_data/”目录下生成如下三个文件：

- iceberg-train-1176.tfrecord：训练数据集
- iceberg-eval-295.tfrecord：验证数据集
- iceberg-test-8424.tfrecord：预测数据集

### 2. 训练模型
将模型训练脚本上传至OBS桶中（您也可以在DLS的开发环境中编写模型训练脚本，并转成py文件），然后创建训练作业进行模型训练，操作步骤如下：

**步骤 1**  &#160; &#160; 参考<a href = "https://support.huaweicloud.com/usermanual-dls/dls_01_0040.html">“上传业务数据”</a>章节内容，将模型训练脚本文件<a href ="codes/train_iceberg.py">train\_iceberg.py</a>上传至华为云OBS桶 （假设OBS桶路径为：s3://dls-obs/iceberg/iceberg_code/）。

**步骤 2**  &#160; &#160; 返回“深度学习服务管理”控制台，在“训练作业管理”界面。 单击左上角的“创建训练作业”，参考图5填写参数。（其中train_url是训练输出文件路径，需要用户自定义创建路径）

图5 训练作业参数

<img src="images/训练作业参数.PNG" width="1000px" />


**步骤 3**  &#160; &#160; 单击“提交作业”，完成训练作业创建， 训练作业完成如图6.

图6 训练作业完成

<img src="images/训练作业完成.PNG" width="1000px" />


**<font color=red>注意：</font>**

**<font color=red>训练时间超过一定时间，请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。</font>**

**<font color=red>训练作业已完成，删除训练作业，以免一直占用资源。</font>**

### 3. 预测结果
待训练作业运行完成后，在“s3://dls-obs/iceberg/iceberg_log/”目录下生成模型文件。由于我们只需要进行一次预测，因此不需要部署在线预测服务。相关的预测操作已经在“train_iceberg.py”文件写好，预测结果将输出到“submission.csv”文件。我们使用训练作业进行预测，操作步骤如下：

**步骤 1**  &#160; &#160; 在“训练作业管理”界面，单击左上角的“创建训练作业”，参考图7填写参数。

图7 训练作业参数-预测
 
<img src="images/训练作业参数-预测.png" width="1000px" />


**步骤 2**  &#160; &#160; 单击“提交作业”，完成训练作业创建。

**步骤 3**  &#160; &#160; 训练作业执行完成后，在“训练作业管理”界面，单击iceberg_eval作业名称，进入作业的详情界面。在“训练日志”中，可以查看到在eval数据集上的loss值为0.271，如图8。

图8 训练作业日志

<img src="images/训练作业日志.PNG" width="1000px" />

**<font color=red>注意：</font>**

**<font color=red>训练时间超过一定时间，请及时手动停止，释放资源。否则会导致欠费，尤其对于使用GPU训练的模型项目。</font>**

**<font color=red>训练作业已完成，删除训练作业，以免一直占用资源。</font>**
 
### 4. 查看结果

在“s3://dls-obs/iceberg/iceberg_log/”目录下，能看到用于保存预测结果的“submission.csv”文件。“submission.csv”文件中的预测结果如图9。

图9 预测结果

<img src="images/预测结果.PNG" width="600px" />


