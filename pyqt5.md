- mianinterfacetest.ui布局设置 如何使stackwidget充满布局
- stackwidget右键布局 水平布局
- todo: 记得重构项目的结构 要不打包成软件绝对路径恐怕会出问题
- widgt组件变型为frame 再变回去就可以实现widgt的操作
- 右面组件大小分配不了的解决方案 垂直策略改成fixed
- 初始界面的话 分三块 基础的数据管理 第一块本地原始图像 第二块sql原始图像
- 第二个页面 加载预测
- 第三个页面 可视化或者程序说明？
- 如何解决原始图片 原图进来的问题
- pixmap = QPixmap(self.file_name)
- pixmap = pixmap.scaled(self.ui.original_img.width(), self.ui.original_img.height())
- stackedwidget的组件page3 page2 上有个小禁止符号 这时候 选择page上的鼠标右键 设置布局为水平布局
- page2的路径配置界面 只使用水平竖直布局 预览会变形 所以将外层布局更改为qwidget然后布局为栅格布局
- ![img.png](img.png)
- 预处理 refuge image:resize mask resize and change to png
- 用于演示的代码 最多十张 都是test 或者val
- 单张预测的功能
- single_predicted_image_demo 下面origa refuge 
- 每个数据集的每张图片对应single image and double image
- 批量生成的图片 实际上还是走单张预测的保存路径 只是生成的pdf可以选择保存的文件夹路径
- original_test_image_demo
- preprocessed_image_demo
- single_predicted_image_demo
- batch_image_PDFs_demo
- 注意默认路径的使用 三个路径 需要用if设置默认路径
- todolist 0413任务
- 第一个页面的三个介绍图片的插入 已完成
- 第一二个页面的切换 已完成
- 第三个页面的batch_list清除 已完成
- 第三个页面的sample button 改成打开文件夹 已完成
- 考虑给预测加一个进度条！！！！ 未完成 涉及回调
- ssh数据库连接

