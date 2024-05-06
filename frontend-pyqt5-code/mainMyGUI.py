import time
import cv2
from login import *
from sshtunnel import SSHTunnelForwarder
import pymysql
from login_simple_version import *
from register_simple_version import *
from MainInterfaceTest import *

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
import sys

import torch
from torch.autograd import Variable
from retinacode.models import unet
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import matplotlib
from retinacode.utils.metrics import compute_metrics
from PyQt5.QtWidgets import QMessageBox
from GenerateReport import generate_related_report
from PreprocessImage import preprocess_single_image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

# 登录界面
class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setAttribute(QtCore.AA_EnableHighDpiScaling)
        # self.ui = Ui_MainWindow_login()
        self.ui = Ui_MainWindow_login_simple_version()
        self.ui.setupUi(self)
        self.i = 0
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(self.go_to_inter)
        self.ui.pushButton_2.clicked.connect(self.go_to_register)
        # 建立连接 记得关闭连接！
        self.server = SSHTunnelForwarder(
            ssh_address_or_host=('8.130.97.173', 22),  # 指定ssh登录的跳转机的address
            ssh_username='root',  # 跳转机的用户
            ssh_password='Guofenwei521.',  # 跳转机的密码
            remote_bind_address=('localhost', 3306))
        self.server.start()
        db = 'FinalYearProject'
        self.conn = pymysql.connect(
            user="user_ali",
            passwd="root",
            host="127.0.0.1",  # 此处必须是 127.0.0.1
            db=db,
            port=self.server.local_bind_port)

        self.show()

    def go_to_inter(self):
        self.account = self.ui.lineEdit.text()
        self.password = self.ui.lineEdit_2.text()


        cursor = self.conn.cursor()
        # cursor.execute('SELECT * FROM AccountAndCode;')
        sql = "select * from AccountAndCode where account ='" + self.account + "';"
        print(sql)
        cursor.execute(sql)
        datas = cursor.fetchall()
        print(datas)
        if str(datas[0][2]) == self.password:
            print("登录成功")
            InterfaceWindow()
            self.close()
            # 关闭 各个连接
            self.server.stop()
            cursor.close()
            self.conn.close()
            self.i += 1

        if self.i == 0:
            # 如果账号密码不正确，发出警告
            print(QMessageBox.warning(self, "Warning", "Account or password is not correct！", QMessageBox.Yes))

    def go_to_register(self):
        self.register_page = InterfaceRegister()
        self.register_page.show()
        self.close()

    # 拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class InterfaceRegister(QMainWindow, Ui_MainWindow_register_simple_version):
    def __init__(self):
        # 继承(QMainWindow,Ui_MainWindow)父类的属性
        super(InterfaceRegister, self).__init__()
        # 初始化界面组件
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.register)
        self.pushButton.clicked.connect(self.back)

        # 建立连接 记得关闭连接！
        self.server = SSHTunnelForwarder(
            ssh_address_or_host=('8.130.97.173', 22),  # 指定ssh登录的跳转机的address
            ssh_username='root',  # 跳转机的用户
            ssh_password='Guofenwei521.',  # 跳转机的密码
            remote_bind_address=('localhost', 3306))
        self.server.start()
        db = 'FinalYearProject'
        self.conn = pymysql.connect(
            user="user_ali",
            passwd="root",
            host="127.0.0.1",  # 此处必须是 127.0.0.1
            db=db,
            port=self.server.local_bind_port)

    def register(self):
        # print("你点击了Register按钮")
        self.new_account = self.lineEdit.text()
        self.new_code1 = self.lineEdit_2.text()
        self.new_code2 = self.lineEdit_3.text()

        cursor = self.conn.cursor()
        sql_1 = "select * from AccountAndCode where account ='" + self.new_account + "';"
        cursor.execute(sql_1)
        datas = cursor.fetchall()

        if not datas:
            if self.new_code1 == self.new_code2:
                sql_2 = "INSERT INTO AccountAndCode (account,code) VALUES ('" + self.new_account + "','" + self.new_code2 + "');"
                # print(sql_2)
                cursor.execute(sql_2)
                self.conn.commit()
                # print("执行完毕")
                print(QMessageBox.warning(self, "Attention", "Register successfully", QMessageBox.Yes))
                self.login_window = LoginWindow()
                self.login_window.show()
                self.close()
            else:
                print(QMessageBox.warning(self, "Waring", "Different code", QMessageBox.Yes))
        else:
            print(QMessageBox.warning(self, "Waring", "Account has existed", QMessageBox.Yes))

        self.server.stop()
        cursor.close()
        self.conn.close()

    def back(self):
        self.back_login = LoginWindow()
        self.back_login.show()
        self.close()

class InterfaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # 这个路径是用于显示在四个lineEdit上的 用于全局路径的配置
        self.path_for_save_preprocessed_image = ''
        self.path_for_save_single_image = ''
        self.path_for_save_batch_pdfs = ''
        self.path_for_save_ssh = ''
        # self.train_weights = r'D:\RetainSeg\trainingrecords\checkpoint\refuge_unet_esa_grid_Lovasz\refuge_unet_esa_grid_Lovasz_best_epoch_23_0.932.pkl'
        # 初始化两个模型
        # Unet with weights of refuge
        # Unet wit weights of origa
        self.unet_refuge = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=True)
        # 下一行使用的是绝对路径 改成相对路径试试
        # self.train_weights_refuge = r'D:\RetainSeg\trainingrecords\checkpoint\refuge_unet_esa_grid_Lovasz\refuge_unet_esa_grid_Lovasz_best_epoch_23_0.932.pkl'
        self.train_weights_refuge = '../trainingrecords/checkpoint/refuge_unet_esa_grid_Lovasz/refuge_unet_esa_grid_Lovasz_best_epoch_23_0.932.pkl'
        self.ckpt_refuge = torch.load(self.train_weights_refuge)
        self.ckpt_refuge = self.ckpt_refuge['model_state_dict']
        self.new_state_dict_refuge = OrderedDict()
        for k, v in self.ckpt_refuge.items():
            # name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            self.new_state_dict_refuge[k] = v  # 新字典的key值对应的value为一一对应的值。
        self.unet_refuge.load_state_dict(self.new_state_dict_refuge)
        self.unet_refuge.eval()

        self.unet_origa = unet.UNet(num_classes=3, in_channels=3, is_esa=True, is_grid=True)
        # self.train_weights_origa = r'D:\RetainSeg\trainingrecords\checkpoint\origia_unet_esa_grid_Lovasz\origia_unet_esa_grid_Lovasz_best_epoch_19_0.908.pkl'
        self.train_weights_origa = '../trainingrecords/checkpoint/origia_unet_esa_grid_Lovasz/origia_unet_esa_grid_Lovasz_best_epoch_19_0.908.pkl'
        self.ckpt_origa = torch.load(self.train_weights_origa)
        self.ckpt_origa = self.ckpt_origa['model_state_dict']
        self.new_state_dict_origa = OrderedDict()
        for k, v in self.ckpt_origa.items():
            # name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            self.new_state_dict_origa[k] = v  # 新字典的key值对应的value为一一对应的值。
        self.unet_origa.load_state_dict(self.new_state_dict_origa)
        self.unet_origa.eval()

        self.pre_data_path_refuge = r'D:\RetainSeg\processed_data\reguge\test\image'  # refuge 本地数据集的路径
        self.pre_data_path_origa = r'D:\RetainSeg\processed_data\origa\val\image'  # origa 本地数据集的路径
        self.batch_image_files_list = []

        # 0413修改内容 添加page1的初始顺序
        # 修改page1的tab1 tab2的顺序
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.progressBar.setValue(0)
        # 修改page 的权重
        self.ui.stackedWidget.setCurrentIndex(1)
        #
        self.ui.left_predict_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.left_second_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.left_third_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.choose_image_button.clicked.connect(self.select_image)
        self.ui.predict_mask_button.clicked.connect(self.predict_original_image)
        self.ui.top_maximize_button.clicked.connect(self.resize_win)  # 注意不要有括号
        self.ui.OK.clicked.connect(self.message_has_loaded)
        # 第三个页面的按钮绑定
        self.ui.choose_image_batch_button.clicked.connect(self.choose_many_file_show_on_list)
        # self.ui.choose_save_path.clicked.connect()
        self.ui.clear_file_list.clicked.connect(self.batch_clear_file_list)
        self.ui.open_folder.clicked.connect(self.open_folder_dialog_for_PDFs)
        self.ui.generate.clicked.connect(self.batch_predict_many_images)

        # 第一个初始界面的按钮绑定
        self.ui.pushButton_preprocess_path.clicked.connect(self.path_configuration_choose_preprocess_path)
        self.ui.pushButton_single_path.clicked.connect(self.path_configuration_choose_single_predict_path)
        self.ui.pushButton_batch_path.clicked.connect(self.path_configuration_choose_batch_predict_path)
        self.ui.pushButton_ssh_path.clicked.connect(self.path_configuration_choose_ssh_path)
        self.ui.lineEdit_ssh.setText("/home/PDFuploads/")
        self.ui.choose_image_process_button.clicked.connect(self.choose_image_to_preprocess)
        # 预处理图片
        self.ui.process_button.clicked.connect(self.preprocess_image_function)
        self.ui.model_button.clicked.connect(self.show_model_introduction)
        self.ui.refuge_button.clicked.connect(self.show_refuge_introduction)
        self.ui.origa_button.clicked.connect(self.show_origa_introduction)
        self.show()

    def message_has_loaded(self):
        QMessageBox.information(self, 'Message',
                                'The checkpoints of the current model have been loaded. Please select a preprocessed image', QMessageBox.Ok,
                                QMessageBox.Ok)

    # todo 当前界面出现的时候 全程只创建2个模型(不同的权重) predict_res函数里根据界面上check box值选择相应的数据集 然后每次预测选用相应的模型
    def select_image(self):
        # 打开文件对话框，选择图片文件
        self.file_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '',
                                                        'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if self.file_name:
            pixmap = QPixmap(self.file_name)
            pixmap = pixmap.scaled(self.ui.original_img.width(), self.ui.original_img.height())
            self.ui.original_img.setPixmap(pixmap)
            self.ui.original_img.setScaledContents(True)
        self.ui.show_path_file_lineEdit_page_1.setText(self.file_name)

    # todo ok button可以改成清除按钮 弹窗事件后期可以考虑加不加

    def predict_original_image(self):
        self.ui.progressBar.setValue(5)
        if self.file_name:
            print(self.file_name)
        else:
            print("No file has been selected")

        predictions_all = []
        labels_all = []

        image_list = []
        image_list.append(self.file_name)

        if (self.ui.REFUGE_checkbox.isChecked() and (not self.ui.ORIGA_checkbox.isChecked())):
            print("Refuge_checkbox is checked")
            self.ui.progressBar.setValue(10)
            with torch.no_grad():
                i = 0
                for image in image_list:
                    predictions_of_current_image = []
                    labels_of_current_image = []
                    print(image)
                    i += 1
                    self.name = image.split("/")[-1]  # name: 'T0001.png
                    org_image = cv2.imread(image)  # ndarray 512, 512, 3 (0-255)
                    show_org_mask = cv2.imread(image.replace('/image', '/mask'))  # 512,512,3 0-255
                    self.ui.progressBar.setValue(18)
                    labels = cv2.imread(image.replace('/image', '/mask'), 0)  # 512,512 0-255
                    labels = labels / 125  # 0-2 dtype float64
                    labels = labels.astype(np.uint8)  # 0 1 2

                    image = Image.open(image)  # PngImageFile in PIL库 rbg mode 512*512
                    ori_size = image.size  # tuple (512,512)
                    image = transforms.ToTensor()(image)  # 转化成tensor (3,512,512)
                    image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)  # tensor(1,3,512,512)
                    outputs = self.unet_refuge(image)  # # [1, 2, 224, 224], 此2应该为类别数 # tensor(1,3,512,512)
                    self.ui.progressBar.setValue(30)
                    if isinstance(outputs, list):
                        # 若使用deep supervision，用最后一个输出来进行预测
                        print("outputs is list")
                        predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(
                            np.int)  # ndarray(1,512,512)
                    else:
                        # 将概率最大的类别作为预测的类别
                        print("outputs is not list")
                        predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
                    labels = labels.astype(np.int)  # int32 ndarray 512,512
                    predictions_all.append(predictions)  # 1 512 512 0-2
                    labels_all.append(labels)

                    # 0403更新
                    predictions_of_current_image.append(predictions)  # 1 512 512 0-2
                    labels_of_current_image.append(labels)
                    self.ui.progressBar.setValue(45)
                    #
                    outputs = outputs.squeeze(0)  # (3,512,512)
                    mask_original = torch.max(outputs, 0)[1].cpu().numpy()  # ndarray (512 512) 0-2

                    # 下面进行拼接展示,依次为原图，GTmask，预测的mask
                    mask_255 = mask_original * 125  # 512 512 0-250
                    mask_255 = np.expand_dims(mask_255, axis=2)  # 增加维度 (512,512,1) 0-250
                    pred_mask = np.concatenate((mask_255, mask_255, mask_255), axis=-1)  # (512,512,3) 0-250

                    cat_img = np.hstack([show_org_mask, pred_mask])
                    cv2.imwrite(self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-double-images.png", cat_img)
                    print("save double ok")
                    # np.save('mask_original', mask_original)
                    # 加载numpy数组

                    colors = [
                        [0, 0, 0, 0],  # 类别 0，黑色？
                        [255, 0, 0, 125],  # 类别 1，绿色
                        [0, 255, 0, 125],  # 类别 2，蓝色
                    ]
                    # 将语义分割图像可视化为彩色图像
                    # 隐射到这上面
                    color_image = np.zeros((512, 512, 4), dtype=np.uint8)
                    for class_id, color in enumerate(colors):
                        color_image[mask_original == class_id] = color
                    # 显示彩色可视化结果
                    plt.figure()
                    self.ui.progressBar.setValue(50)
                    rgb_mode_ori_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                    print(rgb_mode_ori_image.shape)
                    plt.imshow(rgb_mode_ori_image)
                    plt.axis('off')
                    plt.imshow(color_image)
                    print(color_image.shape)
                    # plt.show()

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

                    plt.savefig(self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-single-images.png",
                                bbox_inches='tight', pad_inches=0)
                    # gui界面上显示图片
                    # 如果没有设置过路径 那么就使用默认路径
                    if len(self.path_for_save_single_image) == 0:
                        self.path_for_save_single_image = 'D:/RetainSeg/single_predicted_image_demo'

                    file_path_for_single = self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-single-images.png"
                    file_path_for_double = self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-double-images.png"

                    print(file_path_for_single)
                    print(file_path_for_double)
                    if file_path_for_single:
                        print("singel found")
                        pixmap = QPixmap(file_path_for_single)
                        pixmap = pixmap.scaled(self.ui.predict_img.width(), self.ui.predict_img.height())
                        self.ui.progressBar.setValue(65)
                        self.ui.predict_img.setPixmap(pixmap)
                        self.ui.predict_img.setScaledContents(True)
                    if file_path_for_double:
                        print("double found")
                        pixmap = QPixmap(file_path_for_double)
                        pixmap = pixmap.scaled(self.ui.gt_mask_img.width(), self.ui.gt_mask_img.height())
                        self.ui.gt_mask_img.setPixmap(pixmap)
                        self.ui.progressBar.setValue(85)
                        self.ui.gt_mask_img.setScaledContents(True)
                    print("end")
                    # 使用混淆矩阵计算语义分割中的指标
                    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(
                        predictions_of_current_image,
                        labels_of_current_image,
                        num_classes=3)
                    # todo 定义一个函数 传入可视化的图片的路径 四个checkbox的状态 单独定义reportlab相关的函数
                    metrics_list = [dsc, iou, pc, f1]
                    self.ui.progressBar.setValue(97)
                    self.insert_row_value(metrics_list)

        elif (self.ui.ORIGA_checkbox.isChecked() and (not self.ui.REFUGE_checkbox.isChecked())):
            print("Origa_checkbox is checked")
            self.ui.progressBar.setValue(8)
            with torch.no_grad():
                i = 0
                for image in image_list:
                    predictions_of_current_image = []
                    labels_of_current_image = []
                    print(image)
                    i += 1
                    self.ui.progressBar.setValue(14)
                    self.name = image.split("/")[-1]  # name: 'T0001.png
                    org_image = cv2.imread(image)  # ndarray 512, 512, 3 (0-255)
                    show_org_mask = cv2.imread(image.replace('/image', '/mask'))  # 512,512,3 0-255

                    labels = cv2.imread(image.replace('/image', '/mask'), 0)  # 512,512 0-255
                    labels = labels / 125  # 0-2 dtype float64
                    labels = labels.astype(np.uint8)  # 0 1 2

                    image = Image.open(image)  # PngImageFile in PIL库 rbg mode 512*512
                    ori_size = image.size  # tuple (512,512)
                    image = transforms.ToTensor()(image)  # 转化成tensor (3,512,512)
                    image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)  # tensor(1,3,512,512)
                    outputs = self.unet_origa(image)  # # [1, 2, 224, 224], 此2应该为类别数 # tensor(1,3,512,512)
                    self.ui.progressBar.setValue(28)
                    if isinstance(outputs, list):
                        # 若使用deep supervision，用最后一个输出来进行预测
                        print("outputs is list")
                        predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(
                            np.int)  # ndarray(1,512,512)
                    else:
                        # 将概率最大的类别作为预测的类别
                        print("outputs is not list")
                        predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
                    labels = labels.astype(np.int)  # int32 ndarray 512,512
                    predictions_all.append(predictions)  # 1 512 512 0-2
                    labels_all.append(labels)
                    self.ui.progressBar.setValue(28)

                    #
                    predictions_of_current_image.append(predictions)  # 1 512 512 0-2
                    labels_of_current_image.append(labels)
                    #
                    outputs = outputs.squeeze(0)  # (3,512,512)
                    mask_original = torch.max(outputs, 0)[1].cpu().numpy()  # ndarray (512 512) 0-2

                    # 下面进行拼接展示,依次为原图，GTmask，预测的mask
                    mask_255 = mask_original * 125  # 512 512 0-250
                    mask_255 = np.expand_dims(mask_255, axis=2)  # 增加维度 (512,512,1) 0-250
                    pred_mask = np.concatenate((mask_255, mask_255, mask_255), axis=-1)  # (512,512,3) 0-250

                    cat_img = np.hstack([show_org_mask, pred_mask])
                    self.ui.progressBar.setValue(34)

                    if len(self.path_for_save_single_image) == 0:
                        self.path_for_save_single_image = 'D:/RetainSeg/single_predicted_image_demo'

                    cv2.imwrite(self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-double-images.png", cat_img)
                    print("save double ok")
                    # np.save('mask_original', mask_original)
                    # 加载numpy数组

                    colors = [
                        [0, 0, 0, 0],  # 类别 0，黑色？
                        [255, 0, 0, 125],  # 类别 1，绿色
                        [0, 255, 0, 125],  # 类别 2，蓝色
                    ]
                    # 将语义分割图像可视化为彩色图像
                    # 隐射到这上面
                    color_image = np.zeros((512, 512, 4), dtype=np.uint8)
                    self.ui.progressBar.setValue(48)
                    for class_id, color in enumerate(colors):
                        color_image[mask_original == class_id] = color
                    # 显示彩色可视化结果
                    plt.figure()
                    rgb_mode_ori_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                    print(rgb_mode_ori_image.shape)
                    plt.imshow(rgb_mode_ori_image)
                    plt.axis('off')
                    plt.imshow(color_image)
                    print(color_image.shape)
                    # plt.show()

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

                    plt.savefig(self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-single-images.png",
                                bbox_inches='tight', pad_inches=0)
                    # gui界面上显示图片
                    # 如果没有设置过路径 那么就使用默认路径
                    if len(self.path_for_save_single_image) == 0:
                        self.path_for_save_single_image = 'D:/RetainSeg/single_predicted_image_demo'

                    file_path_for_single = self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-single-images.png"
                    file_path_for_double = self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-double-images.png"
                    self.ui.progressBar.setValue(57)
                    print(file_path_for_single)
                    print(file_path_for_double)
                    if file_path_for_single:
                        print("singel found")
                        pixmap = QPixmap(file_path_for_single)
                        pixmap = pixmap.scaled(self.ui.predict_img.width(), self.ui.predict_img.height())
                        self.ui.progressBar.setValue(62)
                        self.ui.predict_img.setPixmap(pixmap)
                        self.ui.predict_img.setScaledContents(True)
                    if file_path_for_double:
                        print("double found")
                        pixmap = QPixmap(file_path_for_double)
                        pixmap = pixmap.scaled(self.ui.gt_mask_img.width(), self.ui.gt_mask_img.height())
                        self.ui.progressBar.setValue(87)
                        self.ui.gt_mask_img.setPixmap(pixmap)
                        self.ui.gt_mask_img.setScaledContents(True)
                    print("end")
                    # 使用混淆矩阵计算语义分割中的指标
                    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(
                        predictions_of_current_image,
                        labels_of_current_image,
                        num_classes=3)
                    # todo 定义一个函数 传入可视化的图片的路径 四个checkbox的状态 单独定义reportlab相关的函数
                    metrics_list = [dsc, iou, pc, f1]
                    self.ui.progressBar.setValue(92)
                    self.insert_row_value(metrics_list)


    def insert_row_value(self, metrics_list):
        # self.ui.tableWidget.setHorizontalHeaderLabels(['bg', 'disc', 'cup'])
        # tableWidget.setHorizontalHeaderLabels(['列1', '列2', '列3'])
        print(metrics_list)
        # todo 把metrics list 修成两个一组
        data = []
        first_row = ['Dice  ', '       ']
        data.append(first_row)
        second_row = metrics_list[0][1:]
        data.append(second_row)

        third_row = ['IOU   ', '       ']
        data.append(third_row)
        forth_row = metrics_list[1][1:]
        data.append(forth_row)

        fifth_row = ['Pc    ', '       ']
        data.append(fifth_row)
        sixth_row = metrics_list[2][1:]
        data.append(sixth_row)

        seven_row = ['F1   ', '       ']
        data.append(seven_row)
        eight_row = metrics_list[3][1:]
        data.append(eight_row)
        self.ui.progressBar.setValue(99)
        for i, row_data in enumerate(data):
            for j, cell_data in enumerate(row_data):
                if i % 2 != 0:
                    cell = QTableWidgetItem(str(cell_data)[0:5])
                    self.ui.tableWidget.setItem(i, j, cell)
                else:
                    cell = QTableWidgetItem(cell_data)
                    self.ui.tableWidget.setItem(i, j, cell)
        self.ui.progressBar.setValue(100)


    def choose_many_file_show_on_list(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if len(self.batch_image_files_list):
            self.batch_image_files_list = []
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            for file_name in file_names:
                self.batch_image_files_list.append(file_name)
                self.ui.listWidget.addItem(file_name)
        # 显示一下保存路径
        self.ui.lineEdit_show_save_path.setText(self.path_for_save_batch_pdfs)

    # todo 这个函数绑定在generate按钮上 有个文件名list 挨个预测 生成图像 然后再挨个生成对应的pdf文本 指标 直径 面积和周长
    # todo 建议封装到另一个文件里
    def batch_predict_many_images(self):
        print("已进入batch_predict_many_images")
        if (self.ui.checkBox_page3_refuge.isChecked() and (not self.ui.checkBox_page3_origa.isChecked())):
            print("checkbox_on_page3_refuge is checked")
            with torch.no_grad():
                i = 0
                for image in self.batch_image_files_list:
                    predictions_of_current_image = []
                    labels_of_current_image = []
                    image_path = image
                    i += 1
                    self.name = image.split("/")[-1]  # name: 'T0001.png
                    print(self.name)
                    # 从路径读取成图像
                    org_image = cv2.imread(image)  # ndarray 512, 512, 3 (0-255)
                    # 从路径读取mask图像
                    show_org_mask = cv2.imread(image.replace('/image', '/mask'))  # 512,512,3 0-255
                    print("labels path: " + image.replace('/image', '/mask'))
                    labels = cv2.imread(image.replace('/image', '/mask'), 0)  # 512,512 0-255
                    labels = labels / 125  # 0-2 dtype float64
                    labels = labels.astype(np.uint8)  # 0 1 2

                    image = Image.open(image)  # PngImageFile in PIL库 rbg mode 512*512
                    ori_size = image.size  # tuple (512,512)
                    image = transforms.ToTensor()(image)  # 转化成tensor (3,512,512)
                    image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)  # tensor(1,3,512,512)
                    outputs = self.unet_refuge(image)  # # [1, 2, 224, 224], 此2应该为类别数 # tensor(1,3,512,512)
                    if isinstance(outputs, list):
                        # 若使用deep supervision，用最后一个输出来进行预测
                        print("outputs is list")
                        predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(
                            np.int)  # ndarray(1,512,512)
                    else:
                        # 将概率最大的类别作为预测的类别
                        print("outputs is not list")
                        predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
                    labels = labels.astype(np.int)  # int32 ndarray 512,512
                    predictions_of_current_image.append(predictions)  # 1 512 512 0-2
                    labels_of_current_image.append(labels)
                    outputs = outputs.squeeze(0)  # (3,512,512)
                    mask_original = torch.max(outputs, 0)[1].cpu().numpy()  # ndarray (512 512) 0-2

                    # 下面进行拼接展示,依次为原图，GTmask，预测的mask
                    mask_255 = mask_original * 125  # 512 512 0-250
                    mask_255 = np.expand_dims(mask_255, axis=2)  # 增加维度 (512,512,1) 0-250
                    pred_mask = np.concatenate((mask_255, mask_255, mask_255), axis=-1)  # (512,512,3) 0-250
                    # 堆叠两张图片 然后 保存
                    cat_img = np.hstack([show_org_mask, pred_mask])
                    cv2.imwrite(self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-double-images.png", cat_img)
                    print("save double ok")

                    colors = [
                        [0, 0, 0, 0],  # 类别 0，黑色？
                        [255, 0, 0, 125],  # 类别 1，绿色
                        [0, 255, 0, 125],  # 类别 2，蓝色
                    ]
                    # 将语义分割图像可视化为彩色图像
                    # 隐射到这上面
                    color_image = np.zeros((512, 512, 4), dtype=np.uint8)
                    for class_id, color in enumerate(colors):
                        color_image[mask_original == class_id] = color
                    # 显示彩色可视化结果
                    plt.figure()
                    rgb_mode_ori_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                    print(rgb_mode_ori_image.shape)
                    plt.imshow(rgb_mode_ori_image)
                    plt.axis('off')
                    plt.imshow(color_image)
                    print(color_image.shape)
                    # plt.show()

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # 可视化单张的预测结果 然后 保存
                    plt.savefig(self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-single-images.png",
                                bbox_inches='tight', pad_inches=0)


                    if len(self.path_for_save_single_image) == 0:
                        self.path_for_save_single_image = 'D:/RetainSeg/single_predicted_image_demo'

                    file_path_for_single = self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-single-images.png"
                    file_path_for_double = self.path_for_save_single_image + '/refuge/' + str(self.name[0:-4]) + "-double-images.png"

                    if file_path_for_single:
                        print("singel found")
                        pixmap = QPixmap(file_path_for_single)
                        pixmap = pixmap.scaled(self.ui.predict_img.width(), self.ui.predict_img.height())
                        self.ui.predict_img.setPixmap(pixmap)
                        self.ui.predict_img.setScaledContents(True)
                    if file_path_for_double:
                        print("double found")
                        pixmap = QPixmap(file_path_for_double)
                        pixmap = pixmap.scaled(self.ui.gt_mask_img.width(), self.ui.gt_mask_img.height())
                        self.ui.gt_mask_img.setPixmap(pixmap)
                        self.ui.gt_mask_img.setScaledContents(True)
                    print("end")
                    # 使用混淆矩阵计算语义分割中的指标
                    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(
                        predictions_of_current_image,
                        labels_of_current_image,
                        num_classes=3)
                    # todo 定义一个函数 传入可视化的图片的路径 四个checkbox的状态 单独定义reportlab相关的函数
                    metrics_list = [dsc, iou, pc, f1]
                    print(self.ui.checkBox_metrics.isChecked(), self.ui.checkBox_diameter.isChecked(),
                          self.ui.checkBox_area.isChecked(), self.ui.checkBox_perimeter.isChecked(), )
                    # todo 传入一个保存pdf的路径
                    generate_related_report(image_path, file_path_for_single, file_path_for_double,
                                            metrics_list,
                                            self.ui.checkBox_metrics.isChecked(),
                                            self.ui.checkBox_diameter.isChecked(),
                                            self.ui.checkBox_area.isChecked(),
                                            self.ui.checkBox_perimeter.isChecked(),
                                            pred_mask, mask_original,
                                            self.path_for_save_batch_pdfs)
                    # QMessageBox.information(self, 'Message',
                    #                         'File generated',
                    #                         QMessageBox.Ok, QMessageBox.Ok)

        elif (self.ui.checkBox_page3_origa.isChecked() and (not self.ui.checkBox_page3_refuge.isChecked())):
            print("checkbox_page3_origa is checked")
            with torch.no_grad():
                i = 0
                for image in self.batch_image_files_list:
                    predictions_of_current_image = []
                    labels_of_current_image = []
                    image_path = image
                    i += 1
                    self.name = image.split("/")[-1]  # name: 'T0001.png
                    print(self.name)
                    # 从路径读取成图像
                    org_image = cv2.imread(image)  # ndarray 512, 512, 3 (0-255)
                    # 从路径读取mask图像
                    show_org_mask = cv2.imread(image.replace('/image', '/mask'))  # 512,512,3 0-255
                    print("labels path: " + image.replace('/image', '/mask'))
                    labels = cv2.imread(image.replace('/image', '/mask'), 0)  # 512,512 0-255
                    labels = labels / 125  # 0-2 dtype float64
                    labels = labels.astype(np.uint8)  # 0 1 2

                    image = Image.open(image)  # PngImageFile in PIL库 rbg mode 512*512
                    ori_size = image.size  # tuple (512,512)
                    image = transforms.ToTensor()(image)  # 转化成tensor (3,512,512)
                    image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)  # tensor(1,3,512,512)
                    outputs = self.unet_origa(image)  # # [1, 2, 224, 224], 此2应该为类别数 # tensor(1,3,512,512)
                    if isinstance(outputs, list):
                        # 若使用deep supervision，用最后一个输出来进行预测
                        print("outputs is list")
                        predictions = torch.max(outputs[-1], dim=1)[1].cpu().numpy().astype(
                            np.int)  # ndarray(1,512,512)
                    else:
                        # 将概率最大的类别作为预测的类别
                        print("outputs is not list")
                        predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
                    labels = labels.astype(np.int)  # int32 ndarray 512,512
                    predictions_of_current_image.append(predictions)  # 1 512 512 0-2
                    labels_of_current_image.append(labels)
                    outputs = outputs.squeeze(0)  # (3,512,512)
                    mask_original = torch.max(outputs, 0)[1].cpu().numpy()  # ndarray (512 512) 0-2

                    # 下面进行拼接展示,依次为原图，GTmask，预测的mask
                    mask_255 = mask_original * 125  # 512 512 0-250
                    mask_255 = np.expand_dims(mask_255, axis=2)  # 增加维度 (512,512,1) 0-250
                    pred_mask = np.concatenate((mask_255, mask_255, mask_255), axis=-1)  # (512,512,3) 0-250
                    # 堆叠两张图片 然后 保存
                    cat_img = np.hstack([show_org_mask, pred_mask])
                    cv2.imwrite(self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-double-images.png", cat_img)
                    print("save double ok")

                    colors = [
                        [0, 0, 0, 0],  # 类别 0，黑色？
                        [255, 0, 0, 125],  # 类别 1，绿色
                        [0, 255, 0, 125],  # 类别 2，蓝色
                    ]
                    # 将语义分割图像可视化为彩色图像
                    # 隐射到这上面
                    color_image = np.zeros((512, 512, 4), dtype=np.uint8)
                    for class_id, color in enumerate(colors):
                        color_image[mask_original == class_id] = color
                    # 显示彩色可视化结果
                    plt.figure()
                    rgb_mode_ori_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
                    print(rgb_mode_ori_image.shape)
                    plt.imshow(rgb_mode_ori_image)
                    plt.axis('off')
                    plt.imshow(color_image)
                    print(color_image.shape)
                    # plt.show()

                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    # 可视化单张的预测结果 然后 保存
                    plt.savefig(self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-single-images.png",
                                bbox_inches='tight', pad_inches=0)


                    if len(self.path_for_save_single_image) == 0:
                        self.path_for_save_single_image = 'D:/RetainSeg/single_predicted_image_demo'

                    file_path_for_single = self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-single-images.png"
                    file_path_for_double = self.path_for_save_single_image + '/origa/' + str(self.name[0:-4]) + "-double-images.png"

                    if file_path_for_single:
                        print("singel found")
                        pixmap = QPixmap(file_path_for_single)
                        pixmap = pixmap.scaled(self.ui.predict_img.width(), self.ui.predict_img.height())
                        self.ui.predict_img.setPixmap(pixmap)
                        self.ui.predict_img.setScaledContents(True)
                    if file_path_for_double:
                        print("double found")
                        pixmap = QPixmap(file_path_for_double)
                        pixmap = pixmap.scaled(self.ui.gt_mask_img.width(), self.ui.gt_mask_img.height())
                        self.ui.gt_mask_img.setPixmap(pixmap)
                        self.ui.gt_mask_img.setScaledContents(True)
                    print("end")
                    # 使用混淆矩阵计算语义分割中的指标
                    iou, miou, dsc, mdsc, ac, pc, mpc, se, mse, sp, msp, f1, mf1 = compute_metrics(
                        predictions_of_current_image,
                        labels_of_current_image,
                        num_classes=3)
                    # todo 定义一个函数 传入可视化的图片的路径 四个checkbox的状态 单独定义reportlab相关的函数
                    metrics_list = [dsc, iou, pc, f1]
                    print(self.ui.checkBox_metrics.isChecked(), self.ui.checkBox_diameter.isChecked(),
                          self.ui.checkBox_area.isChecked(), self.ui.checkBox_perimeter.isChecked(), )
                    # todo 传入一个保存pdf的路径
                    generate_related_report(image_path, file_path_for_single, file_path_for_double,
                                            metrics_list,
                                            self.ui.checkBox_metrics.isChecked(),
                                            self.ui.checkBox_diameter.isChecked(),
                                            self.ui.checkBox_area.isChecked(),
                                            self.ui.checkBox_perimeter.isChecked(),
                                            pred_mask, mask_original,
                                            self.path_for_save_batch_pdfs)
                    # QMessageBox.information(self, 'Message',
                    #                         'File generated',
                    #                         QMessageBox.Ok, QMessageBox.Ok)
        QMessageBox.information(self, 'Message',
                                'File generated',
                                QMessageBox.Ok, QMessageBox.Ok)

    def batch_clear_file_list(self):
        # 点击了 clear button 需要清空list里的东西 然后把视图显示一下
        if len(self.batch_image_files_list) != 0:
            self.batch_image_files_list = []
        self.ui.listWidget.clear()
        self.ui.lineEdit_show_save_path.setText('')

    def open_folder_dialog_for_PDFs(self):
        # print("进入open_folder_dialog_for_PDFs方法")
        # print(self.path_for_save_batch_pdfs)

        # QMessageBox.information(self, '消息', '这是一个信息框', QMessageBox.Ok, QMessageBox.Ok)
        # 打开文件夹
        folder_path = self.path_for_save_batch_pdfs
        # todo 删除服务器下所有的pdf文件
        # rm /path/to/file.pdf
        # rm /path/to/*.pdf
        import paramiko
        # 创建SSH连接
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname="8.130.97.173", port=22, username="root", password="Guofenwei521.")
        # 创建SFTP连接
        sftp = ssh.open_sftp()
        # 上传PDF文件
        # todo 保存的文件的地址 注意list里有很多文件 需要用for循环全部上传
        # todo 或者某个文件夹下的所有文件名称 修改成相应的地址
        file_path_list_to_delete = []
        for file_name in os.listdir(folder_path):
            # 拼接文件的绝对路径
            abs_file_path = os.path.join(folder_path, file_name)
            # 如果是文件，将其绝对路径添加到列表中
            if os.path.isfile(abs_file_path):
                file_path_list_to_delete.append(abs_file_path)

        for local_path in file_path_list_to_delete:
            local_path = local_path.replace('\\', '/')
            print(local_path)
            remote_path = "/home/PDFuploads/" + local_path.split('/')[-1]
            print(remote_path)
            try:
                sftp.put(local_path, remote_path, confirm=True)
                print("文件上传成功！")
            except Exception as e:
                print("文件上传失败，错误信息：", e)
        # local_path = "D:/TestCVZone/hello.pdf"
        # remote_path = "/home/PDFuploads/hello.pdf"
        # try:
        #     sftp.put(local_path, remote_path)
        #     print("文件上传成功！")
        # except Exception as e:
        #     print("文件上传失败，错误信息：", e)
        # 关闭连接
        sftp.close()
        ssh.close()
        QMessageBox.information(self, 'Message', 'All the pdfs has been uploaded to the cloud server (ip: 8.130.97.173)', QMessageBox.Ok, QMessageBox.Ok)
        # 点击完按钮之后打开本地的文件夹
        os.startfile(folder_path)

    def path_configuration_choose_preprocess_path(self):
        dir_choose = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if dir_choose:
            self.path_for_save_preprocessed_image = dir_choose
            self.ui.lineEdit_preprocess.setText(dir_choose)

    def path_configuration_choose_single_predict_path(self):
        dir_choose_single_predict = QFileDialog.getExistingDirectory(self,
                                                                     "选取文件夹",
                                                                     "./")  # 起始路径
        if dir_choose_single_predict:
            self.path_for_save_single_image = dir_choose_single_predict
            self.ui.lineEdit_single.setText(dir_choose_single_predict)

    def path_configuration_choose_batch_predict_path(self):
        dir_choose_batch_predict = QFileDialog.getExistingDirectory(self,
                                                                    "选取文件夹",
                                                                    "./")  # 起始路径
        if dir_choose_batch_predict:
            self.path_for_save_batch_pdfs = dir_choose_batch_predict
            self.ui.lineEdit_batch.setText(dir_choose_batch_predict)

    def path_configuration_choose_ssh_path(self):
        dir_choose_ssh = QFileDialog.getExistingDirectory(self,
                                                          "选取文件夹",
                                                          "./")  # 起始路径
        if dir_choose_ssh:
            self.path_for_save_ssh = dir_choose_ssh
            self.ui.lineEdit_ssh.setText(dir_choose_ssh)
            # self.ui.lineEdit_ssh.setText("/home/PDFuploads/")


    def choose_image_to_preprocess(self):
        # 打开文件对话框，选择图片文件
        self.choosed_image_to_preprocess_name, _ = QFileDialog.getOpenFileName(self, 'Select Image', '',
                                                        'Image Files (*.png *.jpg *.jpeg *.bmp)')

        if self.choosed_image_to_preprocess_name:
            pixmap = QPixmap(self.choosed_image_to_preprocess_name)
            pixmap = pixmap.scaled(self.ui.show_choosed_image_label.width(), self.ui.show_choosed_image_label.height())
            self.ui.show_choosed_image_label.setPixmap(pixmap)
            self.ui.show_choosed_image_label.setScaledContents(True)
            self.ui.lineEdit_show_information.setText("")

    def preprocess_image_function(self):
        preprocess_single_image(self.choosed_image_to_preprocess_name, self.path_for_save_preprocessed_image)
        self.ui.lineEdit_show_information.setText("Image preprocessing completed!")

    def show_model_introduction(self):
        img = cv2.imread('model_introduction.PNG')
        cv2.imshow('model', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_refuge_introduction(self):
        img = cv2.imread('refuge_introduction.PNG')
        cv2.imshow('refuge', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_origa_introduction(self):
        img = cv2.imread('origa_introduction.PNG')
        cv2.imshow('origa', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_win(self):
        if self.isMaximized():
            self.showNormal()
            self.ui.top_maximize_button.setIcon(QtGui.QIcon(u":/icons/icons/maxmize.png"))
        else:
            self.showMaximized()
            self.ui.top_maximize_button.setIcon(QtGui.QIcon(u":/icons/icons/minisize.png"))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))


if __name__ == '__main__':
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 解决分辨率下显示不全的问题
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())
