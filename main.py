from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import Qt
import sys
from stable_diffusion_pytorch import pipeline
from stable_diffusion_pytorch import model_loader
from PIL import Image
from PyQt5.QtGui import QPixmap
from style_transfer.image_transfer import get_image

## 定义DiffusionWindow类，继承QWidget
class DiffusionWindow(QWidget):
    def __init__(self, parent=None):
        super(DiffusionWindow, self).__init__(parent)
        ## 载入Ui文件
        uic.loadUi("diffusion_window.ui", self)
        ## 如果使用上传的图片则置为True
        self.use_image = False
        ## 上传图片的地址
        self.use_image_name = ''
        ## 设置固定窗口大小
        self.setFixedSize(1280, 720)
        ## 点击获取位置光标，用于文本框光标取消
        self.setFocusPolicy(Qt.StrongFocus)
        ## 绑定运行跑图函数
        self.RunButton.clicked.connect(self.sample_image)
        ## 绑定清空文本函数
        self.ClearTextButton.clicked.connect(self.clear_text)
        ## 绑定上传图片函数
        self.UploadButton.clicked.connect(self.upload_image)
        ## 绑定保存图片函数
        self.SaveImageButton.clicked.connect(self.save_image)
        ## 绑定清空图片函数（上传图）
        self.ClearUploadImageButton.clicked.connect(self.clear_upload_image)
        ## 绑定清空图片函数 （运行图）
        self.ClearImageButton.clicked.connect(self.clear_image)
        ## 跑图成功置为True
        self.have_image = False

    def upload_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png")
        jpg = QPixmap(imgName).scaled(self.UploadImageLabel.width(), self.UploadImageLabel.height())
        if imgType:
            self.UploadImageLabel.setPixmap(jpg)
            self.use_image = True
            self.use_image_name = imgName

    def sample_image(self):
        text = self.textEdit.toPlainText()
        prompts = [text]
        models = model_loader.preload_models('cuda')
        if self.use_image:
            input_images = [Image.open(self.use_image_name)]
            image = pipeline.generate(prompts, input_images=input_images, models=models, sampler="k_lms",
                                      n_inference_steps=50)
        else:
            image = pipeline.generate(prompts, models=models, sampler="k_lms", n_inference_steps=50)
        self.have_image = True
        image[0].save('output.jpg')
        pix = QPixmap('output.jpg')
        self.GeneratedImageLabel.setPixmap(pix)

    def clear_upload_image(self):
        self.UploadImageLabel.setPixmap(QPixmap(""))
        self.use_image = False

    def clear_image(self):
        self.GeneratedImageLabel.setPixmap(QPixmap(""))
        self.have_image = False

    def clear_text(self):
        self.textEdit.clear()

    def save_image(self):
        if self.have_image:
            file_path, _ = QFileDialog.getSaveFileName(self, '保存图片', "", "*.jpg;;*.png")
            if file_path:
                pixmap = self.GeneratedImageLabel.pixmap()
                pixmap.save(file_path)

class StyleTransferWindow(QWidget):
    def __init__(self, parent=None):
        super(StyleTransferWindow, self).__init__(parent)
        ## 载入Ui文件
        uic.loadUi("style_transfer_window.ui", self)
        ## 如果使用上传的图片则置为True
        self.have_content_image = False
        self.have_style_image = False
        ## 上传图片的地址
        self.content_path = ''
        self.style_path = ''
        ## 设置固定窗口大小
        self.setFixedSize(1280, 720)
        ## 绑定运行跑图函数
        self.RunButton.clicked.connect(self.sample_image)
        ## 绑定上传图片函数
        self.UploadContentButton.clicked.connect(self.upload_content_image)
        self.UploadStyleButton.clicked.connect(self.upload_style_image)
        ## 绑定保存图片函数
        self.SaveImageButton.clicked.connect(self.save_image)
        ## 绑定清空图片函数（上传图）
        self.ClearContentButton.clicked.connect(self.clear_content_image)
        self.ClearStyleButton.clicked.connect(self.clear_style_image)
        ## 绑定清空图片函数 （运行图）
        self.ClearTransferButton.clicked.connect(self.clear_image)
        ## 成功置为True
        self.have_image = False

    def upload_content_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png")
        jpg = QPixmap(imgName).scaled(self.ContentImageLabel.width(), self.ContentImageLabel.height())
        if imgType:
            self.ContentImageLabel.setPixmap(jpg)
            self.have_content_image = True
            self.content_path = imgName
    
    def upload_style_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png")
        jpg = QPixmap(imgName).scaled(self.StyleImageLabel.width(), self.StyleImageLabel.height())
        if imgType:
            self.StyleImageLabel.setPixmap(jpg)
            self.have_style_image = True
            self.style_path = imgName

    def sample_image(self):
        if self.have_image:
            self.clear_image()
        if self.have_style_image and self.have_content_image:
            img = get_image(self.content_path, self.style_path)
        img.save('./output/output.jpg')
        self.have_image = True
        pix = QPixmap('./output/output.jpg')
        self.TransferImageLabel.setPixmap(pix)

    def clear_style_image(self):
        self.StyleImageLabel.setPixmap(QPixmap(""))
        self.have_style_image = False
    
    def clear_content_image(self):
        self.ContentImageLabel.setPixmap(QPixmap(""))
        self.have_content_image = False

    def clear_image(self):
        self.TransferImageLabel.setPixmap(QPixmap(""))
        self.have_image = False

    def save_image(self):
        if self.have_image:
            file_path, _ = QFileDialog.getSaveFileName(self, '保存图片', "", "*.jpg;;*.png")
            if file_path:
                pixmap = self.TransferImageLabel.pixmap()
                pixmap.save(file_path)

## 定义主窗口函数
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        uic.loadUi("mainwindow.ui", self)
        self.setFixedSize(800, 400)

        ## 默认设备为cpu
        self.device = 'cpu'
        ## 设置cpu按钮默认绑定
        self.CPU_Button.setChecked(True)
        ## 设置diffusion按钮默认禁用
        self.Diffusion_Button.setEnabled(False)
        ## 绑定cpu和gpu按钮与设备更新函数
        self.CPU_Button.toggled.connect(self.device_update)
        self.GPU_Button.toggled.connect(self.device_update)
        ## 两个任务窗口
        self.diffusion_window = DiffusionWindow()
        self.style_transfer_window = StyleTransferWindow()
        ## 设置任务按钮与具体任务绑定
        self.Diffusion_Button.clicked.connect(self.diffusion)
        self.Style_Button.clicked.connect(self.style_transfer)

    ## 设备更新函数
    def device_update(self):
        if self.CPU_Button.isChecked():
            self.device = 'cpu'
            self.Diffusion_Button.setEnabled(False)
        else:
            self.device = 'gpu'
            self.Diffusion_Button.setEnabled(True)

    def diffusion(self):
        self.diffusion_window.setWindowModality(Qt.ApplicationModal)
        self.diffusion_window.show()

    def style_transfer(self):
        print('style_transfer')
        self.style_transfer_window.setWindowModality(Qt.ApplicationModal)
        self.style_transfer_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec()





