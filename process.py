# -*- coding: utf-8 -*-

from Tkinter import *
from PIL import ImageTk, Image
import tkMessageBox as box
import tkFileDialog
import cv, cv2 
import numpy as np 
import ImageFilter

def load_image(): 
    global im
    global name
    global save_im
    name = tkFileDialog.askopenfilename(initialdir = 'E:/Python') # 调用Win API 获得选择文件名称
    if name != '':
        if is_image(name): # 判断图片格式
            im = Image.open(name)
            show_pri_image(im)
            pri_hist_show(im)
            save_im = im
        else:
            box.showerror("ERROR", "please choose a image file")
    else:
        box.showerror("ERROR", "please choose a file")
    
def is_image(filename):
    im = Image.open(filename)
    if im.format == 'JPEG' or im.format == 'TIFF':
        return 1
    else:
        return 0

# 保存图像
def save():
    global save_im
    save_im.save('test.jpg', 'JPEG', quality=100)

# 显示原图
def show_pri_image(im):
    img = ImageTk.PhotoImage(im)
    im_label1.configure(image = img)
    im_label1.image = img

# 原图直方图
def pri_hist_show(im):
    img = hist_process(im)
    hist_label1.configure(image = img)
    hist_label1.image = img

# 显示处理后的图像
def show_image(im):
    global save_im
    save_im = im
    img = ImageTk.PhotoImage(im)
    im_label2.configure(image = img)
    im_label2.image = img

# 显示直方图            
def hist_show(im):
    img = hist_process(im)
    hist_label2.configure(image = img)
    hist_label2.image = img

# 生成直方图
hist_height = 150
def hist_process(im):
    hist = [0 for i in range(256)]
    w,h = im.size
    pix_sum = w * h
    lim = im.convert('L')
    matrix = lim.load()
    for i in range(w): #计算每个灰度的像素个数
        for j in range(h):
            hist[matrix[i,j]] += 1
    maxi = max(hist)
    hist_img = Image.new('L', (256*2, hist_height))
    hist_matrix = hist_img.load()
    for i in range(256):
        height = (hist[i] * hist_height) / maxi
        for j in range(hist_height-height):
            hist_matrix[i*2, j] = 255
            hist_matrix[i*2+1, j] = 255
    img = ImageTk.PhotoImage(hist_img)
    return img
    
    """
    # 显示直方图信息
    # 像素总数
    print u'像素总数' + str(pix_sum)
    """
    
# 采样和量化处理
def cl_process():
    try:
        w,h = im.size
        nim = Image.new('L',im.size)
        lim = im.convert('L')
        lpix = lim.load()
        npix = nim.load()
        caiyang = ['1','2','4','8','16']
        lianghua = ['256','128','64','32','16','8','4','2']
        n=int(caiyang[int(value_caiyang.get())])
        m=int(lianghua[int(value_lianghua.get())])
        for i in range(w):
            for j in range(h):
                npix[i, j] = lpix[i-i%n, j-j%n]
        for i in range(w):
            for j in range(h):
                npix[i, j] = int(npix[i, j] * m / 256) * 256 /(m-1)
        show_image(nim)
        hist_show(nim)
    except:
         box.showerror("ERROR", "Something go wrong!")

# 均衡化处理
def junhenghua():
    w,h = im.size
    pix_sum = w * h
    nim = Image.new('L', im.size)
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    p = [0 for i in range(256)]
    for i in range(w):
        for j in range(h):
            p[lpix[i,j]] += 1
    for i in range(256):
        p[i] = p[i] * 10000 / pix_sum
    max = 255;
    min = 0;
    for i in range(256):
        if p[i] == 0:
            pass
        else:
            min = i + 1
            break
    for i in range(256):
        j = 255 - i
        if p[j] == 0:
            pass
        else:
            max = j + 1
            break
    
    c = [0 for i in range(256)]
    c[0] = p[0]
    for i in range(1,256):
        c[i] = p[i] + c[i-1]
    for i in range(w):
        for j in range(h):
            npix[i, j] = c[lpix[i,j]] * (max-min) / 10000 + min
    show_image(nim)
    hist_show(nim)

# 图像线性增强
def xianxing1():
    multiple = 1.2
    nim = im.point(lambda i: i * multiple)
    show_image(nim)
    hist_show(nim)

# 图像线性减弱
def xianxing2():
    multiple = 0.8
    nim = im.point(lambda i: i * multiple)
    show_image(nim)
    hist_show(nim)

# 图像非线性变换
def feixianxing():
    nim = im.point(lambda i: (i + i * 0.8 *(255 - i) / 255))
    show_image(nim)
    hist_show(nim)

# 最临近插值法放大1.5倍
def linjinchazhi():
    multiple = 1.5
    w,h = im.size
    nw = int(multiple * w)
    nh = int(multiple * h)
    nim = Image.new('L', (nw, nh))
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(nw):
        for j in range(nh):
            x = int(i/multiple)
            y = int(j/multiple)
            npix[i, j] = lpix[x, y]
    show_image(nim)
    hist_show(nim)

# 双线性插值法放大1.5倍
def shuangxianxing():
    multiple = 1.5
    w,h = im.size
    nw = int(multiple * w)
    nh = int(multiple * h)
    nim = Image.new('L', (nw, nh))
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(nw):
        for j in range(nh):
            x = float(i)/multiple
            y = float(j)/multiple
            u = x - int(x)
            v = y - int(y)
            if int(x) == w-1 or int(y) == h-1:
                npix[i, j] = lpix[int(x),int(y)]
            else:
                npix[i, j] = (1-u)*(1-v)*lpix[int(x),int(y)] + (1-u)*v*lpix[int(x),int(y)+1] + u*(1-v)*lpix[int(x)+1,int(y)] + u*v*lpix[int(x)+1,int(y)+1]
    show_image(nim)
    hist_show(nim)

# 旋转45
def xuanzhuan():
    angle = 45
    nim = im.rotate(angle)
    show_image(nim)
    hist_show(im)

# 傅立叶变换
def FFT(image, flag = 0):
    w = image.width
    h = image.height
    iTmp = cv.CreateImage((w,h),cv.IPL_DEPTH_32F,1)
    cv.Convert(image,iTmp)
    iMat = cv.CreateMat(h,w,cv.CV_32FC2)
    mFFT = cv.CreateMat(h,w,cv.CV_32FC2)
    for i in range(h):
        for j in range(w):
            if flag == 0:
                num = -1 if (i+j)%2 == 1 else 1
            else:
                num = 1
            iMat[i,j] = (iTmp[i,j]*num,0)
    cv.DFT(iMat,mFFT,cv.CV_DXT_FORWARD)
    return mFFT

def FImage(mat):
    w = mat.cols
    h = mat.rows
    size = (w,h)
    iAdd = cv.CreateImage(size,cv.IPL_DEPTH_8U,1)
    for i in range(h):
        for j in range(w):
            iAdd[i,j] = mat[i,j][1]/h + mat[i,j][0]/h
    return iAdd

def fuliye():
    image = cv.LoadImage(name,0)
    mAfterFFT = FFT(image)
    iAfter = FImage(mAfterFFT)
    cv.ShowImage('傅立叶变换',iAfter)

# 离散余弦变换
def lisanyuxian():
    img1 = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    h, w = img1.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h, :w] = img1
    vis1 = cv2.dct(vis0)
    img2 = cv.CreateMat(vis1.shape[0], vis1.shape[1], cv.CV_32FC3)
    cv.CvtColor(cv.fromarray(vis1), img2, cv.CV_GRAY2BGR)
    cv.ShowImage('离散余弦变换', img2)

# 平滑
def pinghua1():
    imgfilted = im.filter(ImageFilter.SMOOTH);
    show_image(imgfilted)
    hist_show(imgfilted)

# 平滑（加强）
def pinghua2():
    imgfilted = im.filter(ImageFilter.SMOOTH_MORE);
    show_image(imgfilted)
    hist_show(imgfilted)

# 锐化
def ruihua():
    imgfilted = im.filter(ImageFilter.SHARPEN);
    show_image(imgfilted)
    hist_show(imgfilted)

# 哈夫曼压缩
import heapq
def huffman():
    """
    计算每种灰度的像素点的个数和概率存到data中
    data = [(0.01,'0'),(0.02,'1').....(0.005,'255')]
    """
    lim = im.convert('L')
    lpix = lim.load()
    count = [0 for i in range(256)]
    w,h = lim.size
    Sum = w * h
    for i in range(w):
        for j in range(h):
            count[lpix[i,j]] += 1
    count = map(lambda x: float(x)/float(Sum), count)
    data = []
    for i in range(0,256):
        data.append((count[i],str(i)))
    huffTree = makeHuffTree(data)
    print_buffer = list()
    encodeHuffTree(huffTree,print_buffer)
    printCode(print_buffer)
    

def makeHuffTree(symbolTupleList):
    trees = list(symbolTupleList)
    heapq.heapify(trees)
    while len(trees) > 1:  #每次合并减少一个可合并节点
        childR, childL = heapq.heappop(trees), heapq.heappop(trees) #弹出最小两个节点
        parent = (childL[0] + childR[0], childL, childR) #合并节点
        heapq.heappush(trees, parent)    #推回新节点
    return trees[0]

def encodeHuffTree(huffTree, print_buffer,prefix = ''):
    if len(huffTree) == 2:
        print_buffer.append((huffTree[1],prefix))  #加入缓冲
    else:
        encodeHuffTree(huffTree[1], print_buffer,prefix + '0') #左子树
        encodeHuffTree(huffTree[2], print_buffer,prefix + '1') #右子树

def printCode(pbuffer):
    pbuffer.sort();
    for node in pbuffer:
        print node[0]+'\t'+node[1]
    

# 基于拉普拉斯算子的边缘检测
def bianyuan():
    w,h = im.size
    nim = Image.new('L',im.size)
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    muban = [0, -1, 0, -1, 4, -1, 0, -1, 0] #拉普拉斯算子模板
    X = [0 for i in range(9)]
    for i in range(1,h-1):
        for j in range(1,w-1):
            for k in range(3):
                for l in range(3):
                    X[k*3+l] = lpix[i-1+k,j-1+l] * muban[k*3+l]
            npix[i, j] = sum(X)
    show_image(nim)
    hist_show(nim)

def canny():
    Img1 = cv.LoadImage(name,0)
    PCannyImg = cv.CreateImage(cv.GetSize(Img1), cv.IPL_DEPTH_8U, 1)
    cv.Canny(Img1, PCannyImg, 50, 150, 3)
    cv.NamedWindow("canny", 1)
    cv.ShowImage("Canny", PCannyImg)
    cv.WaitKey(0)
    cv.DestroyWindow("canny")
    
def xihua():
    def VThin(image,array):
        h = image.height
        w = image.width
        NEXT = 1
        for i in range(h):
            for j in range(w):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                    if image[i,j] == 0  and M != 0:                  
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image

    def HThin(image,array):
        h = image.height
        w = image.width
        NEXT = 1
        for j in range(w):
            for i in range(h):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1   
                    if image[i,j] == 0 and M != 0:                  
                        a = [0]*9
                        for k in range(3):
                            for l in range(3):
                                if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                    a[k*3+l] = 1
                        sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                        image[i,j] = array[sum]*255
                        if array[sum] == 1:
                            NEXT = 0
        return image

    def Xihua(image,array,num=10):
        iXihua = cv.CreateImage(cv.GetSize(image),8,1)
        cv.Copy(image,iXihua)
        for i in range(num):
            VThin(iXihua,array)
            HThin(iXihua,array)
        return iXihua

    def Two(image):
        w = image.width
        h = image.height
        size = (w,h)
        iTwo = cv.CreateImage(size,8,1)
        for i in range(h):
            for j in range(w):
                iTwo[i,j] = 0 if image[i,j] < 200 else 255
        return iTwo

    
    array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
            1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
            0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
            1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
            1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
            1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
            0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
            1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
            0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
            1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
            1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
            1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
            1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
            1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

    image = cv.LoadImage(name,0)
    iTwo = Two(image)
    iThin = Xihua(iTwo,array)
    cv.ShowImage('xihua',iThin)
    cv.WaitKey(0)

# 24位真彩色转灰度图像
def convert1():
    w,h = im.size
    nim = Image.new('L', im.size)
    pix = im.load()
    npix = nim.load()
    for i in range(w):
        for j in range(h):
            npix[i,j] = pix[i, j][0] * 0.299 + pix[i, j][1] * 0.587+ pix[i, j][2] * 0.114
    show_image(nim)
    hist_show(nim)
    
def main():
    root = Tk()
    root.title('Image')
    root.geometry("1050x680+150+0")
    
    global im_label1 # 显示原图
    global im_label2 # 显示处理后的图
    global hist_label1 # 原图直方图
    global hist_label2 # 处理后的直方图
    emp_img = ImageTk.PhotoImage(Image.new('L',(1,1)))
    im_label1 = Label(root, image = emp_img, width = 512, height = 512, justify = 'left')
    im_label1.grid(row = 1, column = 0)
    im_label2 = Label(root, image = emp_img, width = 512, height = 512, justify = 'right')
    im_label2.grid(row = 1, column = 1)

    #creat a histogram picture
    hist_img = ImageTk.PhotoImage(Image.new('L',(1,1)))
    hist_label1 = Label(root, image = hist_img, width = 512, height = 150)
    hist_label1.grid(row = 2,column = 0)
    hist_label2 = Label(root, image = hist_img, width = 512, height = 150)
    hist_label2.grid(row = 2,column = 1)

    global value_caiyang # 采样
    global value_lianghua # 量化
    
    value_caiyang = StringVar()
    value_lianghua = StringVar()
    
    # 创建滑动条
    Scale(root,
           from_ = 0,
           to = 4,
           orient = HORIZONTAL,
           variable = value_caiyang).grid(row = 0, column = 0)
    Scale(root,
           from_ = 0,
           to = 7,
           orient = HORIZONTAL,
           variable = value_lianghua).grid(row = 0,column = 1)
    
    # 创建菜单
    menubar = Menu(root)

    filemenu = Menu(menubar, tearoff = 0)
    filemenu.add_command(label = 'Open', command = lambda:load_image())
    filemenu.add_command(label = 'Save', command = lambda:save())
    menubar.add_cascade(label= 'File', menu = filemenu)
    
    menu1 = Menu(menubar, tearoff = 0)
    menu1.add_command(label ='采样和量化', command = lambda:cl_process())
    menu1.add_command(label = '均衡化', command = lambda:junhenghua())
    menu1.add_command(label = '线性增强', command = lambda:xianxing1())
    menu1.add_command(label = '线性减弱', command = lambda:xianxing2())
    menu1.add_command(label = '非线性变换', command = lambda:feixianxing())
    menubar.add_cascade(label = '点运算', menu = menu1)
    
    menu2 = Menu(menubar, tearoff = 0)
    menu2.add_command(label = '最临近插值', command = lambda:linjinchazhi())
    menu2.add_command(label = '双线性插值', command = lambda:shuangxianxing())
    menu2.add_command(label = '逆时针45度', command = lambda:xuanzhuan())
    menubar.add_cascade(label = '放大及旋转', menu = menu2)
    
    menu3 = Menu(menubar, tearoff = 0)
    menu3.add_command(label = '傅立叶变换', command = lambda:fuliye())
    menu3.add_command(label = '离散余弦变换',command = lambda:lisanyuxian())
    menubar.add_cascade(label = '图像变换', menu = menu3)

    menu4 = Menu(menubar, tearoff = 0)
    menu4.add_command(label = '平滑1', command = lambda:pinghua1())
    menu4.add_command(label = '平滑2', command = lambda:pinghua2())
    menu4.add_command(label = '锐化', command = lambda:ruihua())
    menubar.add_cascade(label = '图像增强', menu = menu4)

    menu7 = Menu(menubar, tearoff = 0)
    menu7.add_command(label = '哈夫曼编码', command = lambda:huffman())
    menubar.add_cascade(label = '压缩', menu = menu7)

    menu5 = Menu(menubar, tearoff = 0)
    menu5.add_command(label = '边缘检测(La)', command = lambda:bianyuan())
    menu5.add_command(label = '边缘检测(Canny)', command = lambda:canny())
    menu5.add_command(label = '细化', command = lambda:xihua())
    menubar.add_cascade(label = '图像分割', menu = menu5)
    root.config(menu = menubar) 

    menu6 = Menu(menubar, tearoff = 0)
    menu6.add_command(label = '24位转灰度', command = lambda:convert1())
    menubar.add_cascade(label = '灰度化', menu = menu6)

    root.mainloop()

if __name__ == '__main__':
    main()
