# -*- coding: utf-8 -*-
from Tkinter import *
from PIL import ImageTk, Image
##import win32ui
import tkMessageBox as box
import tkFileDialog

def load_image():
    global im
    name = tkFileDialog.askopenfilename(initialdir = 'E:/Python')
    if name != '':
        if is_image(name):
            im = Image.open(name)
            show_pri_image(im)
            pri_hist_show(im)
        else:
            box.showerror("ERROR", "please choose a image file")
    else:
        box.showerror("ERROR", "please choose a file")
    
##def get_filename():
##    dlg = win32ui.CreateFileDialog(1)
##    dlg.SetOFNInitialDir('C:\Users\liam\Desktop')
##    dlg.DoModal()
##    name = dlg.GetPathName()
##    return name
    
def is_image(filename):
    im = Image.open(filename)
    if im.format == 'JPEG' or im.format == 'BMP' or im.fortmat == 'GIF' or im.format == 'TIFF':
        return 1
    else:
        return 0

def show_pri_image(im):
    img = ImageTk.PhotoImage(im)
    im_label1.configure(image = img)
    im_label1.image = img

def pri_hist_show(im):
    hist = hist_process(im)
    maxi = max(hist)
    hist_img = Image.new('L', (256*2, 200))
    hist_matrix = hist_img.load()
    for i in range(256):
        height = (hist[i] * 200) / maxi
        for j in range(200-height):
            hist_matrix[i*2, j] = 255
            hist_matrix[i*2+1, j] = 255
    img = ImageTk.PhotoImage(hist_img)
    hist_label1.configure(image = img)
    hist_label1.image = img

def show_image(im):
    img = ImageTk.PhotoImage(im)
    im_label2.configure(image = img)
    im_label2.image = img

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

def junhenghua():
    w,h = im.size
    pix_sum = w * h
    nim = Image.new('L', im.size)
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    p = []
    for i in range(256):
        p.append(0)
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
    
    c = []
    for i in range(256):
        c.append(0)
    c[0] = p[0]
    for i in range(1,256):
        c[i] = p[i] + c[i-1]
    for i in range(w):
        for j in range(h):
            npix[i, j] = c[lpix[i,j]] * (max-min) / 10000 + min
    show_image(nim)
    hist_show(nim)

def xianxing():
    w,h = im.size
    nim = Image.new('L', im.size)
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(w):
        for j in range(h):
            npix[i, j] = lpix[i, j] * 1.2
    show_image(nim)
    hist_show(nim)

def feixianxing():
    w,h = im.size
    nim = Image.new('L', im.size)
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(w):
        for j in range(h):
            x = lpix[i, j]
            npix[i, j] = x + x * 0.8 *(255 - x) / 255
    show_image(nim)
    hist_show(nim)

def linjinchazhi():
    w,h = im.size
    nw = int(1.5 * w)
    nh = int(1.5 * h)
    nim = Image.new('L', (nw, nh))
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(nw):
        for j in range(nh):
            x = int(i*2/3)
            y = int(j*2/3)
            npix[i, j] = lpix[x, y]
    show_image(nim)
    hist_show(nim)

def shuangxianxing():
    w,h = im.size
    nw = int(1.5 * w)
    nh = int(1.5 * h)
    nim = Image.new('L', (nw, nh))
    lim = im.convert('L')
    lpix = lim.load()
    npix = nim.load()
    for i in range(nw):
        for j in range(nh):
            x = float(i)*2/3
            y = float(j)*2/3
            u = x - int(x)
            v = y - int(y)
            if int(x) == w-1 or int(y) == h-1:
                npix[i, j] = lpix[int(x),int(y)]
            else:
                npix[i, j] = (1-u)*(1-v)*lpix[int(x),int(y)] + (1-u)*v*lpix[int(x),int(y)+1] + u*(1-v)*lpix[int(x)+1,int(y)] + u*v*lpix[int(x)+1,int(y)+1]
    show_image(nim)
    hist_show(nim)

def xuanzhuan():
    lim = im.convert('L')
    nim = lim.rotate(45)
    show_image(nim)
    hist_show(lim)

def hist_process(im):
    hist = []
    for i in range(256):
        hist.append(0)
    w,h = im.size
    im = im.convert('L')
    matrix = im.load()
    for i in range(w):
        for j in range(h):
            hist[matrix[i,j]] += 1
    return hist
            
def hist_show(im):
    hist = hist_process(im)
    maxi = max(hist)
    hist_img = Image.new('L', (256*2, 200))
    hist_matrix = hist_img.load()
    for i in range(256):
        height = (hist[i] * 200) / maxi
        for j in range(200-height):
            hist_matrix[i*2, j] = 255
            hist_matrix[i*2+1, j] = 255
    img = ImageTk.PhotoImage(hist_img)
    hist_label2.configure(image = img)
    hist_label2.image = img
    
def main():
    root = Tk()
    root.title('Image')
    root.geometry("1350x680+0+0")
    
    global im_label1
    global im_label2
    global hist_label1
    global hist_label2
    emp_img = ImageTk.PhotoImage(Image.new('L',(1,1)))
    im_label1 = Label(root, image = emp_img, width = 512, height = 384, justify = 'left')
    im_label1.grid(row = 0, column = 0)
    im_label2 = Label(root, image = emp_img, width = 512, height = 384, justify = 'right')
    im_label2.grid(row = 0, column = 1)

    #creat a histogram picture
    hist_img = ImageTk.PhotoImage(Image.new('L',(1,1)))
    hist_label1 = Label(root, image = hist_img, width = 512, height = 200)
    hist_label1.grid(row = 1,column = 0)
    hist_label2 = Label(root, image = hist_img, width = 512, height = 200)
    hist_label2.grid(row = 1,column = 1)

    global value_caiyang 
    global value_lianghua
    
    value_caiyang = StringVar()
    value_lianghua = StringVar()
  
    Scale(root,
           from_ = 0,
           to = 4,
           orient = HORIZONTAL,
           variable = value_caiyang).grid(row = 2, column = 0)
    Scale(root,
           from_ = 0,
           to = 7,
           orient = HORIZONTAL,
           variable = value_lianghua).grid(row = 2,column = 1)

    btn1 = Button(text=u'打开', command=lambda:load_image())
    btn1.grid(row = 3, column = 0)
    btn2 = Button(text=u'采样和量化', command=lambda:cl_process())
    btn2.grid(row = 3, column = 1)
    btn3 = Button(text=u'均衡化', command=lambda:junhenghua())
    btn3.grid(row = 2, column = 2)
    btn4 = Button(text=u'线性变换', command=lambda:xianxing())
    btn4.grid(row = 2, column = 3)
    btn5 = Button(text=u'非线性变换', command=lambda:feixianxing())
    btn5.grid(row = 2, column = 4)
    btn6 = Button(text=u'最邻近插值', command=lambda:linjinchazhi())
    btn6.grid(row = 3, column = 2)
    btn7 = Button(text=u'双线性插值', command=lambda:shuangxianxing())
    btn7.grid(row = 3, column = 3)
    btn8 = Button(text=u'逆时针45度', command=lambda:xuanzhuan())
    btn8.grid(row = 3, column = 4)
    root.mainloop()
    
if __name__ == '__main__':
    main()
