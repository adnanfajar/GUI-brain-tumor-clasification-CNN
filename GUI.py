

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from functools import partial
from tkinter.filedialog import askopenfile, askopenfilename
import tensorflow as tf
from keras.models import load_model, Model
import cv2
import numpy as np
from keras import backend as K
import time
from PIL import Image, ImageTk, UnidentifiedImageError
import matplotlib.pyplot as plt
from tkinter import messagebox
import imutils


root = tk.Tk()
root.geometry("900x600") 
root.title("Klasifikasi Tumor Otak")
can = Canvas(root)
can.pack(side= LEFT, fill= BOTH , expand= 1)
sb = ttk.Scrollbar(root,	orient= VERTICAL, command=can.yview)  
sb.pack(side = RIGHT, fill = Y)
can.configure(yscrollcommand = sb.set)
can.bind('<Configure>',lambda e: can.configure(scrollregion=can.bbox('all')))


# menu
# menubar = Menu(can.master)
# can.master.config(menu=menubar)
# daftar = Menu(menubar)
# daftar.add_command(label="Exit",command= )


#  Frame ###########
# judul
judul_frame = Frame(can, bd = 30)
can.create_window((300,0),window=judul_frame, anchor= "n")
# label preview gambar
judul_gbr = Frame(can, bd = 1)
can.create_window((480,100),window=judul_gbr, anchor= "n")
# frame input
input_frame = Frame(can, bd = 10, bg="gray")
can.create_window((30,120),window=input_frame, anchor= "n")
# frame gambar
gbr_frame = Frame(can, bd =1,bg="gray",width=150 ,height=100)
can.create_window((480,150),window=gbr_frame, anchor= "n")
# frame hasil test
test_frame = Frame(can, bd = 1,bg="gray")
can.create_window((480,415),window=test_frame, anchor= "n")
# frame keterangan
ket_frame = Frame(can, bd = 10,bg="gray")
can.create_window((33,350),window=ket_frame, anchor= "n")
"""    User Defined Function            """

# open a h5 file from hard-disk
def open_file(initialdir='/'):

    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Bobot Model', '*.h5' ) ]  )
    
    # dialog_var.set("Masukan bobot model yang akan digunakan")
    h5_var.set(file_path)

    return file_path

def load_weights():
    # dialog_var.set("Memasukan model.......")
    weight_path = h5_var.get()
    global model, height, width, channel
    try:
        model = load_model(weight_path)
        model.summary()

        load_input = model.input
        input_shape= list(load_input.shape)

        height = int(input_shape[1])
        width = int(input_shape[2])
        channel = int(input_shape[3])
        print(height, width, channel)
        messagebox.showinfo("Berhasil","Model Berhasil Dimasukan!")
        # dialog_var.set("Model Berhasil Dimasukan!" )
    except(OSError):
        messagebox.showerror("Terjadi Kesalahan","Model Gagal Dimaksukan!")
        # dialog_var.set("Model Gagal Dimaksukan!")

    return
# membuka file explorer gambar
def open_image(initialdir='/'):
    # load_weights()
    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Gambar Test', '*.*' ) ]  )
    # dialog_var.set("Masukan gambar yang akan dites")
    img_var.set(file_path)

    image = Image.open(file_path)
    image = image.resize((256,256)) 
    photo = ImageTk.PhotoImage(image)

    img_label = Label(gbr_frame, image=photo, padx=10, pady=10)
    img_label.image = photo # keep a reference!
    img_label.grid(row=3, column=1)

    return file_path

def load_image():
    # dialog_var.set("Memasukan gambar.............")
    path = img_var.get()
    global imgs
    gbr = cv2.imread(path)
    imgs=prepo(gbr)
    try:
        if channel == 1:
            imgs = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        else:
            imgs = cv2.imread(path)
            imgs=prepo(imgs)
        imgs = cv2.resize(imgs,(height,width)) 
        imgs = imgs.reshape(1, height, width,channel).astype('float32')
        imgs = np.array(imgs) / 255
        print("ini shape=",imgs.shape)
        
        messagebox.showinfo("Berhasil", "Gambar Berhasil Dimasukkan!\nKlik Tombol TEST !!!")
        # dialog_var.set("Gambar Berhasil Dimasukkan!, Klik Tombol TEST")
    except (NameError):
        messagebox.showerror("Terjadi Kesalahan", "Gambar Tidak Ditemukan !!!") 
        # dialog_var.set("Gambar Tidak Ditemukan !!!")
    except (cv2.error,UnidentifiedImageError):
        messagebox.showerror("Terjadi Kesalahan", "File Bukan Gambar !!!") 
        # dialog_var.set("File Bukan Gambar !!!")


    return

# #####################  Test Image
def graph():
    labels='Glioma','Meningioma', 'Tidak Ada', 'Pitutary'
    try:
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))

        ax.pie(chartp, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax.axis('equal')
        # ax.legend(title="Keterangan",
        #     loc="center left",
        #     bbox_to_anchor=(1, 0, 0.5, 1))  

        # plt.setp(autotexts, size=8, weight="bold")

        ax.set_title("Grafik Persentase Kemungkinan ")
        plt.show()
    except NameError:
        messagebox.showerror("Terjadi Kesalahan", "Harap Lakukan Test\n Terlebih Dahulu !!!")

def prepo(img):
    IMG_SIZE = (256,256)
    
    
    img = cv2.resize(
                img,
                dsize=IMG_SIZE,
                interpolation=cv2.INTER_CUBIC
            )

    # grayscale & gaussian blur
    grayy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(grayy, (3, 3), 4)

    # threshold pada citra
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # mencari tepi
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # menyimpan nilai maksimumnya
    c = max(cnts, key=cv2.contourArea)

    # mencari titik ekstrim/ ujung gambar
    pkiri = tuple(c[c[:, :, 0].argmin()][0])
    pkanan = tuple(c[c[:, :, 0].argmax()][0])
    patas = tuple(c[c[:, :, 1].argmin()][0])
    pbawah = tuple(c[c[:, :, 1].argmax()][0])

    # menggambar kontur ke citra 
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # titik tepi paling ujung
    img_pnt = cv2.circle(img_cnt.copy(), pkiri, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, pkanan, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, patas, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, pbawah, 8, (255, 255, 0), -1)

    # crop gambar
    ADD_PIXELS = 0
    new_img = img[patas[1]-ADD_PIXELS:pbawah[1]+ADD_PIXELS, pkiri[0]-ADD_PIXELS:pkanan[0]+ADD_PIXELS].copy()
    # new enhance crop
    # new_img1 = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    k= 3
    e=0
    d=0
    g=cv2.BORDER_REPLICATE
    new_img1 = cv2.GaussianBlur(new_img, (k, k), g)
    new_img1 = cv2.erode(new_img1, None, iterations=e)
    new_img1 = cv2.dilate(new_img1, None, iterations=d)
    new_img1 = cv2.resize(
                new_img1,
                dsize=(256,256),
                interpolation=cv2.INTER_CUBIC
            )
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    new_img11 = cv2.filter2D(new_img1, -1, kernel)
    # new_img2 = cv2.cvtColor(new_img11, cv2.COLOR_BGR2GRAY)
    # new_img2=(np.reshape(new_img2,(1, height, width,channel)))
    # new_img2 = new_img2.reshape(1, height, width,channel).astype('float32')
    print('shape prepo= ',new_img11.shape)
    return new_img11


    
def test_image():
    # load_image()

    # train
    try: 
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
        K.set_value(model.optimizer.lr,1e-3) # set the learning r
    except NameError:
        messagebox.showerror("Terjadi Kesalahan", "Harap Masukan Model dan Gambar\n Terlebih Dahulu !!!") 
        # dialog_var.set(" Harap Masukan Model dan Gambar\n Terlebih Dahulu !!!" )

    # predict
    old = time.time()
    pred_class = model.predict_classes(imgs)
    pred_probs = model.predict(imgs)
    new = time.time()
    print(pred_class)
    
    print("Probs = ",np.argmax(pred_probs[0]))
    waktu=new-old
    waktu = round(waktu,2)
    print("Waktu = ",waktu)
    classification = np.where(pred_probs == np.amax(pred_probs))[1][0]
    
    
    
    if pred_class==3:
        dialog =  "ini Jinak"
        lbl2 = "pituitary_tumor" 
    elif pred_class==2:
        dialog =  "ini Bersih" 
        lbl2 = "no_tumor" 
    elif pred_class==0:
        dialog =  "ini Ganas" 
        lbl2 = "glioma_tumor" 
    elif pred_class==1:
        dialog =  "ini Jinak"
        lbl2 = "meningioma_tumor"

    probn=(pred_probs[0][classification])*100
    probk=round(probn, 2)
    probs=str(probk)

    probn1=(pred_probs[0][1])*100
    probk1=round(probn1, 2)
    probs1=str(probk1)

    probn2=(pred_probs[0][2])*100
    probk2=round(probn2, 2)
    probs2=str(probk2)

    probn3=(pred_probs[0][3])*100
    probk3=round(probn3, 2)
    probs3=str(probk3)

    probn0=(pred_probs[0][0])*100
    probk0=round(probn0, 2)
    probs0=str(probk0)
    global chartp 
    chartp = [probk0,probk1,probk2,probk3]
    # chartp = np.maximum(tf.nn.softmax(pred_probs),0)
    # chartp = chartp.reshape(-1,)
    # softmax = np.amax(pred_probs)
    result_text = " : "+ lbl2 
    # hasil = "Terdeteksi sebagai : "+ lbl2 +" \nDengan Kemungkinan Sebesar = "+ probs+"%\n"+" \nGrade Tumor = "+ dialog+"\nWaktu eksekusi = "+str(waktu)+" detik"
    hasil = "Terdeteksi sebagai : "+ lbl2 +" \nDengan Kemungkinan Sebesar = "+ probs+"%\n"+"\nWaktu eksekusi = "+str(waktu)+" detik"
    test_result_var.set(result_text)
    messagebox.showinfo("showinfo", "Test Selesai")
    dialog_var.set("Tes Selesai !\n "+ hasil  )
    




# tl = Label(top_frame, text="Top frame").pack()
# ##### H5 #################
mu = Label(input_frame, font=("Courier", 15,"bold"), fg="black",bg="gray", text="Muat Model & Gambar")
mu.grid(row=1, column=2)
btn_h5_fopen = Button(input_frame, text='Telusuri Model',  command = lambda:open_file(h5_var.get()), bg="black", fg="white" )
btn_h5_fopen.grid(row=2, column=1)

h5_var = StringVar()
h5_var.set("H:/TES/UJI/POTBESTver_SALINAN_Epoch-25_batch1-128_batch2-32_MP-2x2_CK-3x3_FC-1024+512_lr-0.0001_Dataset-v3.2.2A_model.h5")
h5_entry = Entry(input_frame, textvariable=h5_var, width=40)
h5_entry.grid(row=2, column=2)

btn_h5_confirm = Button(input_frame, text='Muat Model',  command = load_weights , bg="black", fg="white" )
btn_h5_confirm.grid(row=2, column=3)
lbl_input = Label(input_frame, text=' ', bg="gray", fg="white" )
lbl_input.grid(row=3, column=4)

#######   IMAGE input
btn_img_fopen = Button(input_frame, text='Telusuri Gambar',  command =  lambda: (open_image(img_var.get())), bg="black", fg="white" )
btn_img_fopen.grid(row=7, column=1)

img_var = StringVar()
img_var.set("H:/TES/UJI/prep_test/")
img_entry = Entry(input_frame, textvariable=img_var, width=40)
img_entry.grid(row=7, column=2)

btn_img_confirm = Button(input_frame, text='Muat Gambar',  command = load_image , bg="black", fg="white" )
btn_img_confirm.grid(row=7, column=3)



ml = Label(gbr_frame, font=("Courier", 10),bg="gray", fg="white", text="Browse Image Show Below").grid(row=3, column=1)
jl = Label(judul_gbr, font=("Courier", 15,"bold","underline"), fg="black", text="Citra Yang Ditest").grid(row=1, column=1)
jd = Label(judul_frame, font=("Courier", 20,"bold"),bg="gray", fg="black", text="Klasifikasi Tumor Otak").grid(row=1, column=1)







# Test n graph butttom
btn_test = Button(input_frame, text='Test',  command = test_image , bg="green", fg="white" ,activebackground="red")
btn_test.grid(row=8, column=2)
lbl_input2 = Label(input_frame, text=' ', bg="gray", fg="white" )
lbl_input2.grid(row=9, column=2)
but = Button(input_frame, text= "Grafik", command = graph,bg="blue", fg="white" ,activebackground="red")
but.grid(row=10, column=2)

test_result_var = StringVar()
# test_result_var.set("Your result shown here")
test_result_label = Label(test_frame,font=("Courier", 10,"bold"), height=5 , textvariable=test_result_var, bg="gray", fg="black").grid(row=3, column=1)
test_result_judul = Label(test_frame,font=("Courier", 15,"bold","underline"), height=1, width = 15, text="     Hasil Test     ", bg="gray", fg="black").grid(row=1, column=1)




# Info Text
dialog_var = StringVar()
dialog_var.set(" Klasifikasi Citra Otak Untuk :\n 1. Tumor Meningioma\n2. Tumor Glioma\n3. Tumor Hipofisis(Pituitary)\n4. Tidak Ada Tumor ")
messagebox.showinfo("Selamat Datang!", "Silahkan masukan Model dan Gambar!") 

# Label Keterangan 
labelframe1 = LabelFrame(ket_frame, text="Keterangan", bg="gray")
labelframe1.pack()

toplabel = Label(labelframe1,font=("Courier", 15), height=7,width = 35, textvariable=dialog_var, fg="black", bg="lightcyan")
toplabel.pack()




input_frame.mainloop()
print("finished")
