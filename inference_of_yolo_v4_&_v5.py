from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import os
import cv2
import time
import math
import torch
import subprocess

# Define YOLOv5 model class
class YOLOv5Model:
    def __init__(self, weights):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    def detect(self, img):
        results = self.model(img)
        return results




# For the first page ***************************
root = Tk()
root.title('Traffic Sign Detection')
root.geometry('1000x490')
root.resizable(False,False)
# End of for the first page ********************

global model_number
model_number=2   # 0=yolov4;  1=yolov5;  2=not any

# Select image function------------------------
def select_image():
    global image_path
    image_path=filedialog.askopenfilename(initialdir=r"D:\tkinter\testImages",title="Select Image File",filetypes=(("JPG file","*.jpg"),("PNG file","*.png"),("All files","*.*")))
    image = Image.open(image_path)

    image = image.resize((math.floor(main_page_width*.79237),math.floor(main_page_height*.98703)))
    image = ImageTk.PhotoImage(image)
    disp.config(image=image)
    disp.image = image
# End of select image function-----------------
# Select video function------------------------
def select_video():
    global video_path
    video_path=filedialog.askopenfile(initialdir=r"D:\yolov4_test\test_images",title="Select Video File",filetypes=[('Video Files',['*.mp4','*.mov','*.mvi'])]).name
    print(video_path)
# End of select video function-----------------


# Apply_yolov4 function------------------------
def apply_yolov4():
    global frame
    os.add_dll_directory(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    class_names = []
    with open(model_path+"/obj.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # cap = cv2.VideoCapture("video.mp4")
    frame = cv2.imread(image_path)

    net = cv2.dnn.readNet(model_path+"/yolov4-custom_best.weights",
                          model_path+"/yolov4-custom_test.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    print(boxes)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[int(classid)], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 3)

    fps = "FPS: %.2f " % (1 / (end - start))
    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    frame=cv2.resize(frame,(math.floor(main_page_width*.79237),math.floor(main_page_height*.98703)))
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=ImageTk.PhotoImage(Image.fromarray(frame))
    disp.config(image=frame)
    disp.image = frame
    frame1.update()
# End of apply_yolov4 function-----------------

# apply_yolov5 function------------------------
def apply_yolov5():
    # File paths
    weights_path = model_path+'/best.pt'
    source_path = image_path

    # Define the command as a list of arguments
    command = [
        "python", "D:/tkinter/MODELS/YOLOv5/yolov5/mycode.py",
        "--source", source_path,
        "--weights", weights_path,
        "--img-size", "640",
        "--conf", "0.4",
        "--device", "cpu"
    ]

    # Execute the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Capture the output
    output, error = process.communicate()

    stdout_str = output.decode("utf-8")

    detections = stdout_str.strip()

    input_string = detections.replace('\r', '').replace('\n', '').replace('\x1b[0m', '').replace('[', '').replace(
        ']', '').replace(' ', '').replace('(', '').replace(')', '')

    elements = input_string.split(',')
    fps = float(elements[-1])
    elements = elements[:-1]

    if (elements[0] != ""):
        dets = []

        # Convert the string elements to float and split into lists of 6 elements
        for i in range(0, len(elements), 6):
            sublist = [float(x) for x in elements[i:i + 6]]
            dets.append(sublist)

        #=====================================================================================
        #=====================================================================================
        frame = cv2.imread(image_path)
        height, width = frame.shape[:2]
        COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        class_names = ['II-2', 'II-4', 'II-30-50', 'II-34', 'III-107.1-2']

        for det in dets:
            box = det[:4]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            score = float(det[4])
            classid = int(det[5])
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[int(classid)], score)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 3)

        fps = "FPS: %.2f " % fps
        cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        frame = cv2.resize(frame, (math.floor(main_page_width * .79237), math.floor(main_page_height * .98703)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        disp.config(image=frame)
        disp.image = frame
        frame1.update()

        #=====================================================================================
        #=====================================================================================

    else:
        print("No object detected")
        frame = cv2.imread(image_path)
        fps = "FPS: %.2f " % fps
        cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        frame = cv2.resize(frame, (math.floor(main_page_width * .79237), math.floor(main_page_height * .98703)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = ImageTk.PhotoImage(Image.fromarray(frame))
        disp.config(image=frame)
        disp.image = frame
        frame1.update()


    # Check if the process exited successfully
    if process.returncode == 0:
        print("YOLOv5 run successfully.")
    else:
        print("Error occured!!")



# end of apply_yolov5 function------------------------

# apply function------------------------
def apply():
    if model_number==0:
        apply_yolov4()
    if model_number==1:
        apply_yolov5()
# end of apply function------------------------


# Creating main page function +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main_page():
    # frame 1 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    global frame1
    frame1=Frame(root,width=main_page_width*.8,height=main_page_height*1,highlightbackground='red',highlightthickness=3)
    frame1.grid(row=0,column=0)

    global image
    image=Image.open(r"\images\main_page_imagge.png")
    image=image.resize((math.floor(main_page_width*.79237),math.floor(main_page_height*.98703)))
    image=ImageTk.PhotoImage(image)
    global disp
    disp = Label(frame1,image=image)
    disp.place(x=0, y=0)
    # End of frame 1 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # frame 2 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    frame2 = Frame(root, width=main_page_width*.2, height=main_page_height*1, highlightthickness=3)
    frame2.grid(row=0, column=1)

    # frame 21++++++++++++++++++++++++++++++++++++++++++++++
    frame21= Frame(frame2, width=main_page_width*.198, height=main_page_height*1, highlightbackground='black', highlightthickness=3,bg="#0C090A")
    frame21.grid(row=0, column=0,)

    text1=Label(frame21,text="Selecting File",font=30,bg="#6495ED",fg="white")
    text1.place(x=0, y=0, relwidth=1, relheight=.06)
    btn1 = Button(frame21, text="Select Image",bg="#ADDFFF", command=select_image)
    btn1.place(x=0,y=50,relwidth=1,relheight=.06)
    btn2 = Button(frame21, text="Select Video",bg="#ADDFFF", command=select_video)
    btn2.place(x=0, y=100, relwidth=1, relheight=.06)
    btn3 = Button(frame21, text="Device Camera",bg="#ADDFFF")
    btn3.place(x=0, y=150, relwidth=1, relheight=.06)

    def btn_Y4_command():
        global model_number
        model_number = 0
        global model_path
        model_path = "/MODELS/YOLOv4"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv4")
        deselect_all_buttons()
        btn_Y4.config(relief=SUNKEN)  # Highlight the selected button

    def btn_YL4_command():
        global model_number
        model_number = 0
        global model_path
        model_path = "/MODELS/YOLOv4+LSGAN"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv4+LSGAN")
        deselect_all_buttons()
        btn_YL4.config(relief=SUNKEN)  # Highlight the selected button

    def btn_YP4_command():
        global model_number
        model_number = 0
        global model_path
        model_path = "/MODELS/YOLOv4+ProGAN"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv4+ProGAN")
        deselect_all_buttons()
        btn_YP4.config(relief=SUNKEN)  # Highlight the selected button

    def btn_Y5_command():
        global model_number
        model_number = 1
        global model_path
        model_path = "/MODELS/YOLOv5"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv5")
        deselect_all_buttons()
        btn_Y5.config(relief=SUNKEN)  # Highlight the selected button

    def btn_YL5_command():
        global model_number
        model_number = 1
        global model_path
        model_path = "/MODELS/YOLOv5+LSGAN"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv5+LSGAN")
        deselect_all_buttons()
        btn_YL5.config(relief=SUNKEN)  # Highlight the selected button

    def btn_YP5_command():
        global model_number
        model_number = 1
        global model_path
        model_path = "/MODELS/YOLOv5+ProGAN"
        if model_path == '':
            text3.config(text="Not Selected")
        else:
            text3.config(text="YOLOv5+ProGAN")
        deselect_all_buttons()
        btn_YP5.config(relief=SUNKEN)  # Highlight the selected button

    def deselect_all_buttons():
        btn_Y4.config(relief=RAISED)
        btn_YL4.config(relief=RAISED)
        btn_YP4.config(relief=RAISED)
        btn_Y5.config(relief=RAISED)
        btn_YL5.config(relief=RAISED)
        btn_YP5.config(relief=RAISED)


    text2 = Label(frame21, text="Selecting Model and Weight",font=30,bg="#6495ED",fg="white")
    text2.place(x=0, y=200, relwidth=1, relheight=.06)
    btn_Y4 = Button(frame21, text="YOLOv4", bg="#ADDFFF", command=btn_Y4_command)
    btn_Y4.place(x=0, y=250, relwidth=1, relheight=.06)
    btn_YL4 = Button(frame21, text="", bg="#ADDFFF", command=btn_YL4_command)
    btn_YL4.place(x=0, y=300, relwidth=1, relheight=.06)
    btn_YP4 = Button(frame21, text="", bg="#ADDFFF", command=btn_YP4_command)
    btn_YP4.place(x=0, y=350, relwidth=1, relheight=.06)
    btn_Y5 = Button(frame21, text="", bg="#ADDFFF", command=btn_Y5_command)
    btn_Y5.place(x=0, y=400, relwidth=1, relheight=.06)
    btn_YL5 = Button(frame21, text="", bg="#ADDFFF", command=btn_YL5_command)
    btn_YL5.place(x=0, y=450, relwidth=1, relheight=.06)
    btn_YP5 = Button(frame21, text="", bg="#ADDFFF", command=btn_YP5_command)
    btn_YP5.place(x=0, y=500, relwidth=1, relheight=.06)

    text3 = Label(frame21, text="Not Selected", bg="#FFA500", fg="white")
    text3.place(x=0, y=550, relwidth=1, relheight=.06)

    btn10 = Button(frame21, text="Apply", font=35, bg="#6495ED", fg="white", command=apply)
    btn10.place(x=0, y=600, relwidth=1, relheight=.12)

    text4 = Label(frame21, text="", bg="white", fg="white")
    text4.place(x=0, y=700, relwidth=1, relheight=.1)

    # End of frame 21+++++++++++++++++++++++++++++++++++++++
    # frame 22++++++++++++++++++++++++++++++++++++++++++++++
    #frame22 = Frame(frame2, width=main_page_width * .198, height=main_page_height *.393, highlightbackground='black',
    #                highlightthickness=3)
    #frame22.grid(row=1, column=0)
    # End of frame 22+++++++++++++++++++++++++++++++++++++++

    # End of frame 2 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# End of creating main page function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Creating go to first page to main page function +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def goto_main_page():
    for item in root.winfo_children():
        item.destroy()

    # Geometry for the main page *****************************
    root.title('Traffic Sign Detection')
    full_width=root.winfo_screenwidth()
    full_height=root.winfo_screenheight()
    global main_page_width
    main_page_width = full_width-10
    global main_page_height
    main_page_height=full_height-10
    root.geometry('%dx%d+0+0'%(main_page_width,main_page_height))
    root.resizable(True,True)
    # End of geometry for the main page ***********************

    # Calling main page==============================
    main_page()
    # End of calling main page==============================

    root.mainloop()
# End of Creating go to first page to main page function +++++++++++++++++++++++++++++++++++++++++++++++++




#Creating first page---------------------------------------------------
firstPageImage=ImageTk.PhotoImage(file=r"D:\tkinter\images\first_page_image.png")
fi_label=Label(root,image=firstPageImage)
fi_label.place(x=0,y=0,relwidth=1,relheight=1)

button=Button(root,text="Detect Traffic Sign",font=50,bg='white',fg='skyblue',command=goto_main_page)
button.pack(pady=15)
#End of creating first page---------------------------------------------------

root.mainloop()