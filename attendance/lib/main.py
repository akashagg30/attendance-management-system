import cv2
import numpy as np
from PIL import Image
import os
import mysql.connector
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QStackedWidget,QFormLayout,QLineEdit,QComboBox,QTableWidget,QTableWidgetItem


class Backend:
    data='../assets/images/'
    cascade='../assets/files/haarcascade_frontalface_default.xml'
    trained='../assets/files/trainer.yml'
    mydb=mysql.connector.connect(
            host="localhost",
            user="ghost38o",
            password="turniton",
            auth_plugin="mysql_native_password",
             database="mydatabase"
            )
    ids=dict()
    ids[0]="unknown"
    mycursor=mydb.cursor()
    
    def populate(self):
       self.mycursor.execute("select * from students")
       data=self.mycursor.fetchall()
       for x in data:
           self.ids[x[0]]=x[1]
       print(self.ids)
        
        
    def takeData(self,id):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        face_detector = cv2.CascadeClassifier(
            self.cascade)
        # For each person, enter one numeric face id
        face_id = id
        # Initialize individual sampling face count
        count = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1  
                # Save the captured image into the datasets folder
                cv2.imwrite(self.data+"/User." + str(face_id) + '.' + str(count) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('image', img)
                print (count)
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        self.trainer()
        
    # function to get the images and label data
    def getImagesAndLabels(self,path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids    
        
    def trainer(self):
        path = self.data
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(self.cascade);
    
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = self.getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        # Save the model into trainer/trainer.yml
        recognizer.save(self.trained)
        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    def recognizer(self,subject):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(self.trained)
        cascadePath = self.cascade
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        id = 0
        flag=1
        ids=[]
        cam = cv2.VideoCapture(0)
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                if not id in ids:
                    ids.append(id)
                if (confidence < 100):
                   id = self.ids[id]
                   confidence = "  {0}%".format(round(100 - confidence))
                else:
                   id = "unknown"
                   confidence = "  {0}%".format(round(100 - confidence))
    
                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            cv2.imshow('camera', img)
            k = cv2.waitKey(5) & 0xff  # Press 'ESC' for exiting video
            if flag:
                prev=[]
                prev.append(id)
                print(id)
                flag=False
            else:
                if not(id in prev):
                    prev.append(id)
            if k == 27:
                break
            
        for x in self.ids:
            if x in ids:
                present=1
            else:
                present=0
            if not x==0:
                cmd="insert into attendance(roll_no,subject,attend) values(%s,'%s',%s);"%(''.join(str(x)),''.join(subject),''.join(str(present)))
                print(cmd)
                self.mycursor.execute(cmd)
                self.mydb.commit()
                    
            
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
  
    
      
class Gui(Backend):
    def __init__(self):
        self.populate()
        self.app=QApplication([])
        self.window=QWidget()
        self.window.setWindowTitle("Attendance Management")
        
        self.main=QVBoxLayout()
        
        
        #navigation
        self.nav=QWidget()
        self.build_nav()
        self.main.addWidget(self.nav)
        
        #stack for page views
        self.stack=QStackedWidget()
        
        self.createUser=QWidget()
        self.signUp()
        
        self.result=QWidget()
        self.showResult()
        
        self.attendance=QWidget()
        self.buildAttendance()
        
        self.printResult=QWidget()
        self.print_result()
        
        self.stack.addWidget(self.createUser)
        self.stack.addWidget(self.result)
        self.stack.addWidget(self.attendance)
        self.stack.addWidget(self.printResult)
        
        self.main.addWidget(self.stack)
        
        self.window.setLayout(self.main)
        self.window.show()
        self.app.exec_()

    def build_nav(self):
        layout=QHBoxLayout()
        data=QPushButton('add new entry')
        data.clicked.connect(self.toData)
        result=QPushButton('Result')
        result.clicked.connect(self.toResult)
        attendance=QPushButton('take attendance')
        attendance.clicked.connect(self.takeAttendance)
        layout.addWidget(data)
        layout.addWidget(attendance)
        layout.addWidget(result)
        self.nav.setLayout(layout)
        
    def signUp(self):
        layout=QFormLayout()
        self.name=QLineEdit()
        layout.addRow("Name",self.name)
        self.rno=QLineEdit()
        layout.addRow("Roll Number",self.rno)
        submit=QPushButton('Submit')
        layout.addWidget(submit)
        submit.clicked.connect(self.insert)
        self.createUser.setLayout(layout)
        
    def buildAttendance(self):
        layout=QFormLayout()
        self.subject=QLineEdit()
        layout.addRow("subject",self.subject)
        submit=QPushButton('Submit')
        layout.addWidget(submit)
        submit.clicked.connect(self.fun)
        self.attendance.setLayout(layout)
            
    def showResult(self):
        layout=QHBoxLayout()
        self.subjectList=self.fetchSubjects()
        self.subjects=QComboBox()
        self.subjects.addItems(self.subjectList)
        result=QPushButton("show today's attendance")
        result.clicked.connect(self.today)
        overall=QPushButton("show overall attendance")
        overall.clicked.connect(self.overall)
        layout.addWidget(self.subjects)
        layout.addWidget(result)
        layout.addWidget(overall)
        self.result.setLayout(layout)
        
    def print_result(self):
        layout=QVBoxLayout()
        self.table=QTableWidget()
        layout.addWidget(self.table)
        self.printResult.setLayout(layout)
      
        
    def insert(self):
        name=self.name.text()
        rno=self.rno.text()
        cmd="insert into students(roll_no,name) values('%s','%s');"%(''.join(rno),''.join(name))
        self.mycursor.execute(cmd)
        self.mydb.commit()
        print(self.mycursor.rowcount, "record inserted.")
        print(cmd)
        self.populate()
        self.takeData(int(rno,10))
        
    def fetchSubjects(self):
        cmd="select distinct(subject) from attendance"
        self.mycursor.execute(cmd)
        data=self.mycursor.fetchall()
        data=[x[0] for x in data]
        return data
        
    def today(self):
        subject=self.subjects.currentText()
        cmd="select roll_no from attendance where date(date)=curdate() and subject='"+subject+"' and attend=1"
        self.mycursor.execute(cmd)
        data=self.mycursor.fetchall()
        data=[x[0] for x in data]
        
        self.table.setRowCount(len(self.ids)-1)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Roll Number','Name','Status'])
        x=0
        for y in self.ids:
            if y==0:
                continue
            self.table.setItem(x,0,QTableWidgetItem(str(y)))
            self.table.setItem(x,1,QTableWidgetItem(self.ids[y]))
            status="Absent"
            if y in data:
                status="Present"
            self.table.setItem(x,2,QTableWidgetItem(status))
            x+=1
        self.stack.setCurrentIndex(3)
        
    def overall(self):
        self.mycursor.execute("select count(distinct(date(date))) from attendance")
        totalDays=self.mycursor.fetchall()
        totalDays=totalDays[0][0]
        self.table.setRowCount(len(self.ids)-1)
        self.table.setColumnCount(len(self.subjectList)+2)
        self.table.setHorizontalHeaderLabels(['Roll Number','Name']+self.subjectList)
        x=0
        for y in self.ids:
            if y==0:
                continue
            self.table.setItem(x,0,QTableWidgetItem(str(y)))
            self.table.setItem(x,1,QTableWidgetItem(self.ids[y]))
            for i in range(len(self.subjectList)):
                pct=str(self.calculate(y,self.subjectList[i])*100/totalDays)+"%"
                print(y,self.subjectList[i],pct)
                self.table.setItem(x,2+i,QTableWidgetItem(pct))
            x+=1
        self.stack.setCurrentIndex(3)
            
            
    def calculate(self,i,subject):
        cmd="select count(*) from attendance where roll_no="+str(i)+" and subject='"+subject+"' and attend=1;"
        self.mycursor.execute(cmd)
        data=self.mycursor.fetchall()
        return data[0][0]
        
        
    def takeAttendance(self):
        self.stack.setCurrentIndex(2)    
    
    def toData(self):
        self.stack.setCurrentIndex(0)
        
    def toResult(self):
        self.stack.setCurrentIndex(1)   
    
    def fun(self,subject):
        self.recognizer(self.subject.text())
        
        
        
        
        
Gui()
