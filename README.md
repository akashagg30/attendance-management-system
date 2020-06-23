# attendance-management-system

I used PyQt5 to design the Gui of this application. Data about students and their attendance is stored in MySql database in tables student(roll_no int,name varchar(30)) and attendance(roll_no int,date timestamp,subject varchar(30),attend bool) respectively.
I used opencv to recognize the face and later this application generates results after fetching data from the database.

This application can show attendance of any subject marked on that day (table of students present/absent on that day) and can also calculate and show overall percentage of attendance of every subject for all the students.

code in : /lib/main.py
