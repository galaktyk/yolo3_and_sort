
import os
import subprocess as sp
import time
import csv
class save_csv():


    def __init__(self):
        
        self.filename=str(time.strftime("%Y-%m-%d", time.localtime())+'.csv')
     
        
        if not os.path.isfile('/var/www/html/csv/'+self.filename) :  ## if file does not exist then make it!   
            
            with open('/var/www/html/csv/'+self.filename,  mode='a') as f: 
                        
                        writer=csv.writer(f)
                        writer.writerow(['Datetime','in','out','in the shop'])
                        writer.writerow([time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), 0,0,0])
                        print('Make file /var/www/html/csv/'+self.filename)
                       
                        sp.call(['tools/create_dowhtml.sh'])
 
        


    def startday(self):

        with open('/var/www/html/csv/'+self.filename, 'r') as f:
            f = f.readlines()
            lastline = f[len(f)-1]
            lastline=lastline.split(',')

            return (int(lastline[1]),int(lastline[2]))



    def save_this(self,record):

        
      
        with open('/var/www/html/csv/'+self.filename,  mode='a') as f: #append mode
            
            writer=csv.writer(f)
            writer.writerow(record)
        #print('CSV Appended')
        self.df=None #clear




if __name__ == '__main__':
   
    obj=save_csv()
    in_,out_,cust=obj.startday()
    print(in_,out_,cust)








    
