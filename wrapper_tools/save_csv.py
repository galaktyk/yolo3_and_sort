import os
import time
import csv
class save_csv():


    def __init__(self):  # make file if not exist yet      
        self.event_file=str(time.strftime("%Y-%m-%d", time.localtime())+'_evets'+'.csv')
        self.pro_file=str(time.strftime("%Y-%m-%d", time.localtime())+'_prof'+'.csv')  
        if not os.path.isfile('csv/'+self.event_file) :  ## if file does not exist then make it!            
            with open('csv/'+self.event_file,  mode='a') as f:                         
                writer=csv.writer(f)            
                print('[ INFO ] Generated csv/'+self.event_file)





    def save_event(self,id_stay):
        with open('csv/'+self.event_file,  mode='a',newline='') as f: #append mode            
            writer=csv.writer(f)
            record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), str(id_stay)]
            writer.writerow(record)

   

    def save_profile(self,d):
        with open('csv/'+self.pro_file, mode='w',newline='') as f:
            writer=csv.writer(f)


            for row in d.items():     
                writer.writerow(row)




if __name__ == '__main__':
   
    obj=save_csv()
    #id_stay=[1,2,4]
    d={1:['laptop'],2:['phone','laptop']}
    
    obj.save_profile(d)

   





    
