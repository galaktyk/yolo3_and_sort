import os
import time
import csv
class save_csv():


    def __init__(self):  # make file if not exist yet      
        self.event_file=str(time.strftime("%Y-%m-%d", time.localtime())+'_evets'+'.csv')
          
        if not os.path.isfile('csv/'+self.event_file) :  ## if file does not exist then make it!            
            with open('csv/'+self.event_file,  mode='a') as f:                         
                writer=csv.writer(f)            
                print('[ INFO ] Generated csv/'+self.event_file)





    def save_event(self,id_stay):
        with open('csv/'+self.event_file,  mode='a') as f: #append mode            
            writer=csv.writer(f)
            record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), str(id_stay)]
            writer.writerow(record)
            print('[ INFO ] CSV Appended')

    def update_profile(self,_id,devices)
        with open('csv/'+self.event_file,  mode='a') as f:
            writer=csv.writer(f)
            record=[_id, str(id_stay)]
            # pro-tip search id from down to up
            writer.writerow(record)




if __name__ == '__main__':
   
    obj=save_csv()
    id_stay=[1,2,4]
    
    obj.save_event(record)

   





    
