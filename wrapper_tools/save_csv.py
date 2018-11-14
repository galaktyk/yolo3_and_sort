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
            record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), str(id_stay[0]).replace(', ','-'),str(id_stay[1]).replace(', ','-')]
            writer.writerow(record)

   

    def save_profile(self,d):
        with open('csv/'+self.pro_file, mode='w',newline='') as f:
            writer=csv.writer(f)


            for row_indx in list(d.keys()):
                if d[row_indx][0] != ['None']:
                    writer.writerow([row_indx,''.join(d[row_indx][0]),d[row_indx][1]])




if __name__ == '__main__':
   
    obj=save_csv()
    id_stay=[[2,11], [10,3]]
    d={1:[['male'],['laptop']],2:[['female'],['phone']],3:[['None'],['phone']]}
    #obj.save_event(id_stay)
    obj.save_profile(d)

   





    
