import numpy as np

class device_register():
    def startframe(self,devices_box):
        self.devices_box = devices_box
        self.distant=np.array([])



    def update_person(self,person_box,person_id):
        # 1 person enter
        p_cenx,p_ceny,_,_ = person_box
        euc_thresh=np.array([None,None,None])
        for i in range(0,len(self.devices_box)):
            d_cenx,d_ceny,_,d_h = self.devices_box[i].to_xyah()
            euc = ((p_cenx-d_cenx)**2 + (p_ceny-d_ceny)**2)**0.5
            if (euc <= d_h*2) :
                euc_thresh =  np.vstack([euc_thresh,[i,d_cenx,d_ceny]]) 
            else :
                continue

        return euc_thresh[1:]





            
