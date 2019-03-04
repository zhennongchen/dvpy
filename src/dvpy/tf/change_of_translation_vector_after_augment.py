import numpy as np

def change_of_translation_vector_after_augment(volume_center,mpr_center,transform_matrix,padding_size): #x=volume,y=mpr
        # new_mpr_center = original * inverse_transformation_matrix
        # new_translation_vector = new_mpr_center - volume_center
        new_mpr_center=np.linalg.inv(transform_matrix).dot([mpr_center[0],mpr_center[1],mpr_center[2],1])[:3]
      

        trans_v=new_mpr_center - volume_center

        trans_v_n=np.array([trans_v[0]/padding_size[0]*2,trans_v[1]/padding_size[1]*2,trans_v[2]/padding_size[2]*2])
       
        
        
        return trans_v_n.reshape(3,)
    