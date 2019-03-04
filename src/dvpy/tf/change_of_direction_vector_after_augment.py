import numpy as np

def change_of_direction_vector_after_augment(ori_vector,rotation,scale): 
    # new_vector = original * rotation * scale
    ori_1=np.array([ori_vector[0],ori_vector[1],ori_vector[2],1])
    m=np.dot(rotation,scale)
    vector_aug=np.dot(ori_1,m)[:3]
    
    #normalization to length=1
    scale=np.linalg.norm(vector_aug)
    final_v=np.array([i/scale for i in vector_aug])
        
        
    return final_v.reshape(3,)