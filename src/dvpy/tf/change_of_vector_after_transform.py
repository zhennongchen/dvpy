import numpy as np

def change_of_vector_after_transform(original_vector,rotation,scale,padding_size,norm=2): #x=volume,y=mpr
        """In this method, we only track the change of original translation/direction vector during the 
        random transformation. norm=2 means translation vector which will be put in a normalized coordinate system. norm=1 means 
        the vector which will be normalized in the end """
        # Apply the transformation matrix to the original translation vector and put the new one into padding system
        # new vector = original * rotation * scale/(size of padding image/2)
        
        transpose=np.array([[original_vector[0]],[original_vector[1]],[original_vector[-1]],[1]])
        new_translation=np.dot(np.dot(rotation,scale),transpose)        
        final_translation=np.array([new_translation[0]/padding_size[0]*2,new_translation[1]/padding_size[1]*2,new_translation[2]/padding_size[2]*2])
        if norm !=2:
                
                n=np.array([new_translation[0],new_translation[1],new_translation[2]])
                scale=np.linalg.norm(n)
                final_translation=np.array([i/scale for i in n])
        return final_translation.reshape(3,)
    