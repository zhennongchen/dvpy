import numpy as np

def change_of_translation_after_transform(original_translation,rotation,scale,padding_size): #x=volume,y=mpr
        """In this method, we only track the change of original translation vector during the 
        random transformation. """
        # Apply the transformation matrix to the original translation vector and put the new one into padding system
        # new vector = original * rotation * scale/(size of padding image/2)
        
        transpose=np.array([[original_translation[0]],[original_translation[1]],[original_translation[-1]],[1]])
        new_translation=np.dot(np.dot(rotation,scale),transpose)        
        new_translation=np.array([new_translation[0]/padding_size[0]*2,new_translation[1]/padding_size[1]*2,new_translation[2]/padding_size[2]*2])
        return new_translation.reshape(3,)
    