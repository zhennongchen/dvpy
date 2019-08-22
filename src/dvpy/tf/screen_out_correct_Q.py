def screen_out_correct_Q(vector,R):
    if len(vector) != 4:
        print('wrong length!!\n')
    [a,b,c,d] = [vector[0],vector[1],vector[2],vector[3]]
    Eq1 = round(2*b*c - 2*a*d,3)
    Eq2 = round(2*b*d + 2*a*c,3)
    Eq3 = round(2*b*c + 2*a*d,3)
    Eq4 = round(2*c*d - 2*a*b,3)
    Eq5 = round(2*b*d - 2*a*c,3)
    Eq6 = round(2*c*d + 2*a*b,3)
    
    if (a>=0) and (Eq1 == round(R[0,1],3)) and (Eq2 == round(R[0,2],3)) and (Eq3 == round(R[1,0],3)) and (Eq4 == round(R[1,2],3)) and (Eq5 == round(R[2,0],3)) and (Eq6 == round(R[2,1],3)):
        return 1
    return 0