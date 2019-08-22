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
    R1 = round(R[0,1],3)
    R2 = round(R[0,2],3)
    R3 = round(R[1,0],3)
    R4 = round(R[1,2],3)
    R5 = round(R[2,0],3)
    R6 = round(R[2,1],3)
    #print(vector,(Eq1,R1),(Eq2,R2),(Eq3,R3),(Eq4,R4),(Eq5,R5),(Eq6,R6))
    if (a>=0) and Eq1==R1 and Eq2==R2 and Eq3==R3 and Eq4==R4 and Eq5==R5 and Eq6==R6:
        return 1
    return 0