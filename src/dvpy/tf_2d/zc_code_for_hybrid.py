# # this script can be used if we choose to use a hybrid approach in which the model takes slices from a small
# # number of CT studies in each step


# N = 10
# slice_number = 6

# patient_per = 5
# slice_per = 2

# batch_size = int(patient_per * slice_per)


# num_batch = int(N*slice_number / batch_size)


# num_batch_per_group = int(slice_number / slice_per)


# patient_list = np.random.permutation(N)
# array = []
# for p in patient_list:
#     slice_num = np.random.permutation(slice_number)
#     for s in slice_num:
#         array.append([p,s])
        
# new_array = []
# for B in range(0,num_batch):
#     for I in range(0,batch_size):
#         index = int(slice_number * ((int(B / num_batch_per_group) * 2)+ (int(I / slice_per))) + slice_per * int(B%num_batch_per_group)  + I%slice_per)
#         #print(index)
#         new_array.append(array[index])
# print(new_array)