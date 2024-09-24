import pickle
from tqdm import tqdm

label_list = [4, 5, 6, 7, 8, 9, 22, 23, 26, 27, 30, 33, 34, 35, 38, 39, 41, 51, 52, 53, 54, 55, 57, 58, 59, 66, 68, 69, 70, 71, 76, 79, 91, 92, 94, 95, 96, 97, 103, 105, 107, 108, 111, 113, 116, 117, 118]

D = {1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 1,
    12: 0,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1
}

f3 = open(r'/home/chenyanting/pyskl/my_mod/data/dev_hrnet.pkl','rb')
original_dict = pickle.load(f3)


# 遍历annotations字典
for i in tqdm(range(len(original_dict['annotations'])), desc = 'Frame'):
    
    label = original_dict['annotations'][i]['label']
    
        

    # 将这label的值换成自定义字典D的key对应的value        
    binary_label = D[label]
    original_dict['annotations'][i]['label'] = binary_label

print(original_dict['annotations'][13]['label'])
with open('binary_dev_hrnet.pkl', 'wb') as f:
    pickle.dump(original_dict, f)