# coding:utf-8
# 题目要求：
# 写一个函数检验数独是否完成：
# 如果完成，返回 “Finished!”
# 如果未完成，返回 “Try again!”
#
# 数独规则
# 数独为9行9列。
# 每一行和每一列均由 [1-9] 9个不重复数字组成。
# 将 9行x9列 的数独分割为9个小区域，每个区域3行3列，且保证每个小区域数字也是从[1-9] 9 个不重复数组成。
import numpy as np
def sudu_or_not(aboard):
    board =np.array(aboard)
 #取出行列操作
    rows =[board [i,:] for i in range(9)]
    lines =[board [:,j] for j in range(0)]
#进行3X3分区
    sqrs =[board[i:i+3,j:j+3].flatten() for i in [0,3,6] for j in [0,3,6]]
# for i in [0,3,6]:
#     print(i)
#np.vstack:垂直（按照行顺序）的把数组给堆叠起来
#np.hstack:水平(按列顺序)把数组给堆叠起来
    for view in np.vstack((rows, lines, sqrs)):
        if len(np.unique(view)) != 9:
            return('Try again!')
    return ('Finished!')