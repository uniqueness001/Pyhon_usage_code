# coding:utf-8
# 题目要求：
# 你得到一个可能混合大小写字母的字符串，你的任务是把该字符串转为仅使用小写字母或者大写字母，为了尽可能少的改变：
# 如果字符串包含的大字母数小于等于小写字母数，则把字符串转为小写。
# 如果大写的数目大于小写字母数，则把字符串转为全大写。
#
# 比如：
# solve('coDe')=="code"
# solve("CODe")=="CODE"
# way_01
# def solve(s):
#     upper_count = 0
#     lower_count = 0
#     for i in s:
#         if i.isupper():
#             upper_count += 1
#         elif i.islower():
#             lower_count += 1
#     if upper_count > lower_count:
#         return s.upper()
#     else:
#         return s.lower()
# if __name__ == '__main__':
#         result = solve(s=input('请输入：'))
#         print(result)
# way_02
# def check_character(c):
#     return -1 if c.islower() else 1
# def solve(s):
#     N = list(map(check_character , s))
#     return s.upper() if sum(N)>0 else s.lower()
# if __name__ == '__main__':
#     result = solve(s=input('请输入：'))
#     print(result)
# way_03
def solve(s):
    upper = sum(l.isupper() for l in s)
    lower = sum(l.islower() for l in s)
    return [s.lower(),s.upper()][upper>lower]
if __name__ == '__main__':
    result = solve(s=input('请输入：'))
    print(result)