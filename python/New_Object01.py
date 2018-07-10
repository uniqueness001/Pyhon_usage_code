# encoding:utf-8
class people:
    name =''
    age =0
    _weight =0
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self._weight = w
    def spark(self):
        print("%s 说: 我 %d 岁了。" % (self.name, self.age))
p = people('zee',25,60)
p.spark()
