# encoding:utf-8
class Myclass:
    i = 123456
    def f(self):
        return ("hello world!")

x =Myclass()
print("Myclass中的属性是：",x.i)
print("Myclass类中的方法f的输出：",x.f())
