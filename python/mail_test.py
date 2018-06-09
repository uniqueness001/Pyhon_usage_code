# coding:utf-8
import datetime
import time
from poplib import POP3_SSL
popServerAddress ="mail.163.com"
emailAdress =input("请输入邮箱地址：")
pwd =input("请输入密码：")
lastNumber = 1
while True:
    #建立连接
    server = POP3_SSL(popServerAddress,timeout=3)
    #不显示与服务器之间的交互信息
    server.set_debuglevel(0)
    #登陆
    server.user(emailAdress)
    server.pass_(pwd)
    #获取邮箱全部邮件编号
    _, mails, _ = server.list()
    print(mails)
    #退出
    server.quit()
    #获取最新邮件的ID
    newestNumber = int(mails[-1].split()[0])
    if newestNumber != lastNumber :
        print('{}--您有{}封未读的邮件'.format(str(datetime.datetime.now())[:19],
                                      newestNumber-lastNumber))
        lastNumber = newestNumber
    #一分钟后重新检查
    time.sleep(60)
1

