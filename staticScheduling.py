#静态优先数调度算法
n=int(input("输入你要进行的进程数"))
l=[]
for i in range(n):
    d={}
    #d和l的目的是创建一个哈希表，让所有的信息都存储在哈希表内部
    name=input("输入进程名称")
    d['name']=name
    super=int(input("输入进程优先级"))
    d['super']=super
    time=int(input("输入进程运行时间"))
    d['time']=time
    #一开始将所有的进程都出在一个等待的状态
    d['state']='wait'
    l.append(d)
#展示所有的信息
print("当前各个进程的信息为")
for i in l:
    print(i)
#通过冒泡排序算法，以优先级为标准来对所有的数据进行一个排列
for i in range(len(l)):
    for j in range(i+1,len(l)):
        if l[i]['super']<l[j]['super']:
            l[i],l[j]=l[j],l[i]
        #如果遇到了优先级相等，优先让时间短的进程先运行
        if l[i]['super']==l[j]['super'] and l[i]['time']>l[j]['time']:
            l[i], l[j] = l[j], l[i]
print("进程的执行的顺序为：")
for i in range(len(l)):
    #在运行的时候，将状态设置成run
    l[i]['state']='run'
    print("第"+str(i+1)+"个执行的进程的名称为"+l[i]['name'])
    print("此时系统内的状态为")
    for j in l:
        print(j)
    #运行结束，设置成已运行
    l[i]['state']='run_over'
    print("==============================")
l[-1]['state']='run_over'
print("此时系统内的状态为")
for j in l:
    print(j)
print("所有进程进行完毕")

