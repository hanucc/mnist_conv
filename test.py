class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.waiting_time = 0
        self.turnaround_time = 0

    def __repr__(self):
        return f"Process {self.pid} [arrival time={self.arrival_time}, burst time={self.burst_time}]"


def fcfs(processes):
    # 按到达时间排序
    processes.sort(key=lambda p: p.arrival_time)

    current_time = 0
    for process in processes:
        # 如果当前时间小于到达时间，等待进程到达
        if current_time < process.arrival_time:
            current_time = process.arrival_time

        # 执行进程
        process.waiting_time = current_time - process.arrival_time
        process.turnaround_time = process.waiting_time + process.burst_time
        current_time += process.burst_time

    # 计算平均等待时间和平均周转时间
    total_waiting_time = sum(process.waiting_time for process in processes)
    total_turnaround_time = sum(process.turnaround_time for process in processes)
    avg_waiting_time = total_waiting_time / len(processes)
    avg_turnaround_time = total_turnaround_time / len(processes)

    # 输出结果
    print(f"FCFS scheduling: ")
    print(f"Average waiting time = {avg_waiting_time:.2f}")
    print(f"Average turnaround time = {avg_turnaround_time:.2f}")


# 测试
processes = [
    Process(1, 0, 5),
    Process(2, 1, 3),
    Process(3, 2, 8),
    Process(4, 3, 6)
]
fcfs(processes)
