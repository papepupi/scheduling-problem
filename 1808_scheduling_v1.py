import numpy as np
from wildcat.solver.qubo_solver import QuboSolver
from wildcat.network.local_endpoint import LocalEndpoint
from wildcat.annealer.simulated.simulated_annealer import SimulatedAnnealer
from wildcat.annealer.simulated.single_spin_flip_strategy import SingleSpinFlipStrategy
from wildcat.annealer.simulated.temperature_schedule import TemperatureSchedule
from wildcat.util.matrix import hamiltonian_energy
    
schedule = TemperatureSchedule(initial_temperature=500, last_temperature=0.1, scale=0.99)
strategy = SingleSpinFlipStrategy(repetition=50)
annealer = SimulatedAnnealer(schedule=schedule, strategy=strategy)
local_endpoint = LocalEndpoint(annealer=annealer)

#行列の数
N = 4  #行の設定
M = 10  #列の設定
#行列Jを設定
J = np.zeros((N, M, N, M), dtype = np.float32)


#定数の定義
A = 10 #ペナルティの値

#ルール1:講師A, Bは同時に存在できない

for i1 in range(N):
    J[i1, 0, i1, 1] +=  A


for i1 in range(N):
#ルール2:ブースLに太郎と次郎は同時に存在できない
    J[i1, 2, i1, 4] +=  A
#ルール3:ブースRに太郎と次郎は同時に存在できない
    J[i1, 3, i1, 5] +=  A
#ルール4:ブースLとRをを太郎は同一時間に占有できない	
    J[i1, 2, i1, 3] +=  A	
#ルール5:ブースLとRをを次郎は同一時間に占有できない	
    J[i1, 4, i1, 5] +=  A
	
#ルール6: 講師Aは13時、14時のコマしか出勤できない。講師Bは逆に15時、16時のコマしか出勤できない
J[2, 0, 2, 0] +=  A
J[3, 0, 3, 0] +=  A
J[0, 1, 0, 1] +=  A
J[1, 1, 1, 1] +=  A

#クロネッカーデルタを定義
def delta(i, j):
    if(i == j):
        return 1
    else:
        return 0

		
#ルール7:生徒のコマ数の制約条件		
##太郎は3コマの授業をとる
for i1 in range(N):
    for j1 in range(2,4):
        for i2 in range(N):
            for j2 in range(2,4):
                J[i1, j1, i2, j2] +=  A  - 6 * A * delta(i1, i2) * delta(j1, j2) 

##次郎は2コマの授業をとる
for i1 in range(N):
    for j1 in range(4,6):
        for i2 in range(N):
            for j2 in range(4,6):
                J[i1, j1, i2, j2] +=  A  - 4 * A * delta(i1, i2) * delta(j1, j2) 
	

#ルール8:生徒がいるときは必ず講師がいる。3体問題に置き換えて解いていく
## x_j=6 :x_j=0とx_j=1(講師Aと講師B)のOR関数
for i1 in range(N):
    J[i1, 0, i1, 0] += A
    J[i1, 1, i1, 1] += A
    J[i1, 6, i1, 6] += A
    J[i1, 0, i1, 1] += A
    J[i1, 0, i1, 6] -= 2 * A
    J[i1, 1, i1, 6] -= 2 * A    

## x_j=7 :x_j=2とx_j=4(ブースL)のOR関数
for i1 in range(N):
    J[i1, 2, i1, 2] += A
    J[i1, 4, i1, 4] += A
    J[i1, 7, i1, 7] += A
    J[i1, 2, i1, 4] += A
    J[i1, 2, i1, 7] -= 2 * A
    J[i1, 4, i1, 7] -= 2 * A  	

## x_j=8 :x_j=3とx_j=5(ブースR)のOR関数
for i1 in range(N):
    J[i1, 3, i1, 3] += A
    J[i1, 5, i1, 5] += A
    J[i1, 8, i1, 8] += A
    J[i1, 3, i1, 5] += A
    J[i1, 3, i1, 8] -= 2 * A
    J[i1, 5, i1, 8] -= 2 * A  	

## x_j=9 :x_j=7とx_j=8(ブースLとブースR)のOR関数
    J[i1, 7, i1, 7] += A
    J[i1, 8, i1, 8] += A
    J[i1, 9, i1, 9] += A
    J[i1, 7, i1, 8] += A
    J[i1, 7, i1, 9] -= 2 * A
    J[i1, 8, i1, 9] -= 2 * A  	

## 講師と生徒が一方だけいる場合はNG。両方いるか、両方いないときのみOKの関数
## つまり(x_6-x_9)^2 の値が正(1）になるとペナルティ
for i1 in range(N):
    J[i1, 6, i1, 6] += A
    J[i1, 6, i1, 9] -= 2 * A
    J[i1, 9, i1, 9] += A  

#ルール9:講師の出勤にはコストがかかる
B = 4 #できれば超えてほしくない値
for i1 in range(N):
    for j1 in range(2):
        for i2 in range(N):
            for j2 in range(2):
                J[i1, j1, i2, j2] += B * delta(i1, i2) * delta(j1, j2)


#Qの計算
Q = np.empty((N*M,N*M), dtype = np.float32)
x = 0
for i1 in range(N):
    for j1 in range(M):
        y = 0
        for i2 in range(N):
            for j2 in range(M):
                Q[x,y] = J[i1,j1,i2,j2]
                y = y + 1
        x = x + 1
#最適化計算
solver = QuboSolver(Q)

def callback(arrangement):
    e = solver.hamiltonian_energy(arrangement)
    print("Energy: ", solver.hamiltonian_energy(arrangement))
    print("Spins: ", arrangement)

arrangement = solver.solve(callback, endpoint=local_endpoint).result()
result = np.empty((N, M), dtype = np.int)
i = 0
for x in range(N):
    for y in range(M):
        if(arrangement[i] == 1):
            result[x, y] = 1
        else:
            result[x, y] = 0
        i += 1
print("Result: \n", result)

