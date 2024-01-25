import random
import numpy as np
from operator import attrgetter
import bisect
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time

#this is setball
#1=A,2=B,3=C,,,8=H,9=I,10=J
today_setdata=7

lot=1680
df=pd.read_csv("loto6.csv",encoding="shift_jis")

sd=pd.read_csv("setdata.csv",encoding="shift_jis")

sb=pd.read_csv("setbias.csv",encoding="shift_jis")

def check_one():
    one=[]
    two=[]
    check1=0
    check2=0
    check3=0
    check4=0
    rery1=0
    rery2=0
    rery3=0
    rery4=0
    bo=0
    bo2=0
    bo3=0
    bo4=0
    for loto in range(lot):
        rea=df.loc[loto]["1等口数"]
        rea2=df.loc[loto]["2等口数"]
        rea3=df.loc[loto]["3等口数"]
        rea4=df.loc[loto]["4等口数"]
        rea5=df.loc[loto]["5等口数"]
        kya=df.loc[loto]["キャリーオーバー"]
        two.append([rea,rea2,rea3,rea4,rea5,kya])
    for i in range(1604):
        if two[i][0]==0:
            check2=check2+two[i][1]*1000+two[i][2]*30+two[i][3]*0.68+two[i][4]*0.1
            rery2=rery2+1
        else:
            check1=check1+two[i][1]*1000+two[i][2]*30+two[i][3]*0.68+two[i][4]*0.1
            rery1=rery1+1
    for x in range(1604):
        if x>=0:
            if two[x][5]==0:
                if two[x+1][0]==0:
                    bo=bo+1
                else:
                    bo3=bo3+1
            else:
                if two[x+1][0]==0:
                    bo2=bo2+1
                else:
                    bo4=bo4+1
    for y in range(1604):
        if two[y][0]==0:
            check3=check3+two[y+1][1]*1000+two[y+1][2]*30+two[y+1][3]*0.68+two[y+1][4]*0.1
            rery3=rery2+1
        else:
            check4=check4+two[y+1][1]*1000+two[y+1][2]*30+two[y+1][3]*0.68+two[y+1][4]*0.1
            rery4=rery1+1
    
    print("1等",check1/rery1)
    print("2以下",check2/rery2)
    
    print("kya_ok",bo4/bo2)
    print("kya_no",bo3/bo)
    
    print("kya_on 1等",check3/rery3)
    print("kya_no 2等以下",check4/rery4)

loto6=[]
for loto in range(lot):
    d1=df.loc[loto]["第1数字"]
    d2=df.loc[loto]["第2数字"]
    d3=df.loc[loto]["第3数字"]
    d4=df.loc[loto]["第4数字"]
    d5=df.loc[loto]["第5数字"]
    d6=df.loc[loto]["第6数字"]
    rea2=df.loc[loto]["2等口数"]
    rea3=df.loc[loto]["3等口数"]
    rea4=df.loc[loto]["4等口数"]
    rea5=df.loc[loto]["5等口数"]
    tama=df.loc[loto]["セット玉"]
    loto6.append([d1,d2,d3,d4,d5,d6,rea2,rea3,rea4,rea5,tama])

def check_list():
    print(loto6[1][6])
    ant=0
    for i in range(1600):
        ant=loto6[i][9]+ant
    print(ant/1600)

setsr=[]
for ssse in range(18):
    ss3=sb.loc[ssse]["E"]
    setsr.append([ss3])
print(setsr)

def roulette_choice(weight):
    total = sum(weight)
    c_sum = np.cumsum(weight)
    return bisect.bisect_left(c_sum, total * random.random())

def set_set():
    set_value=[]
    if(today_setdata==1):
        for set1 in range(43):
            s1=sb.loc[set1]["A"]
            set_value.append([s1])
    elif(today_setdata==2):
        for set1 in range(43):
            s1=sb.loc[set1]["B"]
            set_value.append([s1])
    elif(today_setdata==3):
        for set1 in range(43):
            s1=sb.loc[set1]["C"]
            set_value.append([s1])
    elif(today_setdata==4):
        for set1 in range(43):
            s1=sb.loc[set1]["D"]
            set_value.append([s1])
    elif(today_setdata==5):
        for set1 in range(43):
            s1=sb.loc[set1]["E"]
            set_value.append([s1])
    elif(today_setdata==6):
        for set1 in range(43):
            s1=sb.loc[set1]["F"]
            set_value.append([s1])
    elif(today_setdata==7):
        for set1 in range(43):
            s1=sb.loc[set1]["G"]
            set_value.append([s1])
    elif(today_setdata==8):
        for set1 in range(43):
            s1=sb.loc[set1]["H"]
            set_value.append([s1])
    elif(today_setdata==9):
        for set1 in range(43):
            s1=sb.loc[set1]["I"]
            set_value.append([s1])
    elif(today_setdata==10):
        for set1 in range(43):
            s1=sb.loc[set1]["J"]
            set_value.append([s1])
            
    set1=set_value[:8]
    set2=set_value[8:15]
    set3=set_value[15:22]
    set4=set_value[22:28]
    set5=set_value[28:33]
    set6=set_value[33:37]
    set7=set_value[37:40]
    set8=set_value[40:42]
    set9=set_value[42:43]
    
    select_set=[]
    weight_set=[8,7,7,6,5,4,3,2,1]
    RouletteS=roulette_choice(weight_set)+1
        
    if RouletteS==1:
        select_set=set1
    elif RouletteS==2:
        select_set=set2
    elif RouletteS==3:
        select_set=set3
    elif RouletteS==4:
        select_set=set4
    elif RouletteS==5:
        select_set=set5
    elif RouletteS==6:
        select_set=set6
    elif RouletteS==7:
        select_set=set7
    elif RouletteS==8:
        select_set=set8
    elif RouletteS==9:
        select_set=set9
        
    return select_set

class Individual(np.ndarray):
    """Container of a individual."""
    fitness = None
    def __new__(cls, a):
        return np.asarray(a).view(cls)


def create_ind(n_gene):
    """Create a individual."""
    set_groupe=set_set()
    #return Individual([random.randint(1, 43) for i in range(n_gene)])
    return Individual([random.choice(set_groupe) for i in range(n_gene)])
   
def create_pop(n_ind, n_gene):
    """Create a population."""
    pop = []
    for i in range(n_ind):
        ind = create_ind(n_gene)
        pop.append(ind)
    return pop

def set_fitness(eval_func, pop):
    """Set fitnesses of each individual in a population."""
    for i, fit in zip(range(len(pop)), map(evalOneMax, pop)):
        pop[i].fitness = fit

def evalOneMax(ind):
    """loto6 score"""
    sama=[]
    fill=0
    rea2=14 #２等なら14 3等なら450
    rea3=450
    rea4=20900
    rea5=326000
    #today_setdata=8
    if today_setdata==1:
        ts=["A"]
    elif today_setdata==2:
        ts=["B"]
    elif today_setdata==3:
        ts=["C"]
    elif today_setdata==4:
        ts=["D"]
    elif today_setdata==5:
        ts=["E"]
    elif today_setdata==6:
        ts=["F"]
    elif today_setdata==7:
        ts=["G"]
    elif today_setdata==8:
        ts=["H"]
    elif today_setdata==9:
        ts=["I"]
    elif today_setdata==10:
        ts=["J"]
    
    
    for num in range(len(loto6)):
        samenum=0
        if ts[0]==loto6[num][10]:
            for six in range(5):
                for five in range(4):
                    if loto6[num][six]==ind[five]:
                    #version1
                    #if loto6[num][6]>=13:
                        #fill=(fill+num/50)*1.3
                    #else: 
                        #fill=(fill+num/50)*0.8
                    
                    #version2
                    #かぶりセット玉ごとに評価
                    #110 432 592 366 92 12 0
                    #被り数ごとに評価する
                    #0:2007208 1:490390 2:66474 3:5275 4:258 5:1607 6:0
                        samenum=samenum+1
            kostex=0
            for kofive in range(4):
                for koset in range(17):
                    if ind[kofive]==setsr[koset]:
                        kostex=kostex+1
            if kostex==0:
                rave=1
            elif kostex==1:
                rave=4
            elif kostex==2:
                rave=6
            elif kostex==3:
                rave=3
            elif kostex==4:
                rave=0.9
            elif kostex==5:
                rave=0.1
            elif kostex==6:
                rave=0
                
            
            if samenum==1:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+5)*same_good*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+5)*same_good*rave*rage3*rage4
                        else:
                            fill=(fill+5)*same_good*rave*rage3
                    else:
                        fill=(fill+5)*same_good*rave
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+5)*same_bad*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+5)*same_bad*rave*rage3*rage4
                        else:
                            fill=(fill+5)*same_bad*rave*rage3
                    else:
                        fill=(fill+5)*same_bad*rave
            if samenum==2:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+4)*same_good*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+4)*same_good*rave*rage3*rage4
                        else:
                            fill=(fill+4)*same_good*rave*rage3
                    else:
                        fill=(fill+4)*same_good*rave
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+4)*same_bad*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+4)*same_bad*rave*rage3*rage4
                        else:
                            fill=(fill+4)*same_bad*rave*rage3
                    else:
                        fill=(fill+4)*same_bad*rave
            if samenum==3:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+3)*same_good*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+3)*same_good*rave*rage3*rage4
                        else:
                            fill=(fill+3)*same_good*rave*rage3
                    else:
                        fill=(fill+3)*same_good*rave
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+3)*same_bad*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+3)*same_bad*rave*rage3*rage4
                        else:
                            fill=(fill+3)*same_bad*rave*rage3
                    else:
                        fill=(fill+3)*same_bad*rave
            if samenum==4:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+1)*same_good*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+1)*same_good*rave*rage3*rage4
                        else:
                            fill=(fill+1)*same_good*rave*rage3
                    else:
                        fill=(fill+1)*same_good*rave
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+1)*same_bad*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+1)*same_bad*rave*rage3*rage4
                        else:
                            fill=(fill+1)*same_bad*rave*rage3
                    else:
                        fill=(fill+1)*same_bad*rave
            if samenum==5:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+2)*same_good*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+2)*same_good*rave*rage3*rage4
                        else:
                            fill=(fill+2)*same_good*rave*rage3
                    else:
                        fill=(fill+2)*same_good*rave
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+2)*same_bad*rave*rage3*rage4*rage5
                            else:
                                fill=(fill+2)*same_bad*rave*rage3*rage4
                        else:
                            fill=(fill+2)*same_bad*rave*rage3
                    else:
                        fill=(fill+2)*same_bad*rave
        
        else:
            for six in range(5):
                for five in range(4):
                    if loto6[num][six]==ind[five]:
                    #version1
                    #if loto6[num][6]>=13:
                        #fill=(fill+num/50)*1.3
                    #else: 
                        #fill=(fill+num/50)*0.8
                    
                    #version2
                    #被り数ごとに評価する
                    #0:2007208 1:490390 2:66474 3:5275 4:258 5:1607 6:0
                    
                        samenum=samenum+1
            
            
            
            if samenum==1:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+5)*diff_good*rage3*rage4*rage5
                            else:
                                fill=(fill+5)*diff_good*rage3*rage4
                        else:
                            fill=(fill+5)*diff_good*rage3
                    else:
                        fill=(fill+5)*diff_good
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+5)*diff_bad*rage3*rage4*rage5
                            else:
                                fill=(fill+5)*diff_bad*rage3*rage4
                        else:
                            fill=(fill+5)*diff_bad*rage3
                    else:
                        fill=(fill+5)*diff_bad
            if samenum==2:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+4)*diff_good*rage3*rage4*rage5
                            else:
                                fill=(fill+4)*diff_good*rage3*rage4
                        else:
                            fill=(fill+4)*diff_good*rage3
                    else:
                        fill=(fill+4)*diff_good
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+4)*diff_bad*rage3*rage4*rage5
                            else:
                                fill=(fill+4)*diff_bad*rage3*rage4
                        else:
                            fill=(fill+4)*diff_bad*rage3
                    else:
                        fill=(fill+4)*diff_bad
            if samenum==3:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+3)*diff_good*rage3*rage4*rage5
                            else:
                                fill=(fill+3)*diff_good*rage3*rage4
                        else:
                            fill=(fill+3)*diff_good*rage3
                    else:
                        fill=(fill+3)*diff_good
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+3)*diff_bad*rage3*rage4*rage5
                            else:
                                fill=(fill+3)*diff_bad*rage3*rage4
                        else:
                            fill=(fill+3)*diff_bad*rage3
                    else:
                        fill=(fill+3)*diff_bad
            if samenum==4:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+1)*diff_good*rage3*rage4*rage5
                            else:
                                fill=(fill+1)*diff_good*rage3*rage4
                        else:
                            fill=(fill+1)*diff_good*rage3
                    else:
                        fill=(fill+1)*diff_good
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+1)*diff_bad*rage3*rage4*rage5
                            else:
                                fill=(fill+1)*diff_bad*rage3*rage4
                        else:
                            fill=(fill+1)*diff_bad*rage3
                    else:
                        fill=(fill+1)*diff_bad
            if samenum==5:
                if loto6[num][6]>=rea2:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+2)*diff_good*rage3*rage4*rage5
                            else:
                                fill=(fill+2)*diff_good*rage3*rage4
                        else:
                            fill=(fill+2)*diff_good*rage3
                    else:
                        fill=(fill+2)*diff_good
                else:
                    if loto6[num][7]>=rea3:
                        if loto6[num][8]>=rea4:
                            if loto6[num][9]>=rea5:
                                fill=(fill+2)*diff_bad*rage3*rage4*rage5
                            else:
                                fill=(fill+2)*diff_bad*rage3*rage4
                        else:
                            fill=(fill+2)*diff_bad*rage3
                    else:
                        fill=(fill+2)*diff_bad
        #if samenum>5:
         #   sama.append(samenum)
    #if sama != []:
       # print("fuck",sama)
    return fill

def generate_roulette(pop,n_ind,tournsize):
    """Selection roulette"""
    roulette=[]
    total = np.sum(pop,key=attrgetter("fitness"))
    print(total)
    for i in range(n_ind):
        roulette[i] = pop[i]/total
    return roulette

def selTournament(pop, n_ind, tournsize):
    """Selection function."""
    chosen = []    
    for i in range(n_ind):
        aspirants = [random.choice(pop) for j in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter("fitness")))
    return chosen

def cxTwoPointCopy(ind1, ind2):
    """Crossover function."""
    size = len(ind1)
    tmp1 = ind1.copy()
    tmp2 = ind2.copy()
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size-1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    return tmp1, tmp2

def mutFlipBit(ind, indpb):
    """Mutation function."""
    tmp = ind.copy()
    for i in range(len(ind)):
        if random.random() < indpb:
            #print(tmp[i])
            set_groupe=set_set()
            selecten=random.choice(set_groupe)
            choice_num=int(selecten[0])
            tmp[i] = choice_num
            #print("mutation!")
    return tmp

#過去のあたりとの被り数を調べる
def past():
    kon0=0
    kon1=0
    kon2=0
    kon3=0
    kon4=0
    kon5=0
    kon6=0
    lotooo=1603
    for buf in range(lotooo):
        for bof in range(lotooo+1):
            count=0
            for numm in range(5):
                if loto6[buf][numm]==loto6[bof][numm]:
                     count=count+1
            if count==0:
                kon0=kon0+1
            elif count==1:
                kon1=kon1+1
            elif count==2:
                kon2=kon2+1
            elif count==3:
                kon3=kon3+1
            elif count==4:
                kon4=kon4+1
            elif count==5:
                kon5=kon5+1
            elif count==6:
                kon6=kon6+1
            
    print(kon0,kon1,kon2,kon3,kon4,kon5,kon6)            

n_gene   = 6   # The number of genes.
n_ind    = 40   # The number of individuals in a population.
CXPB     = 0.5   # The probability of crossover.
MUTPB    = 0.2   # The probability of individdual mutation.
MUTINDPB = 0.01  # The probability of gene mutation.
NGEN     = 40 #The number of generation loo
#ddd=[1.7,1.3,1.1,2.2,1.7,1.2,1]
ddd=[1,1,1,1,1,1,1]
rage3=ddd[0]
rage4=ddd[1]
rage5=ddd[2]
same_good=ddd[3]
same_bad=ddd[4]
diff_good=ddd[5]
diff_bad=ddd[6]

t1=time.time()
random.seed(8)
# --- Step1 : Create initial generation.
pop = create_pop(n_ind, n_gene)
set_fitness(evalOneMax, pop)
best_ind = max(pop, key=attrgetter("fitness"))

# --- Generation loop.
print("Generation loop start.")
print("Generation: 0. Best fitness: " + str(best_ind.fitness))
for g in range(NGEN):    
    #print(pop)
    # --- Step2 : Selection.
    offspring = selTournament(pop, n_ind, tournsize=3)
    # --- Step3 : Crossover.
    crossover = []
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            child1, child2 = cxTwoPointCopy(child1, child2)
            child1.fitness = None
            child2.fitness = None
        crossover.append(child1)
        crossover.append(child2)

    offspring = crossover[:]
        
    # --- Step4 : Mutation.
    mutant = []
    for mut in offspring:
        if random.random() < MUTPB:
            mut = mutFlipBit(mut, indpb=MUTINDPB)
            mut.fitness = None
        mutant.append(mut)

    offspring = mutant[:]
        
    # --- Update next population.
    pop = offspring[:]
    set_fitness(evalOneMax, pop)   
    # --- Print best fitness in the population.
    best_ind = max(pop, key=attrgetter("fitness"))
    print("Generation: " + str(g+1) + ". Best fitness: " + str(best_ind.fitness))
    
    for last in range(6):
            for lost in range(6):
                if lost==0:
                    lo=0
                elif last+lost<=5:
                    if last!=lost:
                        if best_ind[last]==best_ind[lost]:
                            best_ind[last]=random.randint(1, 43)
                            print("do it!")

                            
#best_ind.sort()
bb=[]
for i in range(n_gene):
        bb = list(itertools.chain(bb,best_ind[i]))
print("Generation loop ended. The best individual: ")
for last in range(6):
            for lost in range(6):
                if lost==0:
                    lo=0
                elif last+lost<=5:
                    if last!=lost:
                        if bb[last]==bb[lost]:
                            bb[last]=random.randint(1, 43)
                            print("do it!")
bb.sort()
print(bb)
t2=time.time()
timepass=t2-t1
print("time is",timepass)


