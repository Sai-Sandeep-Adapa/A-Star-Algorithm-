
"""
Authors: Sai Sandeep Adapa
         Sahruday Patti
        
Brief: Implementation of A* Algorithm [Project 3 , Phase 1]

Date: 03/21/2022

"""
# Importing required Libraries
import numpy as np
import heapq as hq
import copy
import matplotlib.pyplot as plt
import cv2


# Generating Obstacles Canvas including Clearance and robot radius [15]
def Obstacles_Map_Plot(Canvas, offset=15):
    
    height, width, __ = Canvas.shape

    for i in range(width):
        for j in range(height):
            if(i<=offset) or (i>=(400-offset)) or (j<=offset) or (j>=(250-offset)):
                Canvas[j][i] = [0,255,0]

            # Including Clearance and robot radius
            if ((i-300)**2+(j-65)**2-((40+offset)**2))<=0:
                Canvas[j][i] = [0,0,255]
            
            if (j+(0.57*i)-(224.285-offset*1.151))>=0 and (j-(0.57*i)+(4.285+offset*1.151))>=0 and (i-(235+offset))<=0 and (j+(0.57*i)-(304.285+offset*1.151))<=0 and (j-(0.57*i)-(75.714+offset*1.151))<=0 and (i-(165-offset))>=0:
                Canvas[j][i] = [0,0,255]

            if ((j+(0.316*i)-(76.392-offset*1.048)>=0) and (j+(0.857*i)-(138.571+offset*1.317)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186+offset*3.352)>=0) and (j-(1.232*i)-(20.652+offset*1.586))<=0 and (j-(0.114*i)-60.909)>=0):
                Canvas[j][i] = [0,0,255]

            # Excluding Clearance and robot radius
            if ((i-300)**2+(j-65)**2-((40)**2))<=0:
                Canvas[j][i] = [51,255,51]
            
            if (j+(0.57*i)-(224.285))>=0 and (j-(0.57*i)+(4.285))>=0 and (i-(235))<=0 and (j+(0.57*i)-(304.285))<=0 and (j-(0.57*i)-(75.714))<=0 and (i-(165))>=0:
                Canvas[j][i] = [51,255,51]

            if ((j+(0.316*i)-(76.392)>=0) and (j+(0.857*i)-(138.571)<=0) and (j-(0.114*i)-60.909)<=0) or ((j-(3.2*i)+(186)>=0) and (j-(1.232*i)-(20.652))<=0 and (j-(0.114*i)-60.909)>=0):
                Canvas[j][i] = [51,255,51]

    return Canvas

# Rounding to the nearest 0.5 value
def Rounding(num):
    
    return round(num*2)/2

# Taking User Inputs : Clearance[5] | Robot Radius[10] | Step Size (1<=L<=10)
def Robot_User_inputs():

    Clearance = 0
    robot_radius = 0
    Step = 0

    while True:
        Clearance = input("Please enter the Clearance parameter of the Robot: ")
        if(int(Clearance)<0):
            print("Invalid Clearance value! Re-enter valid Clearance parameter of the Robot")
        else:
            break

    while True:
        robot_radius = input("Please enter the Radius of the Robot: ")
        if(int(robot_radius)<0):
            print("Invalid Robot Radius! Re-enter valid Robot Radius")
        else:
            break
    
    while True:
        Step = input("Please enter the Step Size [1 to 10]: ")
        if(int(Step)<1 and int(Step)>10):
            print("Invalid Step Size! Re-enter valid Step Size")
        else:
            break
    
    return int(Clearance), int(robot_radius), int(Step)

# Taking User Inputs : Xs,Ys | Xg,Yg | Theta_s | Theta_g
def User_Inputs(Canvas):
   
    initial_state = []
    final_state = []
    initial_angle = 0
    final_angle = 0
   

    while True:
        while True:
            state = input("Please enter the Start Node X Coordinate [Xs]: ")
            if(int(state)<0 or int(state)>Canvas.shape[1]-1):
                print("Invalid Xs Coordinate! Re-enter valid Xs Coordinate ")
                continue
            else:
                initial_state.append(int(state))
                break
        while True:
            state = input("Please enter the Start Node Y Coordinate [Ys]: ")
            if(int(state)<0 or int(state)>Canvas.shape[0]-1):
                print("Invalid Ys Coordinate! Re-enter valid Ys Coordinate")
                continue
            else:
                initial_state.append(int(state))
                break
        
        if(Canvas[Canvas.shape[0]-1 - initial_state[1]][initial_state[0]][0]==255):
            print("**** The provided Start Node [Xs,Ys] is in the Obstacle Region ****")
            initial_state.clear()
        else:
            break

    while True:
        while True:
            state = input("Please enter the Goal Node X Coordinate [Xg]: ")
            if(int(state)<0 or int(state)>Canvas.shape[1]-1):
                print("Invalid Xg Coordinate! Re-enter valid Xg Coordinate")
                continue
            else:
                final_state.append(int(state))
                break
        while True:
            state = input("Please enter the Goal Node Y Coordinate [Yg]: ")
            if(int(state)<0 or int(state)>Canvas.shape[0]-1):
                print("Invalid Yg Coordinate! Re-enter valid Yg Coordinate")
                continue
            else:
                final_state.append(int(state))
            break

        if(Canvas[Canvas.shape[0]-1 - final_state[1]][final_state[0]][0]==255):
            print("**** The provided Goal Node [Xg,Yg] is in the Obstacle Region ****")
            final_state.clear()
        else:
            break
    while True:
        initial_angle = input("Please enter the Initial Angle between 0 to 360 degrees {0,30,60..330,360} [Theta_s]: ")
        if(int(initial_angle)<0 or int(initial_angle)>359 or (int(initial_angle)%30 != 0)):
            print("Please enter a valid Theta_s")
        else:
            initial_state.append(int(initial_angle))
            break

    while True:
        final_angle = input("Please enter the Final Angle between 0 to 360 degrees {0,30,60..330,360} [Theta_g]: ")
        if(int(final_angle)<0 or int(final_angle)>359 or (int(final_angle)%30 != 0)):
            print("Please enter a valid Theta_g")
        else:
            final_state.append(int(final_angle))
            break

    
    return initial_state, final_state

# To check if the present node is within the final (goal) node range [min of 1.5]
def check(node, final):
    
    if(np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))<1.5) and (node[2]==final[2]):
        return True
    else:
        return False

# Cost to goal calculation using Euclidean distance method
def Cost_Function(node, final):
 
    return np.sqrt(np.power(node[0]-final[0],2)+np.power(node[1]-final[1],2))

# To check if the succeeding node is within the Obstacle Space
def Obstacle_Checking(next_width, next_height, Canvas):    
    
    if Canvas[int(round(next_height))][int(round(next_width))][0]==255:
        return False
    else:
        return True

# Navigating the robot at 0 degree angle by the Step size
def Zero_Action_Set(node, Canvas, visited, Step):    
    next_node = node.copy()
    next_angle = next_node[2] + 0    

    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = Rounding(next_node[0] + Step*np.cos(np.deg2rad(next_angle)))
    next_height = Rounding(next_node[1] + Step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<Canvas.shape[0]) and (round(next_width)>0 and round(next_width)<Canvas.shape[1]) and (Obstacle_Checking(next_width,next_height,Canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False
    
# Navigating the robot at -30 degree angle by the Step size
def Minus_Thirty_Action_Set(node, Canvas, visited, Step):
   
    next_node = node.copy()
    next_angle = next_node[2] + 30    
    
    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = Rounding(next_node[0] + Step*np.cos(np.deg2rad(next_angle)))
    next_height = Rounding(next_node[1] + Step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<Canvas.shape[0]) and (round(next_width)>0 and round(next_width)<Canvas.shape[1]) and (Obstacle_Checking(next_width,next_height,Canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle
        
        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False

# Navigating the robot at -60 degree angle by the Step size
def Minus_Sixty_Action_Set(node, Canvas, visited, Step):   
    
    next_node = node.copy()
    next_angle = next_node[2] + 60    
    
    if next_angle < 0:
        next_angle += 360
    
    next_angle %= 360 
    next_width = Rounding(next_node[0] + Step*np.cos(np.deg2rad(next_angle)))
    next_height = Rounding(next_node[1] + Step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<Canvas.shape[0]) and (round(next_width)>0 and round(next_width)<Canvas.shape[1]) and (Obstacle_Checking(next_width,next_height,Canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node,True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False

# Navigating the robot at +30 degree angle by the Step size
def Plus_Thirty_Action_Set(node, Canvas, visited, Step):    
  
    next_node = node.copy()
    next_angle = next_node[2] - 30    

    if next_angle < 0:
        next_angle += 360 
    next_angle %= 360
    next_width = Rounding(next_node[0] + Step*np.cos(np.deg2rad(next_angle)))
    next_height = Rounding(next_node[1] + Step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<Canvas.shape[0]) and (round(next_width)>0 and round(next_width)<Canvas.shape[1]) and (Obstacle_Checking(next_width,next_height,Canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node, True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node, False
    else:
        return False, next_node, False

# Navigating the robot at +60 degree angle by the Step size
def Plus_Sixty_Action_Set(node, Canvas, visited, Step):    
    
    next_node = node.copy()
    next_angle = next_node[2] - 60    
    if next_angle < 0:
        next_angle += 360
    next_angle %= 360
    next_width = Rounding(next_node[0] + Step*np.cos(np.deg2rad(next_angle)))
    next_height = Rounding(next_node[1] + Step*np.sin(np.deg2rad(next_angle)))

    if (round(next_height)>0 and round(next_height)<Canvas.shape[0]) and (round(next_width)>0 and round(next_width)<Canvas.shape[1]) and (Obstacle_Checking(next_width,next_height,Canvas)) :
        next_node[0] = next_width
        next_node[1] = next_height
        next_node[2] = next_angle

        if(visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] == 1):
            return True, next_node,True
        else:
            visited[int(next_height*2)][int(next_width*2)][int(next_angle/30)] = 1
            return True, next_node,False
    else:
        return False, next_node,False

# Implementing A* Algorithm to find path from Start Node to Goal Node
def Astar_Algthm(initial_state, final_state, Canvas, Step):
    
    Open_list = []
    Closed_list = {}
    Backtrack_key = False
    
    nodes_visited = np.zeros((500,800,12))
    
    hq.heapify(Open_list)
    current_c2c = 0
    current_c2g = Cost_Function(initial_state,final_state)
    Tot_cost = current_c2c + current_c2g
    hq.heappush(Open_list,[Tot_cost,current_c2c,current_c2g,initial_state,initial_state])
    
    while len(Open_list)!=0:
        node = hq.heappop(Open_list)
        Closed_list[tuple(node[4])] = node[3]
        if(check(node[4],final_state)):
            print("***GOAL NODE REACHED***")
            Backtrack_key = True
            Backtrack(initial_state,node[4],Closed_list,Canvas)
            break

        current_c2c = node[1]
        current_c2g = node[2]
        Tot_cost = node[0]

        flag, nth_state, dupli = Plus_Sixty_Action_Set(node[4],Canvas,nodes_visited,Step)
        if(flag):
            if tuple(nth_state) not in Closed_list:
                if(dupli):
                    for i in range(len(Open_list)):
                        if tuple(Open_list[i][4]) == tuple(nth_state):
                    
                            cost = current_c2c+Step+Cost_Function(nth_state,final_state)
                            if(cost<Open_list[i][0]):
                                Open_list[i][1] = current_c2c+Step
                                Open_list[i][0] = cost
                                Open_list[i][3] = node[4]
                                hq.heapify(Open_list)
                            break
                else:
                    hq.heappush(Open_list,[current_c2c+Step+Cost_Function(nth_state,final_state),current_c2c+Step,Cost_Function(nth_state,final_state),node[4],nth_state])
                    hq.heapify(Open_list)

        flag, nth_state, dupli = Plus_Thirty_Action_Set(node[4],Canvas,nodes_visited,Step)
        if(flag):
            if tuple(nth_state) not in Closed_list:
                if(dupli):
                    for i in range(len(Open_list)):
                        if tuple(Open_list[i][4]) == tuple(nth_state):
                            cost = current_c2c+Step+Cost_Function(nth_state,final_state)
                            if(cost<Open_list[i][0]):
                                Open_list[i][1] = current_c2c+Step
                                Open_list[i][0] = cost
                                Open_list[i][3] = node[4]
                                hq.heapify(Open_list)
                            break
                else:
                    hq.heappush(Open_list,[current_c2c+Step+Cost_Function(nth_state,final_state),current_c2c+Step,Cost_Function(nth_state,final_state),node[4],nth_state])
                    hq.heapify(Open_list)
                
        flag, nth_state, dupli = Zero_Action_Set(node[4],Canvas,nodes_visited,Step)
        if(flag):
            if tuple(nth_state) not in Closed_list:
                if(dupli):
                    for i in range(len(Open_list)):
                        if tuple(Open_list[i][4]) == tuple(nth_state):
                            cost = current_c2c+Step+Cost_Function(nth_state,final_state)
                            if(cost<Open_list[i][0]):
                                Open_list[i][1] = current_c2c+Step
                                Open_list[i][0] = cost
                                Open_list[i][3] = node[4]
                                hq.heapify(Open_list)
                            break
                else:
                    hq.heappush(Open_list,[current_c2c+Step+Cost_Function(nth_state,final_state),current_c2c+Step,Cost_Function(nth_state,final_state),node[4],nth_state])
                    hq.heapify(Open_list)

        flag, nth_state, dupli = Minus_Thirty_Action_Set(node[4],Canvas,nodes_visited,Step)
        if(flag):
            if tuple(nth_state) not in Closed_list:
                if(dupli):
                    for i in range(len(Open_list)):
                        if tuple(Open_list[i][4]) == tuple(nth_state):
                            cost = current_c2c+Step+Cost_Function(nth_state,final_state)
                            if(cost<Open_list[i][0]):
                                Open_list[i][1] = current_c2c+Step
                                Open_list[i][0] = cost
                                Open_list[i][3] = node[4]
                                hq.heapify(Open_list)
                            break
                else:
                    hq.heappush(Open_list,[current_c2c+Step+Cost_Function(nth_state,final_state),current_c2c+Step,Cost_Function(nth_state,final_state),node[4],nth_state])
                    hq.heapify(Open_list)

        flag,nth_state,dupli = Minus_Sixty_Action_Set(node[4],Canvas,nodes_visited,Step)
        if(flag):
            if tuple(nth_state) not in Closed_list:
                if(dupli):
                    for i in range(len(Open_list)):
                        if tuple(Open_list[i][4]) == tuple(nth_state):
                            cost = current_c2c+Step+Cost_Function(nth_state,final_state)
                            if(cost<Open_list[i][0]):
                                Open_list[i][1] = current_c2c+Step
                                Open_list[i][0] = cost
                                Open_list[i][3] = node[4]
                                hq.heapify(Open_list)
                            break
                else:
                    hq.heappush(Open_list,[current_c2c+Step+Cost_Function(nth_state,final_state),current_c2c+Step,Cost_Function(nth_state,final_state),node[4],nth_state])
                    hq.heapify(Open_list)

    if not Backtrack_key:    
        print("***Solution Not Found***")
        print("Total No. of Explored Nodes: ",len(Closed_list))

# Backtracking the Start Node after the Goal Node is reached
def Backtrack(initial_state,final_state,Closed_list,Canvas):
   
    # Generating the A* visualization video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Astar_sandeep_sahruday.avi',fourcc,400,(Canvas.shape[1],Canvas.shape[0]))
    
    print("Total Number of nodes Explored = ",len(Closed_list)) 
    
    keys = Closed_list.keys() 
    path_stk = [] 
    keys = list(keys)
    for key in keys:
        p_node = Closed_list[tuple(key)]
        cv2.circle(Canvas,(int(key[0]),int(key[1])),2,(255,0,0),-1)
        cv2.circle(Canvas,(int(p_node[0]),int(p_node[1])),2,(255,0,0),-1)
        Canvas = cv2.arrowedLine(Canvas, (int(p_node[0]),int(p_node[1])), (int(key[0]),int(key[1])), (0,255,255), 1, tipLength = 0.2)
        cv2.imshow("Visulaization of Optimal Path",Canvas)
        cv2.waitKey(1)
        out.write(Canvas)

    parent_node = Closed_list[tuple(final_state)]
    path_stk.append(final_state) 
    
    while(parent_node!=initial_state):
        
        path_stk.append(parent_node)
        parent_node = Closed_list[tuple(parent_node)]
    
    path_stk.append(initial_state) 
    print("Optimal Path Steps: ")
    start_node = path_stk.pop()
    print(start_node)
    while(len(path_stk)>0):
        path_node = path_stk.pop()
        cv2.line(Canvas,(int(start_node[0]),int(start_node[1])),(int(path_node[0]),int(path_node[1])),(128,128,128),5)
        print(path_node)
        start_node = path_node.copy()
        out.write(Canvas)

    out.release()

# Calling necessary functions
if __name__ == '__main__':
    
    Canvas = np.ones((250,400,3),dtype="uint8")
    initial_state, final_state = User_Inputs(Canvas)    
    Clearance, robot_radius, Step = Robot_User_inputs()
    Canvas = Obstacles_Map_Plot(Canvas,offset = (Clearance + robot_radius))
     
    # Converting to Image Coordinates:
    initial_state[1] = Canvas.shape[0]-1 - initial_state[1]
    final_state[1] = Canvas.shape[0]-1 - final_state[1]

    # Converting to image coordinates
    if initial_state[2] != 0:
        initial_state[2] = 360 - final_state[2]
    if final_state[2] != 0:
        final_state[2] = 360 - final_state[2]


    

    cv2.circle(Canvas,(int(initial_state[0]),int(initial_state[1])),2,(0,0,255),-1)
    cv2.circle(Canvas,(int(final_state[0]),int(final_state[1])),2,(0,0,255),-1)
    
    Astar_Algthm(initial_state,final_state,Canvas,Step)   
    
    
    cv2.imshow("Visulaization of Optimal Path", Canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    