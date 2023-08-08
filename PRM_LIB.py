# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:34:52 2023

@author: Donghao Li
@email: dpl5368@psu.edu
Obstacle checking algorithm for connections is inspired by Alex Belchou
"""

import cv2,sys,copy,random
import matplotlib.pyplot as plt
import numpy as np
import time

class PRM:
    def __init__(self,path,pxcm):
        self.img = cv2.imread(path)
        self.imgr = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.pxcm = pxcm
        ### reduce grayscale map size
        self.im_gr = cv2.resize(self.imgr, (int(self.imgr.shape[1]*self.pxcm), int(self.imgr.shape[0]*self.pxcm)), interpolation = cv2.INTER_AREA)
    
    
    def binarymap(self,threst):
        #convert into binary map
        t = threst
        _, im_bw = cv2.threshold(self.im_gr,t,255,cv2.THRESH_BINARY) #create binary grid
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        i2 = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
        i3 = cv2.morphologyEx(i2, cv2.MORPH_CLOSE, kernel)
        
        self.im_bw = i3
        
        
    def map_gene(self,NumNode,ConnectionDistance,Connect_num,currentLocation,endLocation):
        #store white dots and black dots seperately
        s1 = time.time()
        self.wt_dot = np.argwhere(self.im_bw==0)
        self.bk_dot = np.argwhere(self.im_bw==255)
        #convert location into array of int
        cur_int = (np.rint(currentLocation)).astype(int)
        end_int = (np.rint(endLocation)).astype(int)
        if self.im_bw[cur_int[0],cur_int[1]] == 255:
            sys.exit('Start point is invalid')
        elif self.im_bw[end_int[0],end_int[1]] == 255:
            sys.exit('End point is invalid')
            
        #Generate accessable nodes randomly
        indices = np.random.choice(len(self.wt_dot), size=NumNode - 2, replace=False)        
        PRM_POINT = np.array(self.wt_dot[indices], dtype=int)
        
        PRM_POINT = np.insert(PRM_POINT,0, cur_int, axis=0)
        PRM_POINT = np.append(PRM_POINT, [end_int],axis=0)
        
        
        self.PRM_point = PRM_POINT
        # Create all the possible connections between nodes
        self.connectivity = []
        distances = np.linalg.norm(self.PRM_point[:, np.newaxis] - self.PRM_point, axis=2)
        np.fill_diagonal(distances, np.inf)
        # Generate a mask for distances within the maximum distance and only Connect_num of connection can one point have
        connection_mask = (distances <= ConnectionDistance) & (distances >0)
        # test_mask = copy.deepcopy(connection_mask)
        s2 = time.time()
        num_connections = np.sum(connection_mask, axis=1)
        excess_connections = num_connections - Connect_num
        for i, excess in enumerate(excess_connections):
            if excess > 0:
                dot_connections = np.where(connection_mask[i])[0]
                random.shuffle(dot_connections)  # Shuffle connections randomly
                connection_mask[i, dot_connections[:excess]] = False
       
        
        # Generate the indices of the pairs that satisfy the distance and obstacle conditions
        i_indices, j_indices = np.where(connection_mask)

        # Get the pairs based on the indices
        # pairs = np.column_stack((i_indices,j_indices))
        connections = []
        for pair in zip(i_indices,j_indices):
            start,end = pair
            if not self.obstacle_check(self.PRM_point[start],self.PRM_point[end]):
                connections.append(pair)
        connections=np.asarray(connections,dtype=int)
        # Make sure currendLocation and endLocation have at least 3 connections
        if np.sum(connection_mask[:,0])<3:
            curr_ind = np.argpartition(connection_mask[:,0], -3)[-3:]
            for i in range(3-np.sum(connection_mask[:,0])):
                connections = np.vstack((np.array([[0,curr_ind[i]]]),connections))
        if np.sum(connection_mask[:,-1])<3:
            end_ind = np.argpartition(connection_mask[:,-1], -3)[-3:]
            for i in range(3-np.sum(connection_mask[:,0])):
                connections = np.vstack((np.array([[self.PRM_point.shape[0]-1,end_ind[i]]]),connections))
        self.connectivity = connections      
        print('Map Gene: '+str(s2-s1))
        print('Collid Check: '+str(time.time()-s2))
        return self.PRM_point,self.connectivity
    
    def path_finding(self,astar=True):
        if astar:
            w=1
        else:
            w=0
        conn = np.asarray(self.connectivity,dtype=int)
        path =[]
        #reward should be n*2 ndarry, while the first column is the index of node and the second column is the reward
        reward = np.hstack((np.arange(self.PRM_point.shape[0]).reshape(-1,1),np.ones((self.PRM_point.shape[0],1))*np.Inf))
        reward[0,1]=0
        unused_reward = copy.deepcopy(reward)
        old_temp = 0
        while unused_reward.shape[0] != 0:
            #find the unsearched node with smallest reward as next explored node
            temp = sorted(unused_reward,key=lambda row: row[1])[0]
            #all reachable nodes were searched and can't connect to endLocation
            if temp[1] == np.Inf:
                sys.exit('Closed Loop')
            #endLocation is the next explored node, search finished
            if temp[0] == self.PRM_point.shape[0]-1:
                #trace back and find the path
                splist = self.path_org(path,old_temp)
                return self.PRM_point[splist,:]
            #read all the connections of current explored node and update the reward
            temp_list = conn[(conn[:,:]==temp[0]).any(axis=1)]
            conn = conn[(conn[:,:]!=temp[0]).all(axis=1)]
            for i in range(temp_list.shape[0]):
                update_idx = int(temp_list[i,:][temp_list[i,:]!=temp[0]])
                unused_idx = np.where(unused_reward[:,0]==update_idx)[0][0]
                #update the reward
                if unused_reward[unused_idx,1]> temp[1]+self.h(self.PRM_point[update_idx,:],self.PRM_point[-1,:])*w:
                    unused_reward[unused_idx,1] = temp[1]+self.h(self.PRM_point[update_idx,:],self.PRM_point[-1,:])*w
                    reward[update_idx,1] = temp[1]+self.h(self.PRM_point[update_idx,:],self.PRM_point[-1,:])*w
                    #temp_list will be updated only if a path with smaller reward is found
                    path.append(temp_list[i,:])
            # delete the reward of searched node
            del_idx = np.where(unused_reward[:,0]==temp[0])[0][0]
            unused_reward = np.delete(unused_reward,(del_idx),axis=0)
            old_temp = int(temp[0])
        #all nodes were searched and endLocation is not reached
        sys.exit('No Path found')

    
    def path_org(self,path,old):
        #backward trace the optimized path
        temp_path = copy.deepcopy(path)
        plist = [old]
        while plist[0]!=0:
            i = -1
            #skip all connections starting from first node in plist to other node.
            while (temp_path[i]==plist[0]).any() and (temp_path[i-1]!=(temp_path[i][temp_path[i]!=plist[0]])).all():
                i -=1
            #find the last connection to the first node in plist.
            while not (temp_path[i]==plist[0]).any():
                i -=1
                if -i>len(temp_path):
                    # It shouldn't give this error. Please contact me.
                    sys.exit('No avaliable path.')
            #update plist and delete searched path.
            plist.insert(0,int(temp_path[i][temp_path[i]!=plist[0]]))
            del temp_path[i:]
        plist.append(self.PRM_point.shape[0]-1)
        return plist
    
    
    def plot_map(self,img_name):
        if img_name =='Original':
            org_plt = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            return org_plt
        elif img_name =='Binary':
            return 255-self.im_bw
        else:
            sys.exit('Plot Name Error')
        
    def h(self,x,y):
        return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
    
    #Obstacle checking algorithm for connections based on Bresenham's line algorithm
    #Return True when obstacle found
    def obstacle_check(self, start, end):
        x1, y1 = start
        x2, y2 = end
    
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
    
        while x1 != x2 or y1 != y2:
            if self.im_bw[x1, y1] == 255:
                return True #Obstacle found
    
            if dx > dy:
                x1 += sx
                dx -= 1
            else:
                y1 += sy
                dy -= 1

        return False
    
    def check_collid(self,start,end):
        x_diff = end[0] - start[0]
        y_diff = end[1] - start[1]
        
        if x_diff == 0 and y_diff ==0:
            # sys.exit('Start point and end point are same')
            return False
        if abs(x_diff==1) and abs(y_diff==1):
            return True
        
        if x_diff == 0:# veretical path
            p_path = np.vstack((np.ones((1,abs(y_diff)))*start[0],np.arange(min(start[1],end[1]),max(start[1],end[1])))).transpose().reshape(-1,1,2)
        elif y_diff == 0:# horizontal path
            p_path = np.vstack((np.arange(min(start[0],end[0]),max(start[0],end[0])),np.ones((1,abs(x_diff)))*start[1])).transpose().reshape(-1,1,2)
        elif abs(x_diff) == abs(y_diff):
            p_path = np.vstack((np.arange(start[0]+np.sign(x_diff),end[0],np.sign(x_diff)),np.arange(start[1]+np.sign(y_diff),end[1],np.sign(y_diff)))).transpose().reshape(-1,1,2)
        else:
            kp = (end[1]-start[1])/(end[0]-start[0])
            yp = start[1]-kp*start[0]
            px_x = np.arange(start[0]+0.5*np.sign(x_diff),end[0],np.sign(x_diff))
            px_y = px_x*kp+yp
            py_y = np.arange(start[1]+0.5*np.sign(y_diff),end[1],np.sign(y_diff))
            py_x = (py_y-yp)/kp
            px   = np.vstack((  px_x+0.5*np.sign(x_diff) , px_y.round() ))
            py   = np.vstack((  py_x.round() , py_y+0.5*np.sign(y_diff) ))
            p_path = np.hstack(( (np.rint(px)).astype(int), (np.rint(py)).astype(int) )).transpose().reshape(-1,1,2)
        return ~(self.bk_dot==p_path).all(axis=2).any()




