import numpy as np

def power_of_2(n):
    start = 1
    flag = False
    while start <= n:
        if start == n: flag = True
        start *= 2
    return flag

def power2_zero(n):
    if n == 0: return True
    start = 1
    flag = False
    while start <= n:
        if start == n: flag = True
        start *= 2
    return flag



'''
# hungary matching - interference between choices 
def find(index, values, spine_port_status, matched_ports, retried, used):
 # scan available spine ports
 for j in range(len(spine_port_status)):
  if spine_port_status[j][0] >= 0: continue

  if int(j/2) * 2 == j: neighbor_index = j + 1
  else: neighbor_index = j - 1

  if values[index] in spine_port_status[j][1] and values[used[neighbor_index]]!=values[index] and retried[j] == 0:
   retried[j] = 1

   if used[j] == -1:
    matched_ports[index] = j
    used[j] = index
    return True
   else:
       old_match = used[j]
       used[j] = index
       if find(old_match, values, spine_port_status, matched_ports, retried, used):
        matched_ports[index] = j
        return True
       used[j] = old_match
 return False


# spine_port_status: [[used_leaf_index, [allowed leaf index]], ...]
def hungary_matching(leaf_indexes, spine_port_status):
 matched_ports = [-1 for i in leaf_indexes]

 used = [-1 for i in spine_port_status]
 for i, value in enumerate(leaf_indexes):
  retried = [0 for i in spine_port_status]
  if not find(i, leaf_indexes, spine_port_status, matched_ports, retried, used):
   return []

 return matched_ports

'''


#normal version
'''
def find(index, values, spine_port_status, matched_ports, retried, used):
    # scan available spine ports
    for j in range(len(spine_port_status)):
        if spine_port_status[j][0] >= 0: continue
        if values[index] in spine_port_status[j][1] and retried[j] == 0:
            retried[j] = 1
            if used[j] == -1 or find(used[j], values, spine_port_status, matched_ports, retried, used):
                matched_ports[index] = j
                used[j] = index
                return True
            #else:
            #    retried[j] = 1
    return False


# spine_port_status: [[used_leaf_index, [allowed leaf index]], ...]
def hungary_matching(leaf_indexes, spine_port_status):

    matched_ports = [-1 for i in leaf_indexes ]

    used = [-1 for i in spine_port_status]
    for i, value in enumerate(leaf_indexes):
        retried = [0 for i in spine_port_status]
        if not find(i, leaf_indexes, spine_port_status, matched_ports, retried, used):
            return []

    return matched_ports
'''


#rotated version

def find(index, values, spine_port_status, matched_ports, retried, used, rotate_index = -1):
    # scan available spine ports
    s_indexes = list(range(len(spine_port_status)))
    if rotate_index >=0:s_indexes = np.roll(s_indexes, rotate_index)
    for j in s_indexes:
        if spine_port_status[j][0] >= 0: continue
        if values[index] in spine_port_status[j][1] and retried[j] == 0:
            retried[j] = 1
            if used[j] == -1 or find(used[j], values, spine_port_status, matched_ports, retried, used, rotate_index):
                matched_ports[index] = j
                used[j] = index
                return True
            #else:
            #    retried[j] = 1
    return False


# spine_port_status: [[used_leaf_index, [allowed leaf index]], ...]
def hungary_matching(leaf_indexes, spine_port_status, rotate_index = -1):

    matched_ports = [-1 for i in leaf_indexes ]

    used = [-1 for i in spine_port_status]
    for i, value in enumerate(leaf_indexes):
        retried = [0 for i in spine_port_status]
        if not find(i, leaf_indexes, spine_port_status, matched_ports, retried, used, rotate_index):
            return []

    return matched_ports



# hungary matching - randomly shuffled version
'''
def find(index, values, spine_port_status, matched_ports, retried, used):
 # scan available spine ports
 r_indexes = list(range(len(spine_port_status)))
 #print(r_indexes)
 np.random.shuffle(r_indexes)
 for j in r_indexes:#range(len(spine_port_status)):
  if spine_port_status[j][0] >= 0: continue
  if values[index] in spine_port_status[j][1] and retried[j] == 0:
   retried[j] = 1
   if used[j] == -1 or find(used[j], values, spine_port_status, matched_ports, retried, used):
    matched_ports[index] = j
    used[j] = index
    return True
   # else:
   #    retried[j] = 1
 return False


# spine_port_status: [[used_leaf_index, [allowed leaf index]], ...]
def hungary_matching(leaf_indexes, spine_port_status):
 matched_ports = [-1 for i in leaf_indexes]

 used = [-1 for i in spine_port_status]
 for i, value in enumerate(leaf_indexes):
  retried = [0 for i in spine_port_status]
  if not find(i, leaf_indexes, spine_port_status, matched_ports, retried, used):
   return []

 return matched_ports
'''


#use a simple but fast way
'''
#tt = [k[0] for k in spine_port_status]
#print("Start:", tt)
tmp_port_available= [k[0] for k in spine_port_status]
for i,li in enumerate(leaf_indexes):
    for j, sp in enumerate(spine_port_status):
        if tmp_port_available[j] >=0 : continue
        if li in sp[1]:
            matched_ports[i] = j
            tmp_port_available[j] = li
            break

#if the simple way fails then use hungary matching
if -1 in matched_ports:
    #tt = [k[0] for k in spine_port_status]
    #print("ss:", tt)
    #print("matched:", matched_ports)
    matched_ports = [-1 for i in leaf_indexes]
    used = [-1 for i in spine_port_status]
    for i, value in enumerate(leaf_indexes):
        retried = [0 for i in spine_port_status]
        if not find(i, leaf_indexes, spine_port_status, matched_ports, retried, used):
            return []
    return matched_ports
else:
    return []
'''