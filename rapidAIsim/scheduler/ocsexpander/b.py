import numpy as np

class NetworkModifier:
  def __init__(self, pod_num):
    self.pod_num = pod_num
  def next_topology(self, clique, u, seed = None):
    # clique: list of cliques
    # each clique is (a list of nodes, weight)
    # clique=None leads to random clique sizes
    if seed is not None:
      np.random.seed(seed)
    for _ in range(3):
      if clique is None:
        # generate random clique sizes
        clique = []
        nodes_left = self.pod_num
        while nodes_left > 0:
          c = np.random.randint(1, min(u, nodes_left)+1)
          clique.append(c)
          nodes_left -= c
        shuffle = np.random.permutation(self.pod_num)
        clique = [shuffle[sum(clique[:i]):sum(clique[:i+1])].tolist() for i in range(len(clique))]
        for i in range(len(clique)):
          if(len(clique[i]) == 1):
            clique[i] = (clique[i], 0)
          else:
            clique[i] = (clique[i], np.random.randint(1, u//(len(clique[i])-1)+1))
      else:
        # verify clique sizes
        for c in clique:
          if (len(c[0])-1)*(c[1]) > u:
            raise ValueError("Clique size exceeds limit")
        assert sum([len(c[0]) for c in clique]) == self.pod_num
        # split or merge cliques
        opt = np.random.rand()
        if(len(clique) >= np.sqrt(self.pod_num)):
          p = 0.7
        else:
          p = 0.3
        new_clique_list = clique
        if opt < p and len(clique) > 1:
          # merge two cliques
          i, j = np.random.choice(len(clique), size=2, replace=False)
          # print("merging", clique[i], clique[j])
          new_clique = clique[i][0] + clique[j][0]
          new_weight = min(u//(len(new_clique)-1), ((len(clique[i][0])-1)*clique[i][1]+(len(clique[j][0])-1)*clique[j][1])//(len(new_clique)-1))
          if new_weight < u//(len(new_clique)-1):
            new_weight += np.random.randint(1, u//(len(new_clique)-1)-new_weight+1)
          new_clique_list = [clique[k] for k in range(len(clique)) if k != i and k != j]
          new_clique_list.append((new_clique,new_weight))
        elif opt >= p:
          # split a clique
          ii = np.random.randint(self.pod_num)
          i = 0
          while ii not in clique[i][0]:
            i += 1
          # print("splitting", clique[i])
          if len(clique[i][0]) == 1:
            return clique
          split_point = np.random.randint(1, len(clique[i][0]))
          np.random.shuffle(clique[i][0])
          new_clique1 = clique[i][0][:split_point]
          new_clique2 = clique[i][0][split_point:]
          if(len(new_clique1) == 1):
            new_weight1 = 0
          else:
            new_weight1 = min(u//(len(new_clique1)-1), (len(clique[i][0])-1)*clique[i][1]//(len(new_clique1)-1))
          if(len(new_clique2) == 1):
            new_weight2 = 0
          else:
            new_weight2 = min(u//(len(new_clique2)-1), (len(clique[i][0])-1)*clique[i][1]//(len(new_clique2)-1))
          new_clique_list = [clique[k] for k in range(len(clique)) if k != i]
          new_clique_list.append((new_clique1,new_weight1))
          new_clique_list.append((new_clique2,new_weight2))
        clique = new_clique_list
    return clique
  def clique_to_cij(self, clique):
    c_ij = np.zeros((self.pod_num, self.pod_num), dtype=int)
    for c in clique:
      nodes, weight = c
      for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
          c_ij[nodes[i]][nodes[j]] = weight
          c_ij[nodes[j]][nodes[i]] = weight
    return c_ij

# a = NetworkModifier(pod_num=8)
# clique = None
# u = 50
# for _ in range(14):
#   clique = a.next_topology(clique, u)
#   print(clique)
#   print(a.clique_to_cij(clique))
