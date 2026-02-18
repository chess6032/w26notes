def probabilities(L:list) -> list:
    return [k/sum(L) for k in L]

def gini_impurity_on_probs(P:list) -> float:
    return sum([p_ki * (1-p_ki) for p_ki in P])

def gini_impurity(L:list) -> float:
    return gini_impurity_on_probs(probabilities(L))

def get_tree_leaves(tree:list, leaves:list=[]) -> list:
    if type(el) != list:
        return [] # I am not even a list
    i_have_children = False
    for el in tree:
        if type(el) == list:
            i_have_children = True
            leaves += get_tree_leaves(el, leaves)
    if i_have_children:
        return leaves
    return leaves + tree # I am a leaf


def tree_impurity(tree:list, impurity_measure:function) -> float:
    sum(len(leaf) * impurity_measure(leaf) for leaf in tree)

if __name__ == "__main__":
    pass