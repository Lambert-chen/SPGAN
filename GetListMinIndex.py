import heapq
def getListMaxNumIndex(num_list, topk=3):
    '''
    Get the position index of the smallest first n values in the list
    '''
    # max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))
    min_num_index = map(num_list.index, heapq.nsmallest(topk, num_list))
    return min_num_index



