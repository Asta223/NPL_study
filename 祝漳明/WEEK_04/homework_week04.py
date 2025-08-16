#week4作业
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#zhuzma:实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence):
    # TODO
    DAG = {}
    N = len(sentence)
    for k in range(N):
        tmplist = []
        i = k
        frag = sentence[k]
        while i < N:
            if frag in Dict:
                tmplist.append(i)
            i += 1
            frag = sentence[k:i+1]
        if not tmplist:
            tmplist.append(k)
        DAG[k] = tmplist
    target = DAG
    return target

def all_cut_2(sentence):
    def dfs(start, path):
        if start == len(sentence):
            target.append('/'.join(path))
            return

        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                dfs(end, path)
                path.pop()

    target = []
    dfs(0, [])
    return target

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

class DAGcode:

    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = all_cut(sentence)
        self.length = len(sentence)
        self.unfinish_path = [[]]
        self.finnish_path = []

    def decode_next(self,path):
        path_length = len("".join(path))
        if path_length == self.length:
            self.finnish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths
        return

    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()
            self.decode_next(path)

def function_1():
    dd = DAGcode(sentence)
    dd.decode()
    return dd.finnish_path

def function_2():
    all_paths = all_cut_2(sentence)
    return all_paths

print(function_1())
print(function_2())
