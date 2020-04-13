import pandas as pd
import numpy as np

class Apriori:

    def __init__(self, init_weight=False):

        self.weight = init_weight

    def loadDatasets(self):
    
        # sample1
        '''
        self.datasets = [
                    ['r', 'z', 'h', 'j', 'p',],
                    ['z', 'y', 'x', 'w', 'v', 'u', 't', 's',],
                    ['z',],
                    ['r', 'x', 'n', 'o', 's',],
                    ['y', 'r', 'x', 'z', 'q', 't', 'p',],
                    ['y', 'z', 'x', 'e', 'q', 's', 't', 'm',],]
        '''

        # sample2
        '''
        chaosList = [chr(x) for x in range(ord('A'), ord('F')+1)]
            #+\[chr(x) for x in range(ord('a'), ord('z')+1)] + [*range(10)] 
        np.random.seed(6143)
        self.datasets = np.random.choice(chaosList, (1000, 10)).tolist()
        '''

        # sample3
        self.datasets = [
                ['li', 'ming', 'zhi'],
                ['alas', 'tuoer', 'stain', 'ein'],
                ['li', 'zhi', 'ming', 'li', 'zhi'],
                ['li', 'alas', 'stain'],
                ['li', 'tuoer', 'ein'],
                ['ein', 'ming', 'alas'],]
                #['li', 'zhi', 'tuoer']]


    def loadAndTransDatasets(self, df_train):
        """
            获取原始训练数据, 给连续特征分组, 返回修改后的新样本数据
            同时可输出特征分组文件, 供手动修改分组权重值"""
        df = df_train.copy()

        # 1 特征转换: 指标值分组和转换
        # 1.1 对类别特征, 用[特征名+'__'+特征值]的结果替换原来的特征值
        # 如将'建筑业'替换成'industry__建筑业'
        for f1 in ['industry', 'city', 'district', 'is_change_premises', 
                'cacellation_reason', 'is_real_premise',
                'is_executee', 'is_financial_black_list', 'alert_level'] :
            df[f1] = df[f1].map(lambda x: f1 + '__' + str(x))
            #print(df[f1].value_counts())
    
        # 1.2 对连续特征值, 先分组, 再用[指标名+'__'+分组]的结果替换分组后的指标值
        for f2 in ['registered_capital', 'employees', 'total_assets', 
                'total_tax', 'industry_index', 'fraud_score',
                'period_abnormal', 'rate_frozon_holdings', 'total_liabilities', 
                'total_debt', 'judical_auction_amount', 'complaints_number_monthly',
                'rate_conciliation', 'sub_enterprises_number',]:
            df[f2] = pd.cut(df[f2], 4, duplicates='drop').map(
                    lambda x: f2 + '__' + str(x))
            #print(df[f2].value_counts())
    
        # 1.3 给个别指标手动指定分组后改名
        #print(df['period_abnormal'].value_counts())
        #print(pd.cut(df['period_abnormal'], bins=5))
    
        del df['enterprise_id']
    
        # 2 输出转换后的数据集
        # 2.1 输出所有指标的所有分组值, 存储在features_group_name.xlsx文件中
        df_features_group = pd.DataFrame()
        for c in df.columns:
            #print(c)
            df1 = df[c].value_counts().reset_index()
            df1.columns = ['feature_group_name', 'counts']
            df1['interval'] = df1['feature_group_name'].map(lambda x: x.split('__')[1])
            #print(df1[['features', 'interval', 'counts']])
            df_features_group = pd.concat([df_features_group, 
                    df1[['feature_group_name', 'interval', 'counts']]])
        #print(df_features_group)
        df_features_group.to_excel('datasets/features_group_name_v2.xlsx', index=False)

        # 2.2 存储分组转换后的指标值
        #print(df.shape, '\n', df.head(), '\n', df.columns)
        df.to_csv('datasets/features_grouped_v2.csv', sep=';', index=False)
    
        # 3 将特征转换后的数据集转换成数组格式, 模型利用self.datsets做正式训练
        self.datasets = df.to_numpy()

    def loadSampleWeight(self):

        self.df_group = pd.DataFrame([
                ['li', 5], ['ming', 2], ['zhi', 5], ['alas', 4],
                ['tuoer', 5], ['stain', 1], ['ein', 4]], 
                columns=['feature_group_name', 'weight'])

    def loadRealWeight(self):
        """导入指标分组权重值"""

        self.df_group = pd.read_excel('datasets/features_group_name_adjusted.xlsx')

    def createSetOf1Item(self):
        """统计数据集中所有不重复的单元素项集, 并返回"""
    
        set_c1 = set()
        for l in self.datasets:
            set_c1.update(l) 
        # 将每个单元素项集的格式改为frozenset
        frozenset_c1 = [*map(frozenset, [[x] for x in set_c1])]
    
        return frozenset_c1
    
    def computeSupport(self, c):
        """给定组合项集, 计算它在总记录中的支持度, 如果查找不到记录则返回0"""

        # 将特征转换后的self.datasets转换成集合格式
        D = [*map(set, self.datasets)]

        # 初始化该输入项集的统计字典
        ssCnt = {}
        numItems = float(len(D))
        # 对每个样本集合
        for tid in D:
            # 如果该项集在该样本出现
            if c.issubset(tid):
                # 且在统计字典中没有出现, 则统计数加1
                if c not in ssCnt.keys():
                    ssCnt[c] = 1
                else:
                    ssCnt[c] += 1
        # 如果循环统计结束该项集都没有在统计字典中, 说明该项集的统计个数为0
        if c not in ssCnt.keys():
            ssCnt[c] = 0
        #print('待计算支持度的项集:', ssCnt)

        # 计算候选项集的支持度, 并筛选出频繁项集
        for key in ssCnt.keys():
            # 根据self.weight=init_weight判断是否给指标分组值乘上权重
            # 再计算各候选项集的支持度
            if self.weight:
                # 在此处乘上筛选出的该项集中所有特征分组值的权重中的最小值
                group_weight = self.df_group['weight'][self.df_group[
                        'feature_group_name'].isin([x for x in key])].min()
                #print(group_weight)
                support = (ssCnt[key] / numItems) * group_weight
            else:
                support = ssCnt[key] / numItems
            # 保存下该候选项集(可能不是频繁项集)的支持度
            ssCnt[key] = support
    
        return ssCnt[c]
    
    def selCombs(self, D, Ck, minSupport):
        """
            function: 挑选高于最小支持度的组合项集
            input: 
                D - 集合格式的数据集, 
                Ck - k元候选项集, 
                minSupport - 最小支持度
            output: 
                Lk - k元频繁项集
                supK - k元候选项集的支持度
        """
    
        #print('数据集记录集合:\n', D)
        #print('\nk元候选项集:\n', Ck)

        #1 统计候选项集的计数字典ssCnt
        ssCnt = {}
        numItems = float(len(D))
        # 依次在每个样本(集合格式)中检查
        # 每个候选项集(集合格式)是否在该样本中
        # 如果在, 则判断该项集是否出现在统计结果中
        # 在的话计数加一, 不在的话添加该项集并设置计数为一
        for tid in D:
            for can in Ck:
                #print(can, tid, can.issubset(tid))
                if can.issubset(tid):
                    if can in ssCnt.keys():
                        ssCnt[can] += 1
                    else:
                        ssCnt[can] = 1
        #print('\n候选项集及其计数:\n', ssCnt)

        #2 根据计数字典计算候选项集的支持度, 并筛选出频繁项集
        Lk = []
        supK = {}
        for key in ssCnt.keys():
            # 根据self.weight=init_weight判断是否给指标分组值乘上权重
            # 再计算候选项集的支持度
            if self.weight:
                # 在此处乘上筛选出的该项集中所有特征分组值的权重中的最小值
                group_weight = self.df_group['weight'][self.df_group[
                        'feature_group_name'].isin([x for x in key])].min()
                #print(group_weight)
                support = (ssCnt[key] / numItems) * group_weight
            else:
                support = ssCnt[key] / numItems
            # 如果该候选项集的支持度大于最小支持度, 则保存该项集到Lk
            if support >= minSupport:
                Lk.insert(0, key)
            # 保存下该候选项集的支持度
            supK[key] = round(support, 4)
        #print('\nk元频繁项集:\n', Lk)
        #print('\nk元频繁项集的支持度:\n', supK)
    
        return Lk, supK
    
    def aprioriGen(self, Lk, k):
        """根据传入的k元频繁项集, 生成k+1元候选项集"""
    
        retList = []
        lenLk = len(Lk)
        for i in range(lenLk):
            for j in range(i+1, lenLk):
                # 注: 原书中对Lk[i]取值后先进行排序再截取的方法不对, 
                # list(frozenset)得到的列表中元素顺序会发生变化
                L1 = list(Lk[i])
                L2 = list(Lk[j])
                L1.sort()
                L2.sort()
                # 仅需要比较k-2个(k从2开始), 即不用比较排序后的最后一位
                # 两组候选项集的前k-1个元素相同, 之后最后一位不同
                # 求集合并集后有k+1个元素, 达成目的(不用循环列表来寻找非重复值)
                if L1[:k-2] == L2[:k-2]:
                    retList.append(Lk[i]|Lk[j])
    
        return retList
    
    def apriori(self, minSupport=0.5):
        """
            利用apriori算法发现频繁项集
            输入: minSupport - 允许的最小支持度, 默认为0.5
            输出: 
                LC - 多元素组合的候选项集列表,
                L - 多元素组合的频繁项集列表, 
                dict_sup - 包含候选项集的支持度,
        """
    
        # 从原始数据集统计单元素项集集合
        C1 = self.createSetOf1Item()
        #print(C1)
        # 将每条数据集记录转换为集合
        D = [*map(set, self.datasets)]
        #print(D)

        # 查找单元素频繁项集, 计算其候选项集的支持度
        L1, sup1 = self.selCombs(D, C1, minSupport)
        #print('单元素候选项集数:', len(C1))
        #print('筛选出的单元素频繁项集数:', len(L1))
        #print('单元素候选项集的支持度:', len(sup1.keys()))

        # 存放不同元素个数的频繁项集列表
        L = [L1]
        LC = [C1]

        '''
        C2 = self.aprioriGen(L1, 2)
        L2, sup2 = self.selCombs(D, C2, minSupport)
        print(C2)
        print('二元素候选项集数:', len(C2))
        print('筛选出的二元素频繁项集数:', len(L2))
        print('二元素候选项集的支持度:', len(sup2.keys()))
        '''

        #'''
        # 由单元频繁项集循环生成多元素候选项集
        # 并筛选出多元素频繁项集, 存储到L中
        # 同时计算多元素候选项集的支持度, 存储到dict_sup
        dict_sup = {}
        dict_sup.update(sup1)
        k = 2
        while (len(L[k-2]) > 0):
            # aprioriGen的生成方法利用了apriori原理, 子项不频繁它的超项也不频繁
            Ck = self.aprioriGen(L[k-2], k)
            Lk, supK = self.selCombs(D, Ck, minSupport)
            #print('\n%s元素候选项集数:'% k, Ck)
            #print('筛选出的%s元素频繁项集数:'% k, Lk)
            #print('%s元素候选项集的支持度:'% k, len(supK.keys()))
            dict_sup.update(supK)
            L.append(Lk)
            LC.append(Ck)
            k += 1
        #print(L)
        #print(dict_sup)
    
        return LC, L, dict_sup
        #'''
    
    def generateRules(self, L, dict_sup, minConf=0.5):
        """
            功能: 利用频繁项集挖掘关联规则
            输入: 
                L: 所有的频繁项集, 
                dict_sup: 项集的支持度, 
                minConf: 置信度阈值, 默认为0.5
            输出:
                list_rules_all: 关联规则列表的列表
        """
    
        # 初始化存储项集数的关联规则列表的列表
        list_rules_all = []
        # 单个元素的频繁项集无法产生关联规则, 所以从两个元素的L[1]开始
        for i in range(1, len(L)):
            # 初始化存储不同元素数的规则列表
            list_rules_one_item = []
            # 对每个频繁项集
            #print(len(L[i]))
            for freqSet in L[i]:
                # H1和freqSet的关系
                # H1表示某个频繁项集中的单元素固定集合列表
                H1 = [frozenset([item]) for item in freqSet]
                #print('freqSet:\t', freqSet)
                #print('H1:\t', H1)

                # 如果是由两个以上元素组成的频繁项集
                if i > 1:
                    # 则通过最初的项集构建更多的关联规则, 再判断关联规则的置信度
                    self.rulesFromConseq(freqSet, H1, dict_sup, 
                            list_rules_one_item, minConf)
                # 否则直接计算当前项集的关联规则的置信度, 挑选关联规则
                else:
                    self.calcConf(freqSet, H1, dict_sup, 
                            list_rules_one_item, minConf)
                #print('\n')
            list_rules_all.append(list_rules_one_item)

        # 上述for循环中传入的list_rules_one_item, 
        # 会在rulesFromConseq()和calcConf()中更新
        return list_rules_all
    
    def calcConf(self, freqSet, H, dict_sup, brl, minConf):
        """
            输入:
                freqSet: 频繁项集
                H: 频繁项集中的单元素固定集合列表
                dict_sup: 计算置信度所需的支持度数据
                brl: 规则列表
                minConf: 允许的最小置信度
            输出: 满足最小置信度的规则列表
        """
    
        # 初始化满足最小可信度minConf的后件规则列表
        prunedH = []

        # 依次以H中的每项作为后件规则, 剩余项作为前件规则, 构造关联规则
        # 只有两个元素项, 即使其中一个指向另一个是关联规则, 但是反过来不一定是
        for conseq in H:
            # 如果(freqSet-conseq)的支持度在dict_sup中获取出错
            # 出错的原因可能是frozenset的存储顺序与获取时的顺序不一致???
            #try:
            conf = dict_sup[freqSet]/dict_sup[freqSet-conseq]
                # 则重新计算该项集的支持度
            #except:
            #    conf = dict_sup[freqSet]/self.computeSupport(freqSet-conseq)
            #print('conseq:\t', conseq)
            #print('freqSet-conseq:\t', freqSet-conseq)
            #print('conf:\t', conf)

            # 筛选出满足最小置信度的规则
            if conf >= minConf:
                #print(freqSet-conseq, '-->', conseq, 'conf:', conf)
                brl.append((freqSet-conseq, conseq, round(conf, 4)))
                # 后件规则存入prunedH中
                prunedH.append(conseq)
        #print(prunedH)
    
        return prunedH
    
    def rulesFromConseq(self, freqSet, H, dict_sup, brl, minConf):
        """
            功能: 从最初的项集生成更多的关联规则, 判断是否满足最小置信度
            输入:
                freqSet: 频繁项集
                H: 频繁项集中的单元素固定集合列表
                dict_sup: 计算关联规则置信度所需的支持度数据
                brl: 规则列表
                minConf: 允许的最小置信度
            输出: .
        """
    
        m = len(H[0])
        if len(freqSet) > (m+1):
            # 生成H中元素的无重复组合, Hmp1会替代H作为后续的后件规则接受检测
            Hmp1 = self.aprioriGen(H, m+1)
            #print('H:', H)
            #print('Hmp1:', Hmp1)
            # 之后Hmp1中的后件规则的元素数会大于等于2
            # 测试Hmp1中元素作为后件规则构成的关联规则是否满足最小置信度
            Hmp1 = self.calcConf(freqSet, Hmp1, dict_sup, brl, minConf)
            # 如果不止一条规则满足要求, 则继续迭代判断是否进一步组合规则
            if len(Hmp1) > 1:
                self.rulesFromConseq(freqSet, Hmp1, dict_sup, brl, minConf)

def trainModel(minSupport, minConf, init_weight, sampleOrReal):

    if sampleOrReal:
        apriori = Apriori(init_weight)
        apriori.loadDatasets()
        apriori.loadSampleWeight()
    else:

        # 1 读入原始数据集, 切分训练集和测试集
        df_raw = pd.read_csv('datasets/features.csv', dtype={'enterprise_id': str})
        #print(df_raw.shape, '\n', df_raw.head()) 
        df_test = df_raw.sample(frac=0.1, replace=False, axis=0, random_state=1) 
        df_train = df_raw[~df_raw['enterprise_id'].isin(df_test['enterprise_id'])]
    
        apriori = Apriori(init_weight)
        apriori.loadAndTransDatasets(df_train)
        apriori.loadRealWeight()

    #print(apriori.datasets)
    #print(apriori.datasets.shape)
    #print(apriori.datasets[:2])

    # 1 计算频繁项集
    list_cand, list_freq, dict_sup = apriori.apriori(minSupport)
    #'''
    #print('多元候选项集数:', sum([len(l) for l in list_cand]), 
    #        [len(l) for l in list_cand])
    print('多元频繁项集数:', sum([len(l) for l in list_freq]),
            [len(l) for l in list_freq])
    #print('多元候选项集的支持度数:', len(dict_sup.keys()))
    #'''

    D = [*map(set, apriori.datasets)]
    numItems = float(len(D))

    #for l in list_freq:
    #    for it in l:
    #        print(set(it), int(np.round(dict_sup[it]*numItems)))

    # 用于展示频繁项集的支持度计算过程(正常情况下不会使用)
    '''
    list_freq_flat = [set(i) for x in list_freq for i in x]
    print(len(list_freq_flat), list_freq_flat)
    for d in list_freq_flat:
        print([i for i in d], [dict_sup[frozenset(i)] for i in d], 
                dict_sup[frozenset(d)])
    for i,j in dict_sup.items():
        print(i, j)
    '''

    # 2 利用频繁项集挖掘关联规则(含可信度)
    list_rules_all = apriori.generateRules(list_freq, dict_sup, minConf)
    rules_count = [len(l) for l in list_rules_all]
    print('关联规则:', sum(rules_count), rules_count)
    #print(list_rules_all)
    #for l in list_rules_all:
    #    for k in l:
    #        print(set(k[0]), set(k[1]), k[2])


    # 挑选后件规则集合中存在级别较高的4/5级, 作为关注关联规则
#    list_rules_sel = [x for y in list_rules_all for x in y 
#            if 'alert_level__5' in x[1] or 'alert_level__4' in x[1]]
#    rules_sel_count = [len(l) for l in list_rules_sel]
#    print('关注规则:', sum(rules_sel_count), rules_sel_count)

    '''
    # 输出这几项的结果数, 详细请见结果文件rules.xlsx
    print('多元频繁项集的列表:', len(list_freq))
    list_freq_counts = [len(x) for x in list_freq]
    print('各元频繁项集数:', list_freq_counts, sum(list_freq_counts))
    print('候选项集的支持度:', len(dict_sup))
    print('关联规则数:', sum([len(x) for x in list_rules_all]))
    print('需关注关联规则数:', len(list_rules_sel))
    #for x in list_rules_sel:
    #    print(x[-1], list(x[0]), '-->\n\t', list(x[1]))
    '''

if __name__ == "__main__":


    #2 训练模型
    minSupport = 0.7
    minConf = 0.9
    init_weight = True
    sampleOrReal = False
    trainModel(minSupport, minConf, init_weight, sampleOrReal)

    #3 测试模型结果
#    testModel(df_test)

    # 测试模型功能, 用于demo展示
    # testModule也对获取原始数据, 划分训练和测试数据集做了同样的操作
#    testModule()




