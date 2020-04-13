import numpy as np
import datetime
import pandas as pd

class GeneratedDatasets:

    def __init__(self, n_samples=20):

        self.n_samples = n_samples
        np.random.seed(61)

    def industryOfEnterprises(self):
        """企业所属行业"""

        l = ["农、林、牧、渔业", '工业*', '建筑业', '批发业', '零售业', 
            '交通运输业*', '仓储业', '邮政业', '住宿业', '餐饮业', 
            '信息传输业*', '软件和信息技术服务业', '房地产开发经营', 
            '物业管理', '租赁和商务服务业', '其他未列明行业*',]
        L = np.random.choice(l, self.n_samples)
        #print('企业所属行业: ', L)

        self.industry_enterprises = L

    def districtOfEnterprises(self):
        """企业所属区域"""

        dict_districts = {
            '石家庄市': ['长安区', '桥西区', '新华区', '井陉矿区', '裕华区', 
                '藁城区', '鹿泉区', '栾城区', '井陉县', '正定县', '行唐县', 
                '灵寿县', '高邑县', '深泽县', '赞皇县', '无极县', '平山县', 
                '元氏县', '赵县', '晋州市', '新乐市', '辛集市'],
            '唐山市': ['路北区', '路南区', '古冶区', '开平区', '丰南区', '丰润区', 
                '曹妃甸区', '滦南县', '乐亭县', '迁西县', '玉田县', '遵化市', 
                '迁安市', '滦州市'],
            '秦皇岛市':    ['海港区', '山海关区', '北戴河区', '抚宁区', '青龙满族自治县', 
                '昌黎县', '卢龙县'],
            '邯郸市': ['邯山区', '丛台区', '复兴区', '峰峰矿区', '肥乡区', '永年区', 
                '临漳县', '成安县', '大名县', '涉县', '磁县', '邱县', '鸡泽县', 
                '广平县', '馆陶县', '曲周县', '武安市', '魏县'],
            '邢台市': ['桥东区', '桥西区', '邢台县', '临城县', '内丘县', '柏乡县', 
                '隆尧县', '任县', '南和县', '巨鹿县', '新河县', '广宗县', '平乡县', 
                '威县', '清河县', '临西县', '南宫市', '沙河市', '宁晋县'],
            '保定市': ['竞秀区', '莲池区', '满城区', '清苑区', '徐水区', '涞水县', 
                '阜平县', '定兴县', '唐县', '高阳县', '容城县', '涞源县', '望都县', 
                '安新县', '易县', '曲阳县', '蠡县', '顺平县', '博野县', '雄县', 
                '安国市', '高碑店市', '涿州市', '定州市'],
            '张家口市': ['桥东区', '桥西区', '宣化区', '下花园区', '崇礼区', '万全区', 
                '张北县', '康保县', '沽源县', '尚义县', '蔚县', '阳原县', '怀安县', 
                '涿鹿县', '赤城县', '怀来县'],
            '承德市': ['双桥区', '双滦区', '鹰手营子矿区', '承德县', '兴隆县', '滦平县', 
                '隆化县', '丰宁满族自治县', '宽城满族自治县', '围场满族蒙古族自治县', 
                '平泉市'],
            '沧州市': ['运河区', '新华区', '沧县', '青县', '东光县', '海兴县', '盐山县', 
                '肃宁县', '南皮县', '吴桥县', '献县', '孟村回族自治县', '泊头市', 
                '黄骅市', '河间市', '任丘市'],
            '廊坊市': ['广阳区', '安次区', '固安县', '永清县', '香河县', '大城县', 
                '文安县', '大厂回族自治县', '霸州市', '三河市'],
            '衡水市': ['桃城区', '冀州区', '枣强县', '武邑县', '武强县', '饶阳县', 
                '安平县', '故城县', '阜城县', '深州市', '景县'],}
        #print(list(dict_districts.keys()))
        L1 = np.random.choice(list(dict_districts.keys()), self.n_samples)
        ufunc_districts = np.frompyfunc(lambda x: np.random.choice(dict_districts[x]), 1, 1)
        L2 = ufunc_districts(L1)
        #print(L1, L2)
        L = np.hstack([L1[:, np.newaxis], L2[:, np.newaxis]])
        #print(L)

        self.city_district_enterprises = L

    def registeredCapital(self):
        """注册资本"""

        # 默认单位为万
        l = np.random.randint(1, 10000, self.n_samples)
        l[l>100] = np.floor(l[l>100]/100)*100
        l[l>10] = np.floor(l[l>10]/10)*10
        # 转换单位为个
        L = l * 10000

        self.registered_capital = L

    def employeeNumbers(self):
        """从业人数"""

        L = np.random.randint(1, 10000, self.n_samples)

        self.employee_numbers = L

    def numbersOfCommecialPremises(self):
        """住所数量 (经营场所是否变更) 1-住所数量为1"""

        # 设置大多数企业没有变更过经营场所
        amount = int(self.n_samples*0.8)
        l = np.hstack([[1]*amount, np.random.randint(2, 20, self.n_samples-amount)])
        # 返回前打乱顺序
        L = np.random.permutation(l)

        self.numbers_commecial_premises = L

    def dateOfAbnormalOperation(self):
        """列入日期,移出日期 (列入经营异常名录时长)"""
        
        # 设置30%企业有经营异常状况, 即有列入和移出日期
        amount = int(self.n_samples*0.3)
        l1 = np.random.randint(-100, 2000, (amount, 2))
        # 按axis=1排序后的数组(由小到大), 
        # 第一列代表移出日期距今天数(为负说明至今未移出), 第二列为列入日期距今天数
        l1.sort(axis=1)
        
        # 构造ufunc格式函数, 计算列入日期和移出日期
        func_date = lambda day: datetime.datetime.strftime(
                datetime.date.today() - datetime.timedelta(days=day), '%Y-%m-%d %X')
        ufunc_date = np.frompyfunc(func_date, 1, 1)
        l = ufunc_date(l1)

        # 如果距今移出天数为负, 那么移出日期为空
        l[l1<0] = None

        # 剩余的70%的企业没有经营异常记录, 列入和移出日期为空
        L = np.vstack([np.array([[None, None]]*(self.n_samples-amount)), l])
        # 打乱顺序
        L = np.random.permutation(L)
        #print(L)

        self.date_abnormal_operation = L

    def reasonOfCancellation(self):
        """类别值 (注销原因), 0-没有注销, 1-原因1, 2-原因2, ..."""

        L = np.random.choice([0, 1, 2, 3, 4,], 
                self.n_samples, (0.8, 0.05, 0.05, 0.05, 0.05,))

        self.reason_cancellation = L

    def isRealtheCommecialPremises(self):
        """类别值 (经营场所是否真实), 0-不真实 1-真实"""

        L = np.random.choice([0, 1,], self.n_samples, (0.2, 0.8))

        self.is_real_commecial_premises = L

    def frezonHoldings(self):
        """冻结股权数额 (冻结股权比例)"""

        # 只有10%的企业的股权被冻结, 冻结比例是注册资本的5%到100%
        L = np.zeros(self.n_samples)
        l = np.random.choice(range(0, self.n_samples), 
                int(self.n_samples*0.1), False)
        ufunc_frezon = np.frompyfunc(lambda x: 
                x * np.random.randint(5, 100)/100, 1, 1)
        L[l] = ufunc_frezon(self.registered_capital[l])
        #print(L)

        self.frezon_holdings = L

    def totalAssets(self):
        """资产总额"""

        # 资产总额为注册资本上下浮动20%的随机值
        #print(self.registered_capital)
        func_assets = lambda x: np.random.randint(int(x * 0.8), int(x * 1.2))
        ufunc_assets = np.frompyfunc(func_assets, 1, 1)
        L = ufunc_assets(self.registered_capital)

        self.total_assets = L

    def totalLiabilities(self):
        """负债总额"""

        # 负债总额是0到资产总额1/2的随机值
        #print(self.total_assets)
        func_liabilities = lambda x: np.random.randint(0, int(x * 0.5))
        ufunc_liabilities = np.frompyfunc(func_liabilities, 1, 1)
        L = ufunc_liabilities(self.total_assets)

        # 50%的企业没有负债(随机将50%的企业的负债总额置为0)
        l = np.random.choice(range(0, self.n_samples), int(self.n_samples*0.5), False)
        L[l] = 0
        #print('负债总额: ', L)

        self.total_liabilities = L

    def totalTaxPayments(self):
        """纳税总额"""

        # 纳税总额是0到资产总额1/4的随机值
        #print(self.total_assets)
        func_tax = lambda x: np.random.randint(0, int(x * 0.25))
        ufunc_tax = np.frompyfunc(func_tax, 1, 1)
        L = ufunc_tax(self.total_assets)
        #print(L)

        self.total_tax_payments = L

    def totalDebt(self):
        """欠息总额"""

        # 欠息总额是0到负债总额1/10的随机值
        #print('负债总额: ', self.total_liabilities)
        func_debt = lambda x: np.random.randint(0, int(x * 0.1)) if x != 0 else x
        ufunc_debt = np.frompyfunc(func_debt, 1, 1)
        L = ufunc_debt(self.total_liabilities)

        # 在此基础上, 再将部分欠息总额置为0
        #print('欠息总额: ', L)

        self.total_debt = L

    def numbersOfSubordinateEnterprises(self):
        """隶属企业个数"""

        # 企业个数从0到20, 0的概率占30%, 1占20%, 2占10%, 3到10占5%
        L = np.random.choice(range(0, 11), self.n_samples, p=[0.3, 0.2, 0.1]+[0.05]*8)

        self.numbers_subordinate_enterprises = L

    def numbersOfComplaints(self):
        """月均投诉次数"""

        # 0次的概率是0.4, 1到9次的概率是0.075
        L = np.random.choice(range(0, 10), self.n_samples, 
                p=[0.4, 0.2,] + [0.075]*4 + [0.025]*4)
        #print(L)

        self.numbers_complaints = L

    def numbersOfComplaintsAndConciliation(self):
        """调节次数,投诉次数 (调解比例)"""

        #print('月均投诉次数: ', self.numbers_complaints)
        L1 = self.numbers_complaints * np.random.randint(1, 13, self.n_samples)
        #print('历史投诉次数(近一年?): ', L1)

        ufunc_int = np.frompyfunc(int, 1, 1)
        L2 = ufunc_int(L1 * np.random.random(self.n_samples))
        #print('调解次数: ', L2)

        L = np.hstack([L2[:, np.newaxis], L1[:, np.newaxis]])
        #print(L)

        self.numbers_complaints_conciliation = L

    def amountOfItemsJudicalAuction(self):
        """司法拍卖物品的金额"""

        # 拍卖金额为注册资本10%到50%的随机值
        #print(self.total_assets)
        func_auction = lambda x: np.random.randint(int(x*0.1), int(x*0.5))
        ufunc_auction = np.frompyfunc(func_auction, 1, 1)
        L = ufunc_auction(self.total_assets)

        # 20%的企业有司法拍卖, 80%的企业拍卖金额为0
        l = np.random.choice(range(0, self.n_samples), int(self.n_samples*0.8), False)
        L[l] = 0
        #print(L)

        self.amount_items_judical_auction = L

    def isCountExecutee(self):
        """类别 (是否为法院被执行人)"""

        L = np.random.choice([0, 1,], self.n_samples, (0.8, 0.2))
        #print('是否为法院被执行人: ', L)

        self.is_count_executee = L

    def industryIndex(self):
        """企业所属行业景气指数"""

        L = np.random.randint(1000, 1500, self.n_samples)
        #print('企业所属行业景气指数: ', L)

        self.industry_index = L

    def fraudScore(self):
        """法人欺诈分"""

        L = np.random.randint(0, 101, self.n_samples)
        #print('法人欺诈分: ', L)

        self.fraud_score = L

    def isInFinancialBlackList(self):
        """类别 (是否命中涉金融黑名单)"""

        L = np.random.choice([0, 1,], self.n_samples, (0.8, 0.2))
        #print('是否命中涉金融黑名单', L)

        self.is_in_financial_black_list = L


    def generatedDatasets(self):

        #'''
        self.industryOfEnterprises()
        self.districtOfEnterprises()
        self.registeredCapital()
        self.employeeNumbers()
        self.numbersOfCommecialPremises()
        self.dateOfAbnormalOperation()
        self.reasonOfCancellation()
        self.isRealtheCommecialPremises()
        self.frezonHoldings()
        self.totalAssets()
        self.totalLiabilities()
        self.totalTaxPayments()
        self.totalDebt()
        self.numbersOfSubordinateEnterprises()
        self.numbersOfComplaints()
        self.numbersOfComplaintsAndConciliation()
        self.amountOfItemsJudicalAuction()
        self.isCountExecutee()
        self.industryIndex()
        self.fraudScore()
        self.isInFinancialBlackList()
        #'''

        '''
        print('industry_enterprises: ', self.industry_enterprises.shape)
        print('city_district_enterprises: ', self.city_district_enterprises.shape)
        print('registered_capital: ', self.registered_capital.shape)
        print('employee_numbers: ', self.employee_numbers.shape)
        print('numbers_commecial_premises: ', self.numbers_commecial_premises.shape)
        print('date_abnormal_operation: ', self.date_abnormal_operation.shape)
        print('reason_cancellation: ', self.reason_cancellation.shape)
        print('is_real_commecial_premises: ', 
                self.is_real_commecial_premises.shape)
        print('frezon_holdings: ', 
                self.frezon_holdings.shape)
        print('total_assets: ', self.total_assets.shape)
        print('total_liabilities: ', self.total_liabilities.shape)
        print('total_tax_payments: ', self.total_tax_payments.shape)
        print('total_debt: ', self.total_debt.shape)
        print('numbers_subordinate_enterprises: ', 
                self.numbers_subordinate_enterprises.shape)
        print('numbers_complaints: ', self.numbers_complaints.shape)
        print('numbers_complaints_conciliation: ', 
                self.numbers_complaints_conciliation.shape)
        print('amount_items_judical_auction: ', 
                self.amount_items_judical_auction.shape)
        print('is_count_executee: ', self.is_count_executee.shape)
        print('industry_index: ', self.industry_index.shape)
        print('fraud_score: ', self.fraud_score.shape)
        print('is_in_financial_black_list: ', 
                self.is_in_financial_black_list.shape)
        '''

    def loadDatasets(self):
        """将多个数组合并为DataFrame, 并导出为csv格式文件"""

        df = pd.DataFrame({
                # 添加企业社会信用代码, 行业, 市, 区
                # enterprise_id往序号前填充满0, 转换为字符串
                'enterprise_id': pd.Series(range(1, self.n_samples+1)).map(
                    lambda x: str(x).zfill(len(str(self.n_samples)))),
                'industry': self.industry_enterprises,
                'city': self.city_district_enterprises[:, 0],
                'district': self.city_district_enterprises[:, 1],
                # 注册资本, 从业人数
                'registered_capital': self.registered_capital,
                'employees': self.employee_numbers,
                # 住所数量, 经营异常列入和移出日期
                'premises_number': self.numbers_commecial_premises,
                'date_out_abnormal': self.date_abnormal_operation[:, 0],
                'date_in_abnormal': self.date_abnormal_operation[:, 1],
                # 注销原因, 经营地址是否真实合法
                'cacellation_reason': self.reason_cancellation,
                'is_real_premise': self.is_real_commecial_premises,
                # 冻结股权数额
                'frezon_holdings': self.frezon_holdings,
                # 总资产, 总负债, 总纳税, 总欠息
                'total_assets': self.total_assets,
                'total_liabilities': self.total_liabilities,
                'total_tax': self.total_tax_payments,
                'total_debt': self.total_debt,
                # 隶属企业个数
                'sub_enterprises_number': self.numbers_subordinate_enterprises,
                # 月均投诉次数, 历史投诉次数, 接受调解次数
                'complaints_number_monthly': self.numbers_complaints,
                'conciliation_number': self.numbers_complaints_conciliation[:, 0],
                'complaints_number': self.numbers_complaints_conciliation[:, 1],
                # 司法拍卖物品金额, 是否法院被执行人
                'judical_auction_amount': self.amount_items_judical_auction,
                'is_executee': self.is_count_executee,
                # 行业指数, 法人欺诈分, 企业是否命中涉金融黑名单
                'industry_index': self.industry_index,
                'fraud_score': self.fraud_score,
                'is_financial_black_list': self.is_in_financial_black_list})
        #print(df.shape, '\n', df.head())

        df.to_csv('datasets/datasets.csv', index=False)

def computeFeatures(datasets):

    df = pd.read_csv('datasets/datasets.csv', dtype={'enterprise_id':str})
    #print(df.shape, '\n', df.head())

    #1 计算 经营地址是否变更
    df['is_change_premises'] = df['premises_number'].map(lambda x: 1 if x>=2 else 0)
    #print(df['is_change_premises'].value_counts())
    del df['premises_number']

    #2 计算 列入经营异常名录的时长
    fill_values = datetime.datetime.strftime(datetime.date.today(), '%Y-%m-%d %X')
    func_date = lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %X')
    deltadays = df['date_out_abnormal'].fillna(fill_values).map(func_date) -\
                df['date_in_abnormal'].fillna(fill_values).map(func_date)
    #print(deltadays)
    df['period_abnormal'] = deltadays.map(lambda x:x.days)
    del df['date_out_abnormal']
    del df['date_in_abnormal']
    #print(df['period_abnormal'])

    #3 计算 冻结股权占比
    #print(df['frezon_holdings'], '\n', df['registered_capital'])
    df['rate_frozon_holdings'] = df['frezon_holdings']/df['registered_capital']
    del df['frezon_holdings']
    #print(df['rate_frozon_holdings'].value_counts())

    #4 接受调解的比例
    df['rate_conciliation'] = df['conciliation_number']/df['complaints_number']
    df['rate_conciliation'].fillna(0, inplace=True)
    df['rate_conciliation'] = df['rate_conciliation'].map(lambda x: round(x, 4))
    del df['conciliation_number']
    del df['complaints_number']
    #print(df['rate_conciliation'])

    #5 添加每个样本的关注级别
    df['alert_level'] = np.random.choice(range(1, 6), df.shape[0], 
            p=[0.1, 0.1, 0.1, 0.3, 0.4])

    df = df[['enterprise_id', 'industry', 'city', 'district', 
            'registered_capital', 'employees', 'is_change_premises', 
            'period_abnormal', 'cacellation_reason', 'is_real_premise', 
            'rate_frozon_holdings', 'total_assets', 'total_liabilities', 
            'total_tax', 'total_debt', 'sub_enterprises_number', 
            'complaints_number_monthly', 'rate_conciliation', 
            'judical_auction_amount', 'is_executee', 'industry_index', 
            'fraud_score', 'is_financial_black_list', 'alert_level',]]

    df.to_csv('datasets/features.csv', index=False)

    #print(df['is_change_premises'].value_counts())

    #print(df.shape)
    #print(df.columns)


if __name__ == "__main__":

    gd = GeneratedDatasets(5000)
    gd.generatedDatasets()
    gd.loadDatasets()

    computeFeatures('datasets/datasets.csv')


