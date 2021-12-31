# scorecardpyEC

# 功能描述（Function Description）
    为评分卡项目https://github.com/ShichenXie/scorecardpy提供额外的工具组件。可以帮助评分卡开发人员提高开发效率
    1.为woe分箱和woe转换提供内存优化和多进程计算的解决方案，使得超高维特征的分箱和woe转换得以运行出结果，并提升运行速度
    2.通过样本权重调整分箱信息
    3.提供强制单调分箱的函数。通过寻找最优的count_distr_limit实现，而不是简单的合并不单调的相邻箱，这样可以更精细的合并箱，使得合并箱操作所带来的信息损失减少。
    4.提供业务合理前提下合并missing分箱的函数，避免了写大量breaks的繁琐。 
    5.提供业务合理前提下合并特殊值分箱的函数，避免了写大量breaks的繁琐。 
    
    Function Description
    Providing some enhanced components for famous score card project https://github.com/ShichenXie/scorecardpy .These enhanced components can promote work efficiency.
    1.Providing memory optimization and multiprocess computing solution to scorecardpy.woebin and scorecardpy.woebin_ply.It make woe_bin and woebin_ply transforming feasible under hyper-high dimensional data and lift computing speed.
    2.Adjusting infomations of bin by sample weights.
    3.Providing a forcing monotonic bin function.It makes merging bins more carefully and reduces infomation loss that instead of merging two neighbouring non-monotonic bins,searching optimal count_distr_limit to get monotonic bin.
    4.Providing a merging missing bin function under according with business that aviods writting many breaks.
    5.Providing a merging special value bin function under according with business that aviods writting many breaks.

# 安装（Install）
pip install scorecardpyEC

# 使用说明 （Usage）:
```
from scorecardpyEC import ECBin
import numpy as np
import pandas as pd
import scorecardpy as sc
class Test_SCEC():
    def test_x2bin():
        test_data = pd.DataFrame({'x1':[1,0,0.5,-1,np.nan],'x2':[1,0.4,np.nan,0,4]})
        test_bins={}
        test_bins['x1'] = pd.DataFrame({'variable':['x1','x1','x1','x1']
                                        ,'bin':['[0,0.4)%,%missing','-1','[0.4,0.7)','[0.7,Inf)']})   
        test_bins['x2'] = pd.DataFrame({'variable':['x2','x2','x2','x2']
                                        ,'bin':['missing','[0,1.0)','[1.0,2.0)','[2.0,Inf)']})
        tmp = test_data.apply(lambda s:ECBin.x2bin(s,test_bins[s.name]))
        print(pd.concat(test_bins))
        print(test_data)
        print(tmp)
        
        
    def test_adj_bin_with_weight():
        dat = sc.germancredit().iloc[:,np.arange(-6,0)]
        np.random.seed(0)
        dat['weight'] = np.random.randint(1,4,1000)
        
        dat.loc[dat.index.isin(np.arange(0,100)),'number.of.existing.credits.at.this.bank']=np.nan
        dat.to_excel('dat.xlsx')
        sv = {'number.of.existing.credits.at.this.bank':[1]}
        bins = sc.woebin(dat.loc[:,dat.columns!='weight'], y="creditability",special_values=sv)
        new_bins = ECBin.adj_bin_with_weight(bins,dat,y="creditability",ylabel={'good':'good','bad':'bad'})
        pd.concat(bins).to_excel('bins.xlsx')
        pd.concat(new_bins).to_excel('new_bins.xlsx')
        
    def test_get_monotonic_info():
        dat = sc.germancredit()
        bins = sc.woebin(dat, y="creditability")
        pd.concat(bins).to_excel('bins.xlsx')
        no_monotonic,increasing,decreasing = ECBin.get_monotonic_info(bins,['present.residence.since'
                            ,'savings.account.and.bonds'
                            ,'installment.rate.in.percentage.of.disposable.income'
                            ,'number.of.existing.credits.at.this.bank'
                            ,'age.in.years'
                            ,'duration.in.month'
                            ,'credit.amount'])
        print(no_monotonic,increasing,decreasing)
        
    def test_woebin_mp():
        dat = sc.germancredit()
        # dat['weight'] = np.random.randint(1,4,1000)
        # ,weight='weight'
        bins1 = ECBin.woebin_mp(dat, y="creditability",step=10,ylabel={'good':'good','bad':'bad'})
        pd.concat(bins1).to_excel('bins1.xlsx')
        
        bins2= sc.woebin(dat,y="creditability")
        pd.concat(bins2).to_excel('bins2.xlsx')
        
    def test_make_monotonic_bins_mp():
        dat = sc.germancredit()
        bins,_ = ECBin.make_monotonic_bins_mp(dat, y="creditability",ylabel={'good':'good','bad':'bad'},step=5)
        pd.concat(bins).to_excel('bins.xlsx')
        
    def test_woevalue_mp():
        dat = sc.germancredit()
        from sklearn.model_selection import train_test_split
        X = dat.loc[:,dat.columns!='creditability']
        y = dat['creditability']
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        dats={'train':X_train,'test':X_test}
        bins = ECBin.woebin_mp(pd.concat([X_train,y_train],axis=1), y="creditability",step=10,ylabel={'good':'good','bad':'bad'})
        woe_dats = ECBin.woevalue_mp(dats,bins,step=4)
        print(woe_dats['train'])
        print(woe_dats['test'])

        
    def test_exp_imp_sc():
        dat = sc.germancredit()
        X = dat.iloc[:,0:4]
        y = dat.creditability.apply(lambda a:1 if a=='bad' else 0)
        dat = pd.concat([X,y],axis=1)
        bins = ECBin.woebin_mp(dat, y="creditability",step=10)
        X = sc.woebin_ply(X, bins)
        X = X.rename(columns=X.columns.to_series().apply(lambda x:x.replace('_woe','')))
        from statsmodels.genmod.generalized_linear_model import GLM
        import statsmodels.api as sm
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import logit 
        glm = GLM(y, sm.add_constant(X),family = Binomial(link=logit))
        clf = glm.fit()  
        clf.intercept_=[clf.params.const]
        clf.coef_=[clf.params[1:]]
        card = sc.scorecard(bins, clf,X.columns,basepoints_eq0=True)
        print(card)
        print('=======================')
        ECBin.export_scorecard(card,'sc.xlsx')
        card = ECBin.import_scorecard('sc.xlsx')
        print(card)
        
    def test_merge_missing_breaks():
        dat = sc.germancredit()
        X = dat.iloc[:,0:4]
        X = X.apply(lambda x: x.astype(str) if x.dtype.name=='category' else x)
        np.random.seed(1)
        X.iloc[np.random.randint(X.shape[0],size=10),0]=np.nan
        X.iloc[np.random.randint(X.shape[0],size=20),1]=np.nan
        X.iloc[np.random.randint(X.shape[0],size=30),2]=np.nan
        X.iloc[np.random.randint(X.shape[0],size=40),3]=np.nan
        y = dat.creditability.apply(lambda a:1 if a=='bad' else 0)
        dat = pd.concat([X,y],axis=1)
        ######not merging####
        sv = {'purpose':['furniture/equipment%,%business','education%,%others']
              ,'duration.in.month':[4,5]}
        bins1 = sc.woebin(dat, y="creditability",special_values=sv)
        pd.concat(bins1).to_excel('base.xlsx')  
        
        #######sample1##################
        ECBin.merge_missing(bins1,missing_count_distr_limit_default=0.03,rule_default='first',merge_with_special_value_default=False)
        pd.concat(bins1).to_excel('bins_missing_merge_first.xlsx')
        
        
        #######sample2##################
        bins1 = sc.woebin(dat, y="creditability",special_values=sv)
        ECBin.merge_missing(bins1,missing_count_distr_limit_default=0.03,rule_default='nearly',merge_with_special_value_default=False)
        pd.concat(bins1).to_excel('bins_missing_merge_nearly.xlsx')
        
        
        #######sample3###############
        missing_count_distr_limit={'purpose':0.05,'credit.history':0.04}
        rule={'purpose':'nearly','credit.history':'last'}
        merge_with_special_value=['purpose']
        bins1 = sc.woebin(dat, y="creditability",special_values=sv)
        ECBin.merge_missing(bins1
                      ,missing_count_distr_limit=missing_count_distr_limit
                      ,rule=rule
                      ,merge_with_special_value=merge_with_special_value)
        pd.concat(bins1).to_excel('bins_missing_merge_comb.xlsx')
        
    def test_merge_special_values():
        dat = sc.germancredit()
        X = dat.iloc[:,0:3]
        X = X.apply(lambda x: x.astype(str) if x.dtype.name=='category' else x)
        np.random.seed(1)
        X.iloc[np.random.randint(X.shape[0],size=20),0]=np.nan
        y = dat.creditability.apply(lambda a:1 if a=='bad' else 0)
        dat = pd.concat([X,y],axis=1)
        ######not merging####
        sv = {'duration.in.month':[4,5]
              ,'credit.history':['delay in paying off in the past']}
        bins1 = sc.woebin(dat, y="creditability",special_values=sv)
        pd.concat(bins1).to_excel('base.xlsx')
        
        #######sample1##################
        ECBin.merge_special_values(bins1,special_value_count_distr_limit={'status.of.existing.checking.account':0.02},special_value_count_distr_limit_default=0.01)
        pd.concat(bins1).to_excel('merge_special_values.xlsx')
    
if __name__ == '__main__':
    Test_SCEC.test_x2bin()
    Test_SCEC.test_adj_bin_with_weight()
    Test_SCEC.test_get_monotonic_info()
    Test_SCEC.test_woebin_mp()
    Test_SCEC.test_make_monotonic_bins_mp()
    Test_SCEC.test_woevalue_mp()
    Test_SCEC.test_exp_imp_sc()
    Test_SCEC.test_merge_missing_breaks()
    Test_SCEC.test_merge_special_values()
```
# 文档和API Documnt and API
## ECBin
### x2bin
        '''
        将一个向量（特征项）的每一个元素按照分箱信息进行转换。
        如果需要将全部特征转换成分箱标签，则可以使用：
        all_df.apply(lambda s:ECBin.x2bin(s,all_bin[s.name]))
        
        Transform a vector (or a feature) to bin label with information of woe_bin.
        If transofming all futures to bin label,uses:
         all_data.apply(lambda s:ECBin.x2bin(s,all_bin[s.name]))   
        
        Parameters    
        ----------
        x:pandas.Series
        一个向量（特征项） 
        a vector(or a feature)
        
        binx:pandas.DataFrame
        x所对应的分箱信息。scorecardpy.woebin()返回值的一个元素。
        Information of bin according with x,an item returned by scorecardpy.woebin().
        
        Returns
        ----------
        pandas.Series
        转换后的分箱标签
        transformed bin label
        
        Examples
        -----------
        see also Test_SCEC.test_x2bin()
        
        '''
### adj_bin_with_weight
        '''
        按照样本权重，把各分箱的样本数进行还原。
        用权重还原并不是改变分箱的breaks。它只是改变count，count_distr，good，bad，badprob，而bin,woe,bin_iv,total_iv,breaks,is_special_values字段保持原值不变
        其目的是依据采样的权重来还原真实的概率。
        使用此方法时应该注意，如果采样的权重只是依据好坏标签，而不是依据该特征，则还原可能不精确，因为对该特征来说，其抽样的概率可能与好坏标签的抽样概率并不一致。
        
        Restore the sample count to original data as weight.
        It`s not changing breaks of bin but changing count,count_distr,good,bad,badprob only and keeping the value of bin,woe,bin_iv,total_iv,breaks,is_special_values.
        The target is restoring true probability with sample weight.
        Note: when sampling if not depending on variables but depending on bad or good flag only,the restoring result may be inexact,due to the sampling probability of the variable is not same with the flag`s of good or bad.
        
        Parameters    
        ----------
        bins:dict
        scorecardpy.woebin()的返回值
        bins returned by scorecardpy.woebin()
        
        dat:Pandas.DataFrame
        全部特征列 + Y标签列 + 权重列
        all features add y and weight columns
        
        y:str
        y标签的列名，默认为'y'
        the name of good or bad target column.Default 'y'
        
        weight：str
        权重的列名。默认 'weight'
        the name of weight column. Default 'weight' 
              
        ylabel:dict 
        dict中必须包含两个key，good：好样本点的标记。bad：坏样本点的标记。 默认 {'good':0,'bad':1}
        Two keys must be in dict,good:the flag of good sample.bad:the flag of bad sample.Default is {'good':0,'bad':1}
        
        
        Returns
        ----------
        dict
        按权重调整后的分箱
        the bins adjusted with weight
        
        Examples
        -----------
        see also Test_SCEC.test_adj_bin_with_weight()
     
        '''
### get_monotonic_info
        '''
        返回woe_bin的各个特征的单调性情况
        
        Parameters    
        ----------
        bins:dict
        scorecardpy.woebin()的返回值
        value returned by scorecardpy.woebin()
        
        cols:array
        需要考察单调性的列名的集合。默认值为None，意即bins中所有的列 
        columns list to be calculated for monotonic.Default is None that means all columns in bins is included
        
        Returns
        ----------
        tuple(array,array,array)
        分别返回非单调性的列名，单调递增的列名，单调递减的列名
        tuple(no_monotonic_columns,increasing_columns,decreasing_columns)
        
        Examples
        -----------
        see also Test_SCEC.test_get_monotonic_info()
     
        ''' 
        
### woebin_mp
        '''
        scorecardpy.woebin()的多进程版本，支持windows。返回值同scorecardpy.woebin()
        
        Parameters    
        ----------
        dt:dataframe
        同scorecardpy.woebin的dt
        same with dt in parameters list of scorecardpy.woebin
               
        y:str
        同scorecardpy.woebin的y
        same with y in parameters list of scorecardpy.woebin
        
        x:array
        同scorecardpy.woebin的x
        same with x in parameters list of scorecardpy.woebin
        
        breaks_list:dict
        同scorecardpy.woebin的breaks_list
        same with breaks_list parameters list of scorecardpy.woebin
        
        special_values:dict
        同scorecardpy.woebin的special_values
        same with special_values parameters list of scorecardpy.woebin
        
        stop_limit:float
        同scorecardpy.woebin的stop_limit
        same with stop_limit parameters list of scorecardpy.woebin
        
        count_distr_limit:float
        同scorecardpy.woebin的count_distr_limit
        same with count_distr_limit parameters list of scorecardpy.woebin
        
        bin_num_limit:int
        同scorecardpy.woebin的bin_num_limit
        same with bin_num_limit parameters list of scorecardpy.woebin
        
        no_cores:int
        使用的CPU核数。默认为os.cpu_count()-1。实际使用CPU核数可能会少于no_cores，见step
        used cpu cores number.Default is os.cpu_count()-1.The actually used cpu cores may be less than no_cores,seeing step
        
        method:str
        同scorecardpy.woebin的method
        same with method parameters list of scorecardpy.woebin
        
        weight:str
        权重的列名。默认为None。如果不为None，则ECBin.woebin_mp运行出分箱结果后会自动调用ECBin.adj_bin_with_weight
        注：用权重还原并不会改变分箱的breaks。它只会改变count，count_distr，good，bad，badprob，而bin,woe,bin_iv,total_iv,breaks,is_special_values字段保持原值不变，参考ECBin.adj_bin_with_weight的weight参数
        
        The column name of weight.Default is None.If not None,call ECBin.adj_bin_with_weight after getting the result of woe_bin.
        Note:It`s not changing breaks of bin but changing count,count_distr,good,bad,badprob only and keeping the value of bin,woe,bin_iv,total_iv,breaks,is_special_values.
        see the weight parameter of ECBin.adj_bin_with_weight
        
        step:int
        每个进程一次执行的特征数量。默认为50
        它的作用是平衡内存开销和进程切换。step越高内存开销越大，但进程切换次数越少。如果对程序运行原理不了解，则不需要调整该参数，保留默认值即可。
        cpu的实际使用核数不仅与no_cores有关，还与step有关。举例来说，如果step=50，有500个特征，如果no_cores>10，则实际只使用了10个核cpu。如果no_cores<10，则实际使用的核数为no_cores所设定的值。
        
        The number of features excuted by one process.Default is 50.
        It is used to make balance between memory usage and process swap.The bigger step value bring bigger memory usage and less process swap.If not understanding theory of program running,keeping the default value is a good choice.
        The number of cpu cores depends on no_cores and step.For example. If step is 50,the number of features is 500,only 10 cores are used when no_cores >10 and no_cores core are used when no_cores<10.
        
        
        ylabel:dict
        好坏标签的标识，默认为：{'good':0,'bad':1}
        The identity of good or bad.Default is {'good':0,'bad':1}
        
        kwargs:dict
        scorecardpy.woebin的其他参数可以通过kwargs传入
        The other parameters in scorecardpy.woebin is packed in kwargs
        
        Returns
        ----------
        dict
        同scorecardpy.woebin的返回值
        same with the value returned by scorecardpy.woebin
        
        Examples
        -----------
        see also Test_SCEC.test_woebin_mp()
     
        '''

### make_monotonic_bins_mp
        '''
        提供多进程的强制单调分箱的函数。
        通过寻找最优的count_distr_limit实现，而不是简单的合并不单调的相邻箱，这样可以更精细的合并箱，使得合并箱所带来的信息损失减少。
        
        Providing a forcing monotonic bin function.It makes merging bins more carefully and reduces infomation loss that Instead of merging two neighbouring non-monotonic bins,searching optimal count_distr_limit to get monotonic bin.
        
        Parameters    
        ----------
        dt:dataframe
        同scorecardpy.woebin的dt
        same with dt in parameters list of scorecardpy.woebin
               
        y:str
        同scorecardpy.woebin的y
        same with y in parameters list of scorecardpy.woebin
        
        x:array
        同scorecardpy.woebin的x
        same with x in parameters list of scorecardpy.woebin
        
        breaks_list:dict
        同scorecardpy.woebin的breaks_list
        same with breaks_list parameters list of scorecardpy.woebin
        
        special_values:dict
        同scorecardpy.woebin的special_values
        same with special_values parameters list of scorecardpy.woebin
        
        stop_limit:float
        同scorecardpy.woebin的stop_limit
        same with stop_limit parameters list of scorecardpy.woebin
        
        bin_num_limit:int
        同scorecardpy.woebin的bin_num_limit
        same with bin_num_limit parameters list of scorecardpy.woebin
        
        no_cores:int
        使用的CPU核数。默认为os.cpu_count()-1。实际使用CPU核数可能会少于no_cores，见step
        used cpu cores number.Default is os.cpu_count()-1.The actually used cpu cores may be less than no_cores,seeing step
        
        method:str
        同scorecardpy.woebin的method
        same with method parameters list of scorecardpy.woebin
        
        weight:str
        权重的列名。默认为None。如果不为None，则ECBin.woebin_mp运行出分箱结果后会自动调用ECBin.adj_bin_with_weight
        注：用权重还原并不会改变分箱的breaks。它只会改变count，count_distr，good，bad，badprob，而bin,woe,bin_iv,total_iv,breaks,is_special_values字段保持原值不变，参考ECBin.adj_bin_with_weight的weight参数
        
        The column name of weight.Default is None.If not None,calling ECBin.adj_bin_with_weight after getting the result of woe_bin.
        Note:It`s not changing breaks of bin but changing count,count_distr,good,bad,badprob only and keeping the value of bin,woe,bin_iv,total_iv,breaks,is_special_values.
        see the weight parameter of ECBin.adj_bin_with_weight
        
        step:int
        每个进程一次执行的特征数量。默认为50
        它的作用是平衡内存开销和进程切换。step越高内存开销越大，但进程切换次数越少。如果对程序运行原理不了解，则不需要调整该参数，保留默认值即可。
        cpu的实际使用核数不仅与no_cores有关，还与step有关。举例来说，如果step=50，有500个特征，如果no_cores>10，则实际只使用了10个核cpu。如果no_cores<10，则实际使用的核数为no_cores所设定的值。
        
        The number of features excuted by one process.Default is 50.
        It is used to make balance between memory usage and process swap.The bigger step value bring bigger memory usage and less process swap.If not understanding theory of program running,keeping the default value is a good choice.
        The number of cpu cores depends on no_cores and step.For example. If step is 50,the number of features is 500,only 10 cores are used when no_cores >10 and no_cores core are used when no_cores<10.
        
        
        ylabel:dict
        好坏标签的标识，默认为：{'good':0,'bad':1}
        The identity of good or bad.Default is {'good':0,'bad':1}
        
        
        min_distr_limit:float 
        最小的count_distr_limit。默认为0.02
        The min of count_distr_limit.Default is 0.02
        
        max_distr_limit:float
        最大的count_distr_limit。默认为0.2
        The max of count_distr_limit.Default is 0.2
        
        distr_step:float
        当还有特征是不单调的分箱时，在上一次的count_distr_limit的基础上加上distr_step后，对不单调的特征重新进行分箱运算。默认值为0.01
        When having remaining non-monotonic variable bins,caculate bins again for this variable with adding distr_step to count_distr_limit last time.  Default is 0.01

        kwargs:dict
        scorecardpy.woebin的其他参数可以通过kwargs传入
        The other parameters in scorecardpy.woebin is packed in kwargs
        
        Returns
        ----------
        dict
        同scorecardpy.woebin的返回值
        same with the value returned by scorecardpy.woebin 
        
        Examples
        -----------
        see also Test_SCEC.test_make_monotonic_bins_mp()
        
        '''
        
### woevalue_mp
        '''
        多进程的woe转换函数。

        Parameters
        ----------
        dats : dict
            多个数据集.例如{'train':train_df,'test':test_df}
            many datas,e.g.{'train':train_df,'test':test_df}
            
        bins : dict
            woebin
            
        step:int
            每个进程一次执行的特征数量。默认为50
            它的作用是平衡内存开销和进程切换。step越高内存开销越大，但进程切换次数越少。如果对程序运行原理不了解，则不需要调整该参数，保留默认值即可。
            cpu的实际使用核数不仅与no_cores有关，还与step有关。举例来说，如果step=50，有500个特征，如果no_cores>10，则实际只使用了10个核cpu。如果no_cores<10，则实际使用的核数为no_cores所设定的值。
        
            The number of features excuted by one process.Default is 50.
            It is used to make balance between memory usage and process swap.The bigger step value bring bigger memory usage and less process swap.If not understanding theory of program running,keeping the default value is a good choice.
            The number of cpu cores depends on no_cores and step.For example. If step is 50,the number of features is 500,only 10 cores are used when no_cores >10 and no_cores core are used when no_cores<10.
        
        no_cores:int
            使用的CPU核数。默认为os.cpu_count()-1。实际使用CPU核数可能会少于no_cores，见step
            used cpu cores number.Default is os.cpu_count()-1.The actually used cpu cores may be less than no_cores,seeing step

        Returns
        -------
        woe_dats : dict
            转化后的全部数据集
            transformed all dats
            
        Examples
        -----------
        see also Test_SCEC.test_woevalue_mp()

        '''
        
### bins2iv
        '''
        从woe_bin中提取出各个特征的iv值
        Getting IV of variables from woebin

        Parameters
        ----------
        bins : dict
            同scorecardpy.woebin的返回值
            same with the value returned by scorecardpy.woebin 

        Returns
        -------
        ivs : pandas.Series
            特征的IV值
            IV of variables
        '''
        
### merge_missing
        '''
        合并missing的函数。将小于missing_count_distr_limit的missing分箱合并到其他分箱中
        A function for merging missing.Merge the missing bin less than missing_count_distr_limit to other bins.

        Parameters
        ----------
        bins : dict
            原始的待合并missing的woebin.该方法执行完后bins会被更改
            Original woebin to be merged with missing bin.The bins will be updated after running this function
            
            
        cols : list
            需要合并的特征名。默认是None，即bins中全部的特征
            The columns to be merged.The default is None means that all features in bins is included
            
        missing_count_distr_limit : dict, optional
            {'特征':最小缺失值占比,...}，如果一个特征的缺失值占比小于指定的值，则该特征的缺失分箱将被合并到其他分箱。如果特征不在该dict里，则其最小的missing占比为missing_count_distr_limit_default。默认值是None，代表所有列的最小占比全部为missing_count_distr_limit_default。
            {'col':min distribution of missing value,...}.If the missing distribution is less than the pointting value,this missing bin of the feature will be merged to other bin.If a feature not in dict,the min distribution of missing value of that feature is set by missing_count_distr_limit_default. The default is None means that the min distribution of all features is missing_count_distr_limit_default
            
        missing_count_distr_limit_default : float, optional
            如果一个特征没有在missing_count_distr_limit被指定其最小缺失值占比，则其为missing_count_distr_limit_default。默认值是0.01
            If not pointted in missing_count_distr_limit,the min distribution of this feature equals missing_count_distr_limit_default. The default is 0.01.
            
        rule : dict, optional
            {'特征':'规则',...}，每个特征的合并规则。rule的取值有{'first','last','nearly'}。'first':与第一个分箱合并。'last':与最后一个分箱合并。'nearly':'与概率最接近的一个分箱合并'。如果某个特征不在rule中，则其规则被设置成rule_default。默认值为None，即所有特征的rule全部为rule_default。
            {'col':'rule',...},the merge rule for every feature. The permitted value of rule are in {'first','last','nearly'}.'first':merge with first bin.'last':merge with last bin.'nearly':merge with bin has a nearest probility with missing bin.If not being in rule,the feature is set by rule_default. The default is None means that the rule of all features is rule_default.
            
        rule_default : str, optional
            rule_default的取值有{'first','last','nearly'}。如果某个特征没有被配置在rule中，则其rule为rule_default所指定的值。默认为'first'
            The permitted value of rule_default are in {'first','last','nearly'}.If not set in rule,the rule of this feature are set by rule_default.The default is 'first'.
            
        merge_with_special_value : list, optional
            如果允许missing分箱与其他非missing的特殊值分箱合并，则记录在merge_with_special_value列表里。注：允许与特殊值分箱合并是指：在计算合并时，特殊值分箱与非特殊值分箱都参与计算。并不是指只在特殊值分箱里选取待合并的分箱。默认是None，即所有特征是否允许与特殊值合并全部由merge_with_special_value_default设置。
            if permitting missing bins to merge with non-missing special values bins,record the feature in merge_with_special_value.Note:What merging with special values bins means is that used by specical values bins and non specical values bins rather than only by specical values for calculating. The default is None means all features are set by merge_with_special_value_default to decide merging with special values or not.
            
        merge_with_special_value_default : boolean, optional
            如果特征没有出现在merge_with_special_value中，则是否可以将missing分箱合并到其他非missing的特殊值分箱就由merge_with_special_value_default指定，True为可以合并到特殊值分箱，False为不可以。默认为False
            If not in merge_with_special_value,permitting missing bin to be merged with non-missing special value bin is set by merge_with_special_value_default.True:merge with special value bin.False:not merge with special value bin. The default is False.

        Returns 
        -------
        None.
        该方法执行完后bins会被更改
        The bins will be updated after running this function.
        
        Examples
        -----------
        see also Test_SCEC.test_merge_missing_breaks()

        '''
        
### merge_special_values
        '''
        合并特殊值。当特殊值的占比小于指定值时，将其合并到与其坏客率接近的分箱
        Merge special values.When the count_distr of special values less than a value you point,merge to a bin having a nearly badprob with special values

        Parameters
        ----------
        bins : dict
            原始的待合并missing的woebin.该方法执行完后bins会被更改
            Original woebin to be merged with missing bin.The bins will be updated after running this function
            
        cols : array,list
            需要合并的特征名。默认是None，即bins中全部的特征
            The columns to be merged.The default is None means that all variables in bins are included.
            
        special_value_count_distr_limit : dict, optional
            {特征名:最小占比,...}.指定每个列的最小占比，当小于该占比时，该列就需要被合并。没有出现在这个dict里的特征，则最小占比为special_value_count_distr_limit_default。默认是None，即所有特征的最小占比全部为special_value_count_distr_limit_default。
            {col_name:min of count_distr,...}.Point a value for every columns individually.When less than this value,this feature need to be merged.The min distribution of column not in dict is special_value_count_distr_limit_default.The default is None means that the min value of distribution of all columns is set by special_value_count_distr_limit_default.
            
        special_value_count_distr_limit_default : float, optional
            默认的最小特殊值占比，如果没有在special_value_count_distr_limit中指定该列的最小特殊值占比，则该列的最小特殊值占比为special_value_count_distr_limit_default。默认是0.01
            If not pointting a min of special value count_distr to a feature,it will be set to special_value_count_distr_limit_default . The default is 0.01.

        Returns
        -------
        None.
        该方法执行完后bins会被更改
        The bins will be updated after running this function.
            
        Examples
        -----------
        see also Test_SCEC.test_merge_special_values()

        '''
