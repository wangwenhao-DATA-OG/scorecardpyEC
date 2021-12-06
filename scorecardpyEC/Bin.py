# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:47:07 2021

@author: 王文皓(wangwenhao)

"""
import numpy as np
import pandas as pd
import os
import scorecardpy as sc
from multiprocessing import Pool
from itertools import repeat


class ECBin():
    '''
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
    '''    
    
    
    def x2bin(x,binx):
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
        def _parse_bin(binx):
            bin_labels = []
            c=0
            for j in binx.iterrows():
                c+=1
                j=j[1]
                if c<10:
                    bin_label='0%d.  %s'%(c,j['bin'])
                else:
                    bin_label='%d.  %s'%(c,j['bin'])
                for tmp_bin in str(j['bin']).split('%,%'):
                    if tmp_bin=='missing':
                        bin_labels.append((1,bin_label))
                    elif '[' not in tmp_bin:
                        try:
                            tmp_bin_num = np.float(tmp_bin)
                        except:
                            tmp_bin_num = tmp_bin
                        bin_labels.append((2,tmp_bin,tmp_bin_num,bin_label)) 
                    elif '[' in tmp_bin:
                        p0=tmp_bin.index('[')
                        p1=tmp_bin.index(',')
                        p2=tmp_bin.index(')')
                        low=tmp_bin[p0+1:p1]
                        up=tmp_bin[p1+1:p2]
                        if str.lower(low) =='-inf':
                            low=-np.inf
                        low=float(low)
                        if str.lower(up) == 'inf':
                            up=np.inf
                        up=float(up)
                        bin_labels.append((3,low,up,bin_label))    
            return bin_labels
        parsed_bin = _parse_bin(binx)
        def _f1(x):
            for inter in parsed_bin:
                if inter[0]==1 and ((not x==x) or (x is None)):
                    return inter[1]
                elif inter[0]==2 and ( x==inter[2] or str(x)==inter[1]):
                    #1为字符串形式，2为数值形式
                    return inter[3]
                elif inter[0]==3 and (float(x)>=inter[1] and float(x)<inter[2]):
                    return inter[3]
            return '00.  no matching'
        return x.apply(_f1)
    

    
    def adj_bin_with_weight(bins,dat,y='y',weight='weight',ylabel={'good':0,'bad':1}):
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
        dat = dat.copy()
        dat[y] = pd.concat([pd.Series(0,dat[y].loc[dat[y]==ylabel['good']].index),pd.Series(1,dat[y].loc[dat[y]==ylabel['bad']].index)]).loc[dat.index]
        
        def _weight_info(dfg):
            d={}
            d['count']=np.around(dfg[weight].sum(),0)
            d['bad']=np.around((dfg[weight]*dfg[y]).sum(),0)
            d['good']=np.around(dfg.loc[dfg[y]==0,weight].sum(),0)
            return pd.Series(d)
        
        new_bins={}    
        for k,v in bins.items():
            if k not in dat.columns:
                continue
            tmp = ECBin.x2bin(dat[k],bins[k])
            tmp = dat[[weight,y]].merge(tmp,left_index=True,right_index=True)
            tmp = tmp.groupby(k).apply(_weight_info).reset_index().rename(columns={k:'bin'})
            def _f1(x):
                return int(x[0:x.index('.  ')])
                
            # tmp['sort']=tmp['bin'].apply(lambda x:int(x[0:x.index('.  ')]))
            tmp['sort']=tmp['bin'].apply(_f1)
            tmp['sort']=tmp['bin'].apply(_f1)
            tmp = tmp.sort_values('sort')
            del tmp['sort']
            tmp.insert(0,'variable',k)
            tmp['bin'] = tmp['bin'].apply(lambda x:x[x.index('.  ')+3:])
            tmp['count_distr']=tmp['count']/tmp['count'].sum()
            tmp['badprob']=tmp['bad']/tmp['count']
            tmp = tmp.merge(v[['bin','woe','bin_iv','total_iv','breaks','is_special_values']],left_on='bin',right_on='bin')           
            tmp[['count_distr',	'badprob','woe','bin_iv','total_iv']]=tmp[['count_distr','badprob','woe','bin_iv','total_iv']].apply(lambda x:np.round(x,4))
            new_bins[k]=tmp            
        return new_bins
    
    def get_monotonic_info(bins,cols=None):
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
        no_monotonic=[]
        increasing = []
        decreasing = []
        for var,df in bins.items():
            if cols is None or (var in cols):
                tmp1=df.query('is_special_values==False')['badprob']
                if tmp1.is_monotonic_decreasing:
                    decreasing.append(var)
                elif tmp1.is_monotonic_increasing:
                    increasing.append(var)
                else:
                    no_monotonic.append(var)
        return no_monotonic,increasing,decreasing
    

    def woebin_mp(dt,y='y',x=None,breaks_list=None,special_values=None,stop_limit=0.1,count_distr_limit=0.05,bin_num_limit=8,no_cores=None,method='tree',weight=None,step=50,ylabel={'good':0,'bad':1},**kwargs):
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
        dat = dt.copy()
        dat[y] = pd.concat([pd.Series(0,dat[y].loc[dat[y]==ylabel['good']].index),pd.Series(1,dat[y].loc[dat[y]==ylabel['bad']].index)]).loc[dat.index]
        if x is None:
            x = list(dat.loc[:,~dat.columns.isin([y,weight])].columns)
        xy = x[:]
        xy.append(y)
        
        if no_cores is None:
            no_cores = os.cpu_count()-1
        start=0
        l = len(x)
        plans=[]
        while(start<l):
            end = start+step
            if end>l:
                end=l
            plans.append((start,end))
            start = end
        
        nc = min(len(plans),no_cores)
        with Pool(nc) as pool:
            result = pool.map_async(ECBin._bin_param_temp, zip(plans,repeat((dat[xy],y,breaks_list, special_values,stop_limit,count_distr_limit,bin_num_limit,method,kwargs))))
            bins_arr = result.get()
        all_bins = {}
        for i in bins_arr:
            all_bins.update(i)
        if weight is not None:
            all_bins = ECBin.adj_bin_with_weight(all_bins,dat,y,weight)
        return all_bins 
    
    def _bin_param_temp(args):
        start,end = args[0]
        dat,y,breaks_list, special_values,stop_limit,count_distr_limit,bin_num_limit,method,kwargs = args[1]
        tmp = dat.iloc[:,range(start,end)]
        tmp = pd.concat([tmp,dat[y]],axis=1)
        bins = sc.woebin(tmp,y,breaks_list = breaks_list,special_values = special_values,stop_limit= stop_limit, count_distr_limit = count_distr_limit,bin_num_limit = bin_num_limit,method = method,**kwargs)
        return bins    
    
    
    def make_monotonic_bins_mp(dt,y='y',x=None,breaks_list = None,special_values = None,stop_limit= 0.1,bin_num_limit = 8,no_cores=None,method = 'tree',weight=None,step=50,ylabel={'good':0,'bad':1},min_distr_limit=0.02,max_distr_limit=0.2,distr_step=0.01,**kwargs):
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
        count_distr_limit = min_distr_limit
        bins = ECBin.woebin_mp(dt,y,x,breaks_list,special_values,stop_limit,count_distr_limit,bin_num_limit,no_cores,method,weight,step,ylabel,**kwargs)

        while(True):
            no_monotonic,_,_ = ECBin.get_monotonic_info(bins)
            print('剩余不单调的特征 (non-monotonic features)',no_monotonic)
            if len(no_monotonic)==0:
                break
            count_distr_limit += distr_step
            if count_distr_limit > max_distr_limit:
                break
            tmp = ECBin.woebin_mp(dt,y,no_monotonic,breaks_list,special_values,stop_limit,count_distr_limit,bin_num_limit,no_cores,method,weight,step,ylabel,**kwargs)
            bins.update(tmp)
            
        return bins,ECBin.get_monotonic_info(bins)
    
    def _woe_param_temp(args):
        start,end = args[0]
        dats,bins = args[1]
        dats_slice={}
        for k,v in dats.items():
            dats_slice[k] = v.iloc[:,range(start,end)]
        bins_slice = {k:v for k,v in bins.items() if k in list(dats_slice.values())[0].columns}
        woe_dat = {k:sc.woebin_ply(v, bins_slice) for k,v in dats_slice.items()}
        return woe_dat      
    
    def woevalue_mp(dats,bins,step=50,no_cores=None):
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
        if no_cores is None:
            no_cores = os.cpu_count()-1
        start=0
        l = len(list(dats.values())[0].columns)
        plans=[]
        while(start<l):
            end = start+step
            if end>l:
                end=l
            plans.append((start,end))
            start = end         
        nc = min(len(plans),no_cores)          
        with Pool(nc) as pool:
            result = pool.map_async(ECBin._woe_param_temp, zip(plans,repeat((dats,bins))))
            woe_dat_arr = result.get()
            
        tmp = {i:[] for i in dats.keys()}
        for woe_dat in woe_dat_arr:
            for k,v in woe_dat.items():
                tmp[k].append(v)
        
        woe_dats = {}
        for k, v in tmp.items():
            woe_dats[k] = pd.concat(v,axis=1)
        return woe_dats
                
    def export_scorecard(scorecard,file):
        '''
        导出一个scorecard
        Export a scorecard to excel
        Parameters
        ----------
        scorecard : DataFrame
            scorecardpy.scorecard的返回值
            The value returned by scorecardpy.scorecard
        
        file : str
            导出的excel文件的名字
            The name of excel to be exported

        Returns
        -------
        None.
        
        Examples
        -----------
        see also Test_SCEC.test_exp_imp_sc()

        '''
        tmp = pd.concat(scorecard)
        tmp = tmp.set_index(['variable','bin'])
        tmp.to_excel(file)    
    

    def import_scorecard(file,sheet_name=0):
        '''
        从excel导入scorecard
        Import a scorecard from excel

        Parameters
        ----------
        file : str
            文件的名称
            name of file
        sheet_name : int,str
            文件的sheet名。默认是0
            name of sheet.Default is 0.

        Returns
        -------
        card : DataFrame
            scorecardpy.scorecard的返回值
            The value returned by scorecardpy.scorecard
            
        Examples
        -----------
        see also Test_SCEC.test_exp_imp_sc()

        '''
        df_sc = pd.read_excel(file,sheet_name,index_col=(0,1))
        df_sc = df_sc.reset_index()
        card={}
        def _f1(x):
            card[x]=df_sc.loc[df_sc.variable==x]           
        pd.Series(df_sc['variable'].unique()).apply(_f1)
        return card
    

    def bins2iv(bins):
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
        if isinstance(bins,dict):
            dfbins = pd.concat(bins)
        else:
            dfbins = bins.copy()
        ivs = dfbins[['variable','total_iv']].drop_duplicates('variable').set_index('variable')['total_iv'].sort_values(ascending=False)
        return ivs
    

    def merge_missing(bins,cols=None
                      ,missing_count_distr_limit=None,missing_count_distr_limit_default=0.01
                      ,rule=None,rule_default='first'
                      ,merge_with_special_value=None,merge_with_special_value_default=False):
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
        def _get_setting(col):
            tmp1 = missing_count_distr_limit_default
            if missing_count_distr_limit is not None:
                tmp1 = missing_count_distr_limit.get(col,missing_count_distr_limit_default)
                
            tmp2 = rule_default
            if rule is not None:
                tmp2 = rule.get(col,rule_default)
                
            tmp3 = merge_with_special_value_default
            if merge_with_special_value is not None:
                if col in merge_with_special_value:
                    tmp3 = True
            return tmp1,tmp2,tmp3   
                                                

        for col,df_bin in bins.items():
            if cols is not None:
                if col not in cols:
                    continue
            
            col_missing_count_distr_limit,col_rule,col_spec = _get_setting(col)
                
            tmp = df_bin.query('bin=="missing" & count_distr <= @col_missing_count_distr_limit')
            if tmp.shape[0] == 0:
                continue
            
            cond = 'bin!="missing"'  if col_spec else 'bin!="missing" & is_special_values==False'
            df_bin_no_missing = df_bin.query(cond)
                
            if col_rule == 'first':
                loc = 0
            elif col_rule == 'last':
                loc = -1
            elif col_rule == 'nearly':
                prob = df_bin.query('bin=="missing"').iloc[0]['badprob']
                loc = df_bin_no_missing['badprob'] - prob
                loc = np.abs(loc).argmin()
                
            bin_str = df_bin_no_missing.iloc[loc]['bin']
            
            all_count = df_bin['count'].sum()
            all_bad = df_bin.bad.sum()
            all_good = df_bin.good.sum()
            df_bin.loc[df_bin.bin==bin_str,'count'] = df_bin.loc[df_bin.bin==bin_str,'count'].values+df_bin.loc[df_bin.bin=='missing','count'].values
            df_bin.loc[df_bin.bin==bin_str,'count_distr'] = df_bin.loc[df_bin.bin==bin_str,'count']/all_count
            df_bin.loc[df_bin.bin==bin_str,'good'] = df_bin.loc[df_bin.bin==bin_str,'good'].values+df_bin.loc[df_bin.bin=='missing','good'].values
            df_bin.loc[df_bin.bin==bin_str,'bad'] = df_bin.loc[df_bin.bin==bin_str,'bad'].values+df_bin.loc[df_bin.bin=='missing','bad'].values
            df_bin.loc[df_bin.bin==bin_str,'badprob'] = df_bin.loc[df_bin.bin==bin_str,'bad']/(df_bin.loc[df_bin.bin==bin_str,'bad'].values+df_bin.loc[df_bin.bin==bin_str,'good'].values)
            df_bin.loc[df_bin.bin==bin_str,'woe'] = np.log(
                    (df_bin.loc[df_bin.bin==bin_str,'bad']/all_bad)
                    /
                    (df_bin.loc[df_bin.bin==bin_str,'good']/all_good)
                )
            df_bin.loc[df_bin.bin==bin_str,'bin_iv'] = (
                (df_bin.loc[df_bin.bin==bin_str,'bad']/all_bad)
                -(df_bin.loc[df_bin.bin==bin_str,'good']/all_good))*df_bin.loc[df_bin.bin==bin_str,'woe']
            df_bin.loc[df_bin.bin==bin_str,'breaks']=df_bin.loc[df_bin.bin==bin_str,'breaks']+'%,%'+'missing'
            df_bin.loc[df_bin.bin==bin_str,'bin']=df_bin.loc[df_bin.bin==bin_str,'bin']+'%,%'+'missing'
            df_bin.drop(index=df_bin.loc[df_bin.bin=='missing'].index,inplace=True)
            df_bin['total_iv']=df_bin.bin_iv.sum()
            df_bin.reset_index(drop=True,inplace=True)

    def merge_special_values(bins,cols=None
                      ,special_value_count_distr_limit=None,special_value_count_distr_limit_default=0.01):
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
            
        special_value_count_distr_limit_default : int, optional
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
        def _get_setting(col):
            tmp1 = special_value_count_distr_limit_default
            if special_value_count_distr_limit is not None:
                tmp1 = special_value_count_distr_limit.get(col,special_value_count_distr_limit_default)               
            return tmp1                                                   

        for col,df_bin in bins.items():
            if cols is not None:
                if col not in cols:
                    continue
            
            col_special_value_count_distr_limit = _get_setting(col)
            while(True):            
                tmp = df_bin.query('is_special_values==True & count_distr <= @col_special_value_count_distr_limit')
                if tmp.shape[0] == 0:
                    break                
                tmp = tmp.iloc[tmp.count_distr.argmin()]
                prob_diff = df_bin.loc[df_bin.bin!=tmp['bin'],'badprob']-tmp['badprob']
                loc = np.abs(prob_diff).argmin()    
                bin_str = df_bin.loc[df_bin.bin!=tmp['bin'],'bin'].iloc[loc]

                all_count = df_bin['count'].sum()
                all_bad = df_bin.bad.sum()
                all_good = df_bin.good.sum()
                df_bin.loc[df_bin.bin==bin_str,'count'] = df_bin.loc[df_bin.bin==bin_str,'count'].values+df_bin.loc[df_bin.bin==tmp['bin'],'count'].values
                df_bin.loc[df_bin.bin==bin_str,'count_distr'] = df_bin.loc[df_bin.bin==bin_str,'count']/all_count
                df_bin.loc[df_bin.bin==bin_str,'good'] = df_bin.loc[df_bin.bin==bin_str,'good'].values+df_bin.loc[df_bin.bin==tmp['bin'],'good'].values
                df_bin.loc[df_bin.bin==bin_str,'bad'] = df_bin.loc[df_bin.bin==bin_str,'bad'].values+df_bin.loc[df_bin.bin==tmp['bin'],'bad'].values
                df_bin.loc[df_bin.bin==bin_str,'badprob'] = df_bin.loc[df_bin.bin==bin_str,'bad']/(df_bin.loc[df_bin.bin==bin_str,'bad'].values+df_bin.loc[df_bin.bin==bin_str,'good'].values)
                df_bin.loc[df_bin.bin==bin_str,'woe'] = np.log(
                        (df_bin.loc[df_bin.bin==bin_str,'bad']/all_bad)
                        /
                        (df_bin.loc[df_bin.bin==bin_str,'good']/all_good)
                    )
                df_bin.loc[df_bin.bin==bin_str,'bin_iv'] = (
                    (df_bin.loc[df_bin.bin==bin_str,'bad']/all_bad)
                    -(df_bin.loc[df_bin.bin==bin_str,'good']/all_good))*df_bin.loc[df_bin.bin==bin_str,'woe']
                df_bin.loc[df_bin.bin==bin_str,'breaks']=df_bin.loc[df_bin.bin==bin_str,'breaks']+'%,%'+tmp['breaks']
                df_bin.loc[df_bin.bin==bin_str,'bin']=df_bin.loc[df_bin.bin==bin_str,'bin']+'%,%'+tmp['bin']
                df_bin.drop(index=df_bin.loc[df_bin.bin==tmp['bin']].index,inplace=True)
                df_bin['total_iv']=df_bin.bin_iv.sum()
                df_bin.reset_index(drop=True,inplace=True)    