# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:08:40 2021

@author: man Yip
"""

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