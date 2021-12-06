# -*- coding:utf-8 -*- 

from .Bin import ECBin

"""
Created on Wed Aug 11 09:47:07 2021

@author: 王文皓(wangwenhao)
功能描述
为评分卡项目https://github.com/ShichenXie/scorecardpy提供强化组件。可以为评分卡开发人员大幅提高开发效率
1.为woe分箱和woe转换提供内存优化和多进程计算的解决方案，使得超高维变量的分箱和woe转换得以运行出结果，并提升运行速度
2.通过样本权重调整分箱信息
3.提供强制单调分箱的函数。通过寻找最优的count_distr_limit实现，而不是简单的合并不单调的相邻箱
4.提供业务合理前提下合并missing的函数，避免了写大量breaks的繁琐。
5.提供业务合理前提下合并特殊值分箱的函数，避免了写大量breaks的繁琐。 


Function Description
Providing some enhanced components for famous score card project https://github.com/ShichenXie/scorecardpy .These enhanced components can greatly promote work efficiency.
1.Providing memory optimization and multiprocess computing solutions to woe_bin and woebin_ply.It make woe_bin and woebin_ply transforming feasible under hyper-high dimensional data and lift computing speed.
2.Adjusting infomations of bins by sample weights.
3.Providing a forcing monotonic bins function.Instead of merging two neighbouring non-monotonic bins,searching optimal count_distr_limit to get monotonic bins.
4.Providing a merging missing function under according with business that aviods writting many breaks.
5.Providing a merging special value bin function under according with business that aviods writting many breaks.
"""

__version__ = '1.1.1'

__all__ = (
    ECBin
)
