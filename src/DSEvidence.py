# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:38:42 2019

@author: Ale
"""

class DSEvidence:
    @staticmethod
    def k_coefficient(p1, p2):
        assert len(p1) == len(p2)
        n = len(p1)
        s = 0.0
        
        for i in range(n):
            s += sum([x*p1[i] for x in (p2[:i] + p2[i+1:])])
        
        return s

    @staticmethod
    def get_joint_mass(p1, p2):
        assert len(p1) == len(p2)
        
        n = len(p1)
        sums = []
        K = DSEvidence.k_coefficient(p1, p2)
        
        for i in range(n):
            sums.append(round((p1[i]*p2[i])/K, 2))
        
        res = [round(x/sum(sums), 2) for x in sums]
                
        return res
