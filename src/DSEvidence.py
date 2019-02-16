# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:38:42 2019

@author: Ale
"""

import numpy as np
import tensorflow as tf

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
            #print(p1[i], p2[i])
            sums.append((p1[i]*p2[i])/K)
        
        res = np.array([x/sum(sums) for x in sums], dtype=np.float32)

        res.reshape(1, -1)
        return res

    """
    @staticmethod
    def tf_k_coefficient(t1, t2):
        assert t1.shape == t2.shape

        def _aux_k_coff(t):
            return DSEvidence.k_coefficient(t[0], t[1])

        return tf.map_fn(_aux_k_coff, elems=(t1, t2), dtype=tf.float32)
    """

    @staticmethod
    def tf_get_joint_mass(t1, t2):
        assert t1.shape == t2.shape

        def _aux_joint(t):
            return DSEvidence.get_joint_mass(list(t[0].numpy()), list(t[1].numpy()))

        return tf.map_fn(_aux_joint, elems=(t1, t2), dtype=tf.float32)
