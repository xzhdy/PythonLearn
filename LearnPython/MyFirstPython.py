# encoding=utf-8
# import json
# from pandas import DataFrame, Series
# import pandas as pd
# import numpy as np
# import pylab as pl
# from numpy.distutils.system_info import agg2_info
#  
#  
# def get_counts(sequenue):
#     counts = {}
#     for x in sequenue:
#         if x in counts:
#             counts[x] += 1
#         else:
#             counts[x] = 1
#     return counts
# def TopCouns(count_dic, n = 10):
#     value_key_pair = [(count, tz) for tz, count in count_dic.items()]       
#     value_key_pair.sort()
#     return value_key_pair[-n:]
#     
#     
# path = 'test.text'
# records = [json.loads(line) for line in open(path)]
# # print(records[0])
# # print(records[0]['tz'])
# time_zones = [rec['tz'] for rec in records if 'tz' in rec]
# # print(time_zones[0])
# counts = get_counts(time_zones)
# tz_sorted = TopCouns(counts)
# # print(tz_sorted)
# frame = DataFrame(records) 
# tz_countsByFrame = frame['tz'].value_counts()
# # print(tz_countsByFrame[:10])
# clean_tz = frame['tz'].fillna('Missing')
# clean_tz[clean_tz == ''] = 'Unknown'
# tz_countsByFrame = clean_tz.value_counts()
# # pl.plot(tz_countsByFrame[:10] , kind = 'barh', rot = 0)
# # tz_countsByFrame[:10].plot(kind = 'barh', rot = 0)
# result = Series([x.split()[0] for x in frame.a.dropna()])
# print(result[:5])
# cframe = frame[frame.a.notnull()]
# operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
# by_tz_os = cframe.groupby(['tz', operating_system])
# 
# agg_counts = by_tz_os.size().unstack().fillna(0)
# indexer = agg_counts.sum(1).argsort()
# count_subset = agg_counts.take(indexer)[-10:]
# count_subset.plot(kind = 'barh', stacked = True)
# normed_subset = count_subset.div(count_subset.sum(1), axis = 0)
# normed_subset.plot(kind = 'barh', stacked = True)
# pl.show()
# pl.show()
# print(tz_countsByFrame[:10])



# import jieba
# from click._compat import raw_input
# import jieba.analyse as anl
# 
# s = "人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。人工智能是一门极富挑战性的科学，从事这项工作的人必须懂得计算机知识，心理学和哲学。人工智能是包括十分广泛的科学，它由不同的领域组成，如机器学习，计算机视觉等等，总的说来，人工智能研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。但不同的时代、不同的人对这种“复杂工作”的理解是不同的。[1]  2017年12月，人工智能入选“2017年度中国媒体十大流行语”。[2] 。"
# for x, w in jieba.analyse.textrank(s, topK = 15, withWeight = True):
#     print("%s %s" % (x, w))
# for x, w in jieba.analyse.extract_tags(s, topK = 15, withWeight = True):
#     print("@@@@@%s %s" % (x, w))

# encoding=utf-8
# import numpy as np
# import matplotlib.pyplot as plt #
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [0.199, 0.389, 0.580, 0.783, 0.980, 1.177, 1.380, 1.575, 1.771]
# 
# A = np.vstack([x, np.ones(len(x))]).T
# a, b = np.linalg.lstsq(A, y, None)[0]
# 
# x = np.array(x)
# y = np.array(y)
# 
# plt.plot(x, y, 'o', label = 'Original data', markersize = 10)
# plt.plot(x, a*x+b, 'r', label = 'Fitted line')
# plt.show()

# from sklearn.ensemble import RandomForestClassifier
# X = [
#     [25, 179, 15, 0],
#     [33, 190, 19, 0],
#     [28, 180, 18, 2],
#     [25, 178, 18, 2],
#     [46, 100, 100, 2],
#     [40, 170, 180, 1],
#     [34, 174, 20, 2],
#     [36, 181, 55, 1],
#     [35, 170, 25, 2],
#     [30, 180, 35, 1],
#     [28, 174, 30, 1],
#     [29, 176, 36, 1],
#     ]
# Y = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
# 
# clf = RandomForestClassifier().fit(X, Y)
# p = [[28, 180, 18, 2]]
# print( clf.predict(p))

# encoding=utf
import sys
import numpy.testing
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib

# ascent = scipy.misc.ascent()
# LENA_X = 512
# LENA_Y = 512
# numpy.testing.assert_equal((LENA_X, LENA_Y), ascent.shape)
# xFactor = 8
# yFactor = 4
# 
# resized = ascent.repeat(yFactor, axis = 0).repeat(xFactor, axis = 1)
# numpy.testing.assert_equal((yFactor*LENA_Y, xFactor*LENA_X), resized.shape)
# 
# matplotlib.pyplot.subplot(211)
# matplotlib.pyplot.imshow(ascent)
# 
# matplotlib.pyplot.subplot(212)
# matplotlib.pyplot.imshow(resized)
# matplotlib.pyplot.show()
 ##########################################################
 
 
#创建视图和副本
# ascent = scipy.misc.ascent()
# acopy = ascent.copy()
# aview = ascent.view()
# 
# matplotlib.pyplot.subplot(221)
# matplotlib.pyplot.imshow(ascent)
# 
# matplotlib.pyplot.subplot(222)
# matplotlib.pyplot.imshow(acopy)
#  
# matplotlib.pyplot.subplot(223)
# matplotlib.pyplot.imshow(aview)
# 
# aview.flat = 0
# matplotlib.pyplot.subplot(224)
# matplotlib.pyplot.imshow(aview)
# 
# matplotlib.pyplot.show()


#反转图像
# import scipy.misc
# import matplotlib.pyplot 
# ascent = scipy.misc.ascent()
# 
# matplotlib.pyplot.subplot(221)
# matplotlib.pyplot.imshow(ascent)
# 
# matplotlib.pyplot.subplot(222)
# matplotlib.pyplot.imshow(ascent[:, ::-1])
# 
# matplotlib.pyplot.subplot(223)
# #由于除法/自动产生的类型是浮点型，因此出现上述错误，修正方法为，将/更改为//
# matplotlib.pyplot.imshow(ascent[:ascent.shape[0]//2,:ascent.shape[1]//2])
# 
# mask = ascent%2 == 0
# masked_ascent = ascent.copy()
# masked_ascent[mask] = 0
# matplotlib.pyplot.subplot(224)
# matplotlib.pyplot.imshow(masked_ascent)
# matplotlib.pyplot.show()

# 高级索引tz', operating_system])
# 
# import scipy.misc
# import matplotlib.pyplot 
# 
# ascent = scipy.misc.ascent()
# xMax = ascent.shape[0]
# yMax = ascent.shape[1]
# 
# ascent[range(xMax), range(yMax)] = 0
# ascent[range(xMax-1, -1, -1), range(yMax)] = 0
# 
# matplotlib.pyplot.imshow(ascent)

# 位置列表型索引
# import scipy.misc
# import numpy.random
# import numpy.testing 
# import matplotlib.pyplot
#  
# ascent = scipy.misc.ascent()
# xMax = ascent.shape[0]tz', operating_system])
# 
# yMax = ascent.shape[1]
#  
# def shuffle_indices(size):
#     arr = numpy.arange(size)
#     numpy.random.shuffle(arr)
#     return arr
#  
# xIndices = shuffle_indices(xMax)
# numpy.testing.assert_equal(len(xIndices), xMax)
#  
# yIndices = shuffle_indices(yMax)
# numpy.testing.assert_equal(len(yIndices), yMax)
#  
# matplotlib.pyplot.imshow(ascent[numpy.ix_(xIndices, yIndices)])
# print(numpy.ix_.__doc__)
# matplotlib.pyplot.show()tz', operating_system])
# 

#numpy.arrange扩增
# import numpy as np

# print(np.arange.__doc__)

# a = np.arange(5)    #起始点0，结束点5，步长1，返回类型array，一维
# print("起始点0，结束点5，步长1，返回类型array，一维")
# print(a)
# 
# a = np.arange(6).reshape(2,3)  #2行3列 tuple类型
# print("2行3列 tuple类型")
# print(a)
# 
# a = np.arange(5,20, step = 2)
# print("5开始，歩频为2")
# print(a)

#2.11广播机制扩展数组
# import scipy.io.wavfile
# import matplotlib.pyplot
# import urllib
# import numpy
# 
# #python3.6 合并了urllib1和urllib2
# response = urllib.request.urlopen('http://www.thesoundarchive.com/austinpowers/smashingbaby.wav')
# print(response.info())
# 
# WAV_FILE = 'smashingbaby.wav'
# #wb
# fileHandle = open(WAV_FILE, 'wb')   
# fileHandle.write(response.read() )
# fileHandle.close()
# 
# # print(scipy.io.wavfile.read.__doc__)
# sample_rate, data = scipy.io.wavfile.read(WAV_FILE)
# print("Data type", data.dtype, "Shape", data.shape)
# 
# matplotlib.pyplot.subplot(2, 1, 1)
# matplotlib.pyplot.title("Original")
# matplotlib.pyplot.plot(data)
# 
# newData = data*0.2
# newData = newData.astype(numpy.uint8)
# print("newData type", newData.dtype, "Shape", newData.shape)
# 
# scipy.io.wavfile.write("quiet.wav", sample_rate, newData)
# 
# matplotlib.pyplot.subplot(2, 1, 2)
# matplotlib.pyplot.title("Quiet")
# matplotlib.pyplot.plot(newData)
# matplotlib.pyplot.show()

import numpy
from PIL import Image
import scipy.misc

ascent = scipy.misc.ascent()
print(ascent)
data = numpy.zeros((ascent.shape[0], ascent.shape[1], 4), dtype = numpy.int8)
print(data)
data[:,:,3] = ascent.copy() 
print(data)
img = Image.frombuffer("RGBA", ascent.shape, data, 'raw', "RGBA", 0, 1)
img.save('lena_frombuffer.png')

data[:,:,3] = 255
data[:,:,0] = 222
print(data)
img.save("lena_modifie.png")































































































































