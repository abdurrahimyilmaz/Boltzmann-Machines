# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:29:39 2020

@author: Abdurrahim
"""

#importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing dataset
#sep, dosyalarda veriler nasıl sınıflandırılmış
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#preparing the training and test set
#K katlamalı çarpraz doğrulama için k-kere yöntem çalıştırılır. Her adımda veri kümesinin 1/k kadar, 
#daha önce test için kullanılmamış parçası, test için kullanılırken, geri kalan kısmı eğitim için 
#kullanılır.Literatürde genelde k 10 seçilir.burada onun için 5 tane farklı u datası var
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#getiing the number of users and movies
#5 parçaya ayırdığımız için hangi uda max sayı kaç onu elde ediyoruz
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting the data into an array with users in lines and movies in columns
#kullanıcılar ile filmlere verdikleri ratingler eşleştiriliyor karmaşık halden düzenli hale getiriliyor
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1): #1den başla sonuna kadar devam et
        id_movies =data[:,1][data[:,0] == id_users] #data içerisinde 0. column 1. user eşit olduğunda onu 1. columna eşitle
        id_ratings = data[:,2][data[:,0] == id_users]#data içerisinde 0. column 1. user eşit olduğunda onu 2. columna eşitle
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings 
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set) #normalde x user y filmine şu puanı vermiş şeklinde bir taplomuz var
test_set = convert(test_set) #bu fonksiyonla 943 kişinin 1682 film için rating eşleştirilmesi liste içinde toplanmış oldu

#converting the data into torch tensors - torch tensorları numpy göre daha etkili daha basit
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#0-5 rating into binary rating for boltzmann machine
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#creating the architecture of neural network
class RBM():
    def __init__(self, nv, nh): #nv = number of visible nodes, nh = number of hidden nodes
        self.w = torch.randn(nh, nv) #ağırlıkların tensoru - ağırlıkları normal diste göre rastgele atar (normal dist, mean = 0 and variance = 1)
        self.a = torch.randn(1, nh) #nh için rasrgele bias, baştaki 2 boyutlu olmasını sağlamak için
        self.b = torch.randn(1, nv) #nv için rastgele bias
    #bu fonksiyonun amacı log likelihood gradient yakınsamak bunu da gibbs sampling ile yapıyoruz
    #gibbs sampling uygulamak için de verilen visible nodelar için hidden nodeların olasılıklarını
    #hesaplamamız gerekiyor bu olasıkları hesapladığımız zaman hidden nodeun aktivasyonununun örnekleyebiliriz
    def sample_h(self, x): #hidden node sampling etme probabilitisine göre verilen v için = p_h_given_v aslında sigmoidten başka bir şey değil aslında
        #x değeri de p_h_given_v deki visible neuranlardır
        #önce v tarafından verilen olasılıkları hesaplamamız lazım
        #sigmoid fonksiyonu w*xe uygulanır
        wx = torch.mm(x, self.w.t()) #iki tensorun çarpımı bu şekilde yapılıyor - transpozunu almamız gerekiyor
        activation = wx + self.a.expand_as(wx) #activation function = wx+bias (bias = a) - biz biasın teker teker uygulandığını bilmiyoruz emin olmak için expand_as kullanırız
        #... bu da biasın teker teker batchin her satırına uygular
        p_h_given_v = torch.sigmoid(activation) #verilen visible node değerlerine göre hidden nodeların olasılıkları demek bu satır = sigmodin linear combinasyonu
        return p_h_given_v, torch.bernoulli(p_h_given_v) #bir de bazı gizli nöronları olasılığa göre döndürürüz bu da bernoulli dağılımı ile yaparız binary sonuç istediğmizi için
    def sample_v(self, y):
        wy = torch.mm(y, self.w) #transpoza gerek yok çünkü tersten yapıyoruz yani düz h veriliyor vnin olasılığını elde ediyoruz
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    #v0 = first input vector(rating of all the movies by one user) - vk = k örnekleme sonrası elde edilen visible nodelar visible-hidden-...-visible k kadar loopu sonucu elde edilen visible node
    #ph0 = ilk iterasyonda verilen v0 vektörüne bağlı olarak hidden nodes 1e eşit olduğu andaki olasılıkların vectoru
    #phk = k örnekleme verilen visible node değerlerine vk göre hidden nodeların olasılıkları
    def train(self, v0, vk, ph0, phk): #contrastive divergence ile eğitim fonksiyonu böylece liklehood gradiente yaklaşıcaz
        #rbm yaptığımız için enerji tabanlı model bundan dolayı enerji fonksiyonunu minimize etmeye çalışıyoruz yani training setin log likelihoodu maximize etmeye çalışıyoruz
        #enerji fonksiyonu da modelin ağırlıklarına bağlı
        #gradientleri direk hesaplamak ağır işlem yükü demek bundan dolayı gradientlere yakınsamaya çalışırız bunu da contrastive divergence ile yaparız
        #burası nasıl update yapacağımız gösterir yani eğitim işleme
        self.w += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() #hidden node olasılığı v0 vektörü verildiği durumda 1e eşit olduğu zamandaki elde edilen v0 zero vektörünün değerleri p0 ile çarpılıp 
        #...k iterasyon sonra elde edilen vk zero vektörünün pk ile çarpılan değerlerinden çıkartılır -- v0 ve vk filmlerin ratingleridir
        self.b += torch.sum((v0 - vk), 0) #0 eklenmesinin sebebi tensoru iki boyutta tutmak için
        self.a += torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0]) #kaç tane visible giriş nodeumuz olacak
nh = 100 #gizli node sayısı tespit ettiğimiz featurea bağlı net değer söylemek zor yaklaşık bi değer veririz
batch_size = 100 #aynı anda kaç veriyi işleyecez bunu belirliyoruz bu da nh gibi tunable bize bağlı

rbm = RBM(nv,nh)

#training the rbm
nb_epoch = 10 #küçük seçtik çünkü binary ve az veri var
for epoch in range(1, nb_epoch + 1): #epochu elle manuel döndürüyoruz
    train_loss = 0 #eğitim kaybını başta sıfır olarak ayarlıyoruz
    s = 0. #float olarak train lossu minimize etmek için counter ekliyoruz trainlossu sürekli buna bölecez
    for id_user in range(0, nb_users - batch_size, batch_size): #training burada başlıyor
        #user user yapmıyacaz batch batch yapacaz onun için aralıklar batch_sizea bağlı
        vk = training_set[id_user:id_user+batch_size] #gibbs örneklemesinin outputu bu olacak 
        #aralıklarda bu epoch için aralıklar olacağı için böyle belirlendi
        v0 = training_set[id_user:id_user+batch_size] #başta vk ile v0 aynı zaten
        ph0,_ = rbm.sample_h(v0) #baştaki probları böyle alıyoruz _ ile 2. returna gerek olmadığı için pasif yaptık
        #ilk visible nodedan başladığı için hidden probu hesaplayacağımızdan bu fonk kullanırız
        for k in range(10): #10 kere loop yapacaz - böylece gradient yakınsayacak sürekli loop yapa yapa
            _,hk = rbm.sample_h(vk)  #hidden nodelarda k stepteki durumu sürekli hesaplıyoruz
            _,vk = rbm.sample_v(hk)  #visible nodelarda k stepteki durumu sürekli hesaplıyoruz
            vk[v0<0] = v0[v0<0] #-1 yani değerlendirilmeyenleri donduruyoruz
        phk,_ = rbm.sample_h(vk) #vkyi elde ettikten sonra phkyi de elde edebiliyoruz
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
    
#testing the rbm
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: ' + str(test_loss/s))

#evaluating the boltzmann machine
"""
Hi guys,

the two ways of evaluating our RBM are with the RMSE and the Average Distance.

RMSE:

The RMSE (Root Mean Squared Error) is calculated as the root of the mean of the squared differences between the predictions and the targets.

Here is the code that computes the RMSE:

Training phase:

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
Test phase:

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))
Using the RMSE, our RBM would obtain an error around 0.46. But be careful, although it looks similar, one must not confuse the RMSE and the Average Distance. A RMSE of 0.46 doesn’t mean that the average distance between the prediction and the ground truth is 0.46. In random mode we would end up with a RMSE around 0.72. An error of 0.46 corresponds to 75% of successful prediction.

Average Distance:

If you prefer to play with the Average Distance, I understand, it’s more intuitive. And that’s what we used in the practical tutorials to evaluate our RBM model:

Training phase:

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # Average Distance here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
Test phase:

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Average Distance here
        s += 1.
print('test loss: '+str(test_loss/s))
With this metric, we obtained an Average Distance of 0.24, which is equivalent to about 75% of correct prediction.

Hence, it works very well and there is a predictive power.

If you want to check that 0.25 corresponds to 75% of success, you can run the following test:

import numpy as np
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v)) # -> you get 0.25
so 0.25 corresponds to 75% of success.

Enjoy Deep Learning!
"""