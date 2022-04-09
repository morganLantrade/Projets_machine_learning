import sklearn
from sklearn import datasets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from sklearn.datasets import make_blobs, make_circles

############################################
#           ACTIVATES FUNCTIONS            #
############################################

def identity(Z):
    return Z

def d_identity(A):
    return np.ones((A.shape))

def sigmoide(Z):
    return 1/(1+np.exp(-Z))

def d_sigmoide(A):
    return A*(1-A)

def tanh(Z): 
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

def d_tanh(A):
    return 1-A*A

def relu(Z):
    return np.where(Z>0,Z,0)

def d_relu(A):
    return np.where(A>0,1,0)


ACTIVATIONS={'identity' : (identity,d_identity) ,'logistic' : (sigmoide,d_sigmoide),
             'tanh': (tanh,d_tanh),'relu': (relu,d_relu)}

############################################
#           COST FUNCTIONS                 #
############################################

def MSE_cost(A,y):
    return 1/y.shape[1]*np.sum((y-A)**2)

def log_loss(A,y,eps=10**(-15)):  #eps car log n'est pas définie en 0
    return - 1/y.shape[1]* np.sum(y *np.log(A+eps) + (1 - y) * np.log(1-A+eps))

def d_MSE_cost(A,y): #la somme 1 à m sera faite en produit matriciel (m,n)
    return 2/y.shape[1]*(A-y)

def d_log_loss(A,y,eps=10**(-15)): #la somme de 1 à m sera faite en produit matriciel avec (m,n)
    return 1/y.shape[1]*(A+eps-y)/((A+eps)*(1-A+eps))

COSTS={ 'log_loss':(log_loss,d_log_loss),'MSE': (MSE_cost,d_MSE_cost)}

############################################
#           GRAPHICS METHODS               #
############################################

def show_learning(nb_iter,couts_A,couts_t,scores_A,scores_t):
    '''Affiche la courbe de couts, d'apprentissage'''
    plt.ion()
    fig, axs = plt.subplots(1,2) 
    fig.set_figheight(5)
    fig.set_figwidth(18)
    #axe 2
    axs[0].plot(nb_iter,couts_A,label='loss apprentissage')
    axs[0].plot(nb_iter,couts_t,label='loss test')
    axs[0].set_title('Loss curve')
    axs[0].set_xlabel('itérations')
    axs[0].set_ylabel('cost')
    axs[0].legend()
    #axe 1
    axs[1].set_ylim(0,1)
    axs[1].plot(nb_iter,scores_A,label='score apprentissage')
    axs[1].plot(nb_iter,scores_t,label='score test')
    axs[1].set_title(f'Learning curve')
    axs[1].set_xlabel('itérations')
    axs[1].set_ylabel('accuracy')
    axs[1].legend()
    plt.show(block=True)



def show_learning2D(nb_iter,couts_A,couts_t,scores_A,scores_t,Xa,ya,Xt,yt,Z,xx, yy):
    '''Affiche la courbe de couts, d'apprentissage et la frontiere'''
    plt.ion()
    fig, axs = plt.subplots(1,3) 
    fig.set_figheight(5)
    fig.set_figwidth(18)
    #axe 2
    axs[0].plot(nb_iter,couts_A,label='loss apprentissage')
    axs[0].plot(nb_iter,couts_t,label='loss test')
    axs[0].set_title('Loss curve')
    axs[0].set_xlabel('itérations')
    axs[0].set_ylabel('cost')
    axs[0].legend()
    #axe 1
    axs[1].set_ylim(0,1)
    axs[1].plot(nb_iter,scores_A,label='score apprentissage')
    axs[1].plot(nb_iter,scores_t,label='score test')
    axs[1].set_title(f'Learning curve')
    axs[1].set_xlabel('itérations')
    axs[1].set_ylabel('accuracy')
    axs[1].legend()

    #axe 3
    axs[2].set_title(f'Frontiere')
    axs[2].contourf(xx, yy, Z, [0,0.5,1],colors=['rosybrown','papayawhip'])
    axs[2].scatter(Xa[:,0],Xa[:,1],c=ya[:,0],s=50)
    axs[2].scatter(Xt[:,0],Xt[:,1],c=yt[:,0],s=50,alpha=0.2)
    
    plt.show(block=True)




############################################
#           MLP CLASS                      #
############################################
class MLP:
    def __init__(self,hidden_layers=[],activation='logistic',cost='MSE',random_state=False,learning_rate=0.2,max_iter=50000,momentum=0.9):    
        self.nb_layers=len(hidden_layers)+1
        self.layers=[] 
        self.weights,self.biais=[],[]
        self.activations=[]
        self.gradients_W=[]
        self.gradients_B=[]
        self.activation=activation
        self.cost=cost
        self.random_state=random_state
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.hidden_layers=hidden_layers
        self.momentum_W=[]
        self.momentum_b=[]
        self.momentum=momentum

    def parameters(self):
        return { 'hidden_layers' : self.hidden_layers,'activation' : self.activation,'cost':self.cost,
                  'random_state': self.random_state,'learning_rate':self.learning_rate,'max_iter':self.max_iter,'momentum':self.momentum}
                                            


    def initWeights(self,X,y):
        '''Retourne une liste des poids de notre modeles
        une liste des biais de notre modeles indice 0 correspond a la couche 1'''
        weights=[]
        biais=[]
        layers=[X.shape[1]]+self.hidden_layers+[y.shape[1]]
        #update layers
        self.layers=[ (layers[i],layers[i+1]) for i in range(self.nb_layers)]
        if not(self.random_state):
            np.random.seed(1)
        for (n0,n1) in self.layers:
            weights.append(np.random.randn(n1,n0))
            biais.append(np.random.randn(n1,1))
        return weights,biais


    def forward(self,X):
        '''Met a jours les activations
        A[0]=X.T '''
        activations=[X.T]
        for i in range(1,self.nb_layers+1):
            W=self.weights[i-1]
            b=self.biais[i-1]
            Z=W.dot(activations[i-1])+b
            # on finit sur une sigmoide si classification binaire else ( TO DOO)
            A=ACTIVATIONS[self.activation][0](Z) if i!=self.nb_layers else sigmoide(Z)  
            activations.append(A)
        return activations



    def backward(self,X,y):
        '''Retourne une liste des gradients de W et de b ou 
        l'indice correspondant à la couche i-1
        '''
        m=X.shape[0]
        #nous allons devoir stocker les dL/dZ pour notre rétroPropagation 
        les_dZ= [COSTS[self.cost][1](self.activations[self.nb_layers],y.T) * ACTIVATIONS[self.activation][1](self.activations[self.nb_layers])] 
        #la premiere propagation arriere est sigmoide
        dWs= [les_dZ[0].dot(self.activations[self.nb_layers-1].T)]  
        dBs=[(1/m if self.cost=='log_loss' else 2/m) * np.sum(les_dZ[0],axis=1,keepdims=True)]

        for L in range(1,self.nb_layers):
            W=self.weights[self.nb_layers-L]
            b=self.biais[self.nb_layers-L]
            A=self.activations[self.nb_layers-L]
            dZ= 1/m * (W.T).dot(les_dZ[L-1])*ACTIVATIONS[self.activation][1](A) #dérivée de la fonction activation
            les_dZ.append(dZ)
            dWs.insert(0,dZ.dot(self.activations[self.nb_layers-L-1].T))
            dBs.insert(0,np.sum(dZ,axis=1,keepdims=True))
        return dWs,dBs

    def update(self):
        '''update weights, biais selon le learning_rate et  momentum '''
        for L in range(self.nb_layers):
            self.momentum_W[L]=self.momentum*self.momentum_W[L]+self.gradients_W[L]
            self.weights[L]=self.weights[L]-self.learning_rate*self.momentum_W[L]
            self.momentum_b[L]=self.momentum*self.momentum_b[L]+self.gradients_B[L]
            self.biais[L]=self.biais[L]-self.learning_rate*self.momentum_b[L]
            

    def predict(self,X):
        '''Seuil 0.5 pour la sigmoide'''
        A=self.forward(X)[self.nb_layers] # derniere couche
        return np.where(A>0.5,1,0) 

    def predict_proba(self,X):
        A=self.forward(X)[self.nb_layers] # derniere couche
        return A

    def fit(self,X,y):
        '''Fonction qui effectue la propagation avant et arriere en mettant a jours les poids et biais'''
        self.weights,self.biais=np.array([[ 1.41683865 ,-0.71898376]]),np.array([[-0.52820104]])#self.initWeights(X,y)
        i=0
        self.activations=self.forward(X)
        self.momentum_W,self.momentum_b=  self.backward(X,y) 
        while i<self.max_iter:
            self.activations=self.forward(X)
            self.gradients_W,self.gradients_B=self.backward(X,y)
            self.update()
            i+=1
            
            

    def fit_show(self,Xa,ya,Xt,yt,bins=10):
        '''Fonction d'apprentissage qui affiche l'apprentissage une fois finie'''
        self.weights,self.biais=self.initWeights(Xa,ya)
        nb_iter,couts_A,couts_t,scores_A,scores_t=[],[],[],[],[]
        i=0
        self.activations=self.forward(Xa)
        self.momentum_W,self.momentum_b=  self.backward(Xa,ya)     
        while i<self.max_iter:
            self.activations=self.forward(Xa)
            self.gradients_W,self.gradients_B=self.backward(Xa,ya)
            self.update()
            i+=1
            if i%bins==0:
                couts_A.append(COSTS[self.cost][0](self.activations[self.nb_layers],ya.T))
                couts_t.append(COSTS[self.cost][0](self.forward(Xt)[self.nb_layers],yt.T))
                scores_A.append(self.score(Xa,ya))
                scores_t.append(self.score(Xt,yt))
                nb_iter.append((i+1))
        if Xa.shape[1]==2: # 2D
            nx, ny = 200, 200
            X=np.concatenate((Xa,Xt),axis=0)
            x_min, x_max = np.min(X[:,0]),np.max(X[:,0])
            y_min, y_max = np.min(X[:,1]),np.max(X[:,1])
            xx, yy = np.meshgrid(np.linspace(x_min-0.1,x_max+0.1, nx),np.linspace(y_min-0.1,y_max+0.1, ny))
            Z = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            Z=Z[0, :].reshape(xx.shape)
            show_learning2D(nb_iter,couts_A,couts_t,scores_A,scores_t,Xa,ya,Xt,yt,Z,xx, yy)
        else:
            show_learning(nb_iter,couts_A,couts_t,scores_A,scores_t)


    def score(self,X,y):
        return accuracy_score((y.T)[0],self.predict(X)[0])
        



if __name__ == '__main__':
    Xa, ya = make_circles(n_samples=400, noise=0.10, factor=0.3, random_state=0)
    Xa=np.concatenate((Xa,Xa+2))
    Xt, yt = make_circles(n_samples=100, noise=0.10, factor=0.3, random_state=0)
    Xt=np.concatenate((Xt,Xt+2))
    ya=np.concatenate((ya,np.where(ya==1,0,1)))
    yt=np.concatenate((yt,np.where(yt==1,0,1)))
    ya = ya.reshape(ya.shape[0],1)
    yt = yt.reshape(yt.shape[0],1)
    
    mlp=MLP(hidden_layers=[32],learning_rate=0.2,max_iter=10000)
    print(mlp.parameters())
    

    

    

        