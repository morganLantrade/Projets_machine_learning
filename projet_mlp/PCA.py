import numpy as np
import matplotlib
import matplotlib.pyplot as plt
class PCA:
    '''Classe qui permet de réduire le nombre de feature par Analyse Par Composances
    X dont les lignes sont les observation et les colonnes sont les features'''

    def __init__(self,X):
        #1 standartisation
        self.X_std=(X-np.mean(X,axis=0))/np.std(X,axis=0)
        
        #2 correlation
        self.corr=np.corrcoef(X.T)
        self.accuracy=20 #number of digits to round
        while not np.array_equal( self.corr,(self.corr).T) and self.accuracy>2:
            self.corr=np.round(self.corr,self.accuracy)
            self.accuracy-=1
        assert self.accuracy>2 , "Matrice de correlation non trouvée"
        
        #3 valeurs propres vecteurs propres
        v,w = np.linalg.eig(self.corr)
        self.eigen_values=v
        self.eigen_matrix=w
        
    def get_eigen_matrix(self):
        return self.eigen_matrix

    def get_eigen_values(self):
        return self.eigen_values

    def get_corr(self):
        return self.corr

    def get_X_std(self):
        return self.X_std
        
    def show(self,n_first_components=None):
        '''Affiche la courbe d'inertie et eboulie des n_first_components'''
        if not n_first_components:
            n_first_components=self.eigen_values.size
        else:
            assert n_first_components <= self.eigen_values.size , f'Le nombre de composantes est plus grande que le nombre de features'
       
        #calculs des inerties et eboulies
        total=np.sum(self.eigen_values)
        the_range_inertie=list(range(0,n_first_components+1))
        the_range_eboulie=list(range(1,n_first_components+1))
        inertie=[ sum(self.eigen_values[i] for i in range(k))/total for k in the_range_inertie]
        eboulie=[ self.eigen_values[v-1] for v in the_range_eboulie]
        

        #affichage
        fig, axs = plt.subplots(1,2)
        fig.set_figheight(4)
        fig.set_figwidth(12)
        axs[0].plot(the_range_inertie,inertie)
        axs[0].set_xlabel('n premieres composantes')
        axs[0].set_ylabel('inertie')
        axs[0].set_title(f'Inertie des {n_first_components} premiere composantes')
        axs[1].bar(the_range_eboulie,eboulie,1)
        axs[1].set_xlabel('n premieres composantes')
        axs[1].set_ylabel('eboulie')
        axs[1].set_title(f'Eboulie   des {n_first_components} premiere composantes')


  


    def transform(self,n_first_components):
        '''Retoure X transforme et réduit à n_first_components'''
        assert n_first_components <= self.eigen_values.size , f'Le nombre de composantes est plus grande que le nombre de features'
        projection=(self.eigen_matrix[:,:n_first_components]).T
        return np.dot(projection,(self.X_std).T).T


