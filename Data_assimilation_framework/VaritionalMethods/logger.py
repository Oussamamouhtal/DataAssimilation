from fourdvar import fourDvar
import matplotlib.pyplot as plt


def plot(qudcost, max_outer, list_max_inner, OuterLoop = False, LineStyle = ':d',label = None):

    """

    Plot the quadratic cost function versus CG iteration.
    Parameters:
      qudcost  : A list or array containing the values of the quadratic cost function.
      max_outer: An integer representing the maximum number of outer loop iterations.
      list_max_inner: A list representing the maximum number of inner loop iterations.
      OuterLoop: A boolean indicating whether to use outer loop or not. Default is False.
      LineStyle: A string defining the line style for the plot. Default is ':d'.
      label    : A string containing the label for the plot. Default is None.

    """
    index = 0  
    plt.xlabel('total CG itérations')
    plt.ylabel('Quadratic cost function')
    plt.yscale('log')
    plt.axvline(x = index, color = 'black') 
    if OuterLoop == True :
      
        for i in range(max_outer-1):
            index = index + list_max_inner[i]+1
            plt.axvline(x = index, color = 'black')
        plt.plot(qudcost, LineStyle, label=label)
    else :
        ## Plot the second inner loops 
        plt.plot(qudcost[list_max_inner[0] + 1: list_max_inner[0]+ list_max_inner[1] + 2], LineStyle, label=label)



# ===================== The Problem Parameters =====================
""" Choose   n > Nt * m_t"""
n = 100
m_t = 30 
Nt = 2  

# ===================== The Solvers Parameters =====================
""" Choose   max_inner =< Nt * m_t"""
max_outer = 2   # max iteration for Gauss Newton loop
list_max_inner = [30, 40]  # max iteration for cg loops. (be sur that len(list_max_inner) = max_outer)



##################


OutUnprec= fourDvar(n,m_t, Nt, max_outer, list_max_inner, 
                             'Unprecon_CG' , IP = False)   
 

OutLmpOne = fourDvar(n,m_t, Nt, max_outer, list_max_inner, 
                            'Spectral_LMP' , IP = False) 

OutLmpOneIP = fourDvar(n,m_t, Nt, max_outer, list_max_inner, 
                            'Spectral_LMP' , IP = True) 

OutLmpLam= fourDvar(n,m_t, Nt, max_outer, list_max_inner,
                            'Spectral_LMP', selectedTHETA = 'lambda_k', 
                            IP = False)
 
OutLmpMed = fourDvar(n,m_t, Nt, max_outer, list_max_inner, 
                            'Spectral_LMP' ,selectedTHETA = 'mediane', IP = False) 


OutLmpOpt = fourDvar(n,m_t, Nt, max_outer, list_max_inner,
                            'Spectral_LMP' ,selectedTHETA = 'ThetaOpt',
                             IP = False) 

OutDeflation = fourDvar(n,m_t, Nt, max_outer, list_max_inner, 
                            'Deflated_CG') 

# ==================== Displays ====================


plot(OutUnprec, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':d',label = 'No sLMP')

plot(OutLmpOne, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':o',label = r'$ \theta = 1$')

plot(OutLmpOneIP, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':o',label = r'$ \theta = 1$' + '+ IP')

plot(OutLmpLam, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':o',label = r'$ \theta = \lambda_k$')

plot(OutLmpMed, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':*',label = r'$ \theta = \frac{\lambda_k +1}{2}$')

plot(OutLmpOpt, max_outer, list_max_inner, OuterLoop = True, 
     LineStyle = ':*',label = r'$ \theta = \theta_{mass}$')

plot(OutDeflation, max_outer, list_max_inner, OuterLoop = True,
      LineStyle = ':*',label = 'Deflated CG')


plt.legend()
plt.show()
       