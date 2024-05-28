from fourdvar import fourDvar
import matplotlib.pyplot as plt


def plot(qudcost, max_outer, max_inner, OuterLoop = False, LineStyle = ':d',label = None):

    """

    Plot the quadratic cost function versus CG iteration.
    Parameters:
      qudcost  : A list or array containing the values of the quadratic cost function.
      max_outer: An integer representing the maximum number of outer loop iterations.
      max_inner: An integer representing the maximum number of inner loop iterations.
      OuterLoop: A boolean indicating whether to use outer loop or not. Default is False.
      LineStyle: A string defining the line style for the plot. Default is ':d'.
      label    : A string containing the label for the plot. Default is None.

    """

    plt.xlabel('total CG itérations')
    plt.ylabel('Quadratic cost function')
    plt.yscale('log')
    plt.axvline(x = 0, color = 'black') 
    if OuterLoop == True :
        for i in range(max_outer-1):
            plt.axvline(x = (i+1)*(max_inner+1), color = 'black')
        plt.plot(qudcost, LineStyle, label=label)
    else :
        plt.plot(qudcost[max_inner + 1: 2 * (max_inner+1)], LineStyle, label=label)



# ===================== The Problem Parameters =====================
""" Choose n > Nt * m_t"""
n = 300
m_t = 40 
Nt = 2  

# ===================== The Solvers Parameters =====================
""" Choose max_inner =< Nt * m_t"""
max_inner = 60  # max iteration for cg loops
max_outer = 2   # max iteration for Gauss Newton loop




##################


OutUnprec= fourDvar(n,m_t, Nt, max_outer, max_inner, 
                             'Unprecon_CG' , IP = False)   
 

OutLmpOne = fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'Spectral_LMP' , IP = False) 

OutLmpOneIP = fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'Spectral_LMP' , IP = True) 

OutLmpLam= fourDvar(n,m_t, Nt, max_outer, max_inner,
                            'Spectral_LMP', selectedTHETA = 'lambda_k', 
                            IP = False)
 
OutLmpMed = fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'Spectral_LMP' ,selectedTHETA = 'mediane', IP = False) 


OutLmpOpt = fourDvar(n,m_t, Nt, max_outer, max_inner,
                            'Spectral_LMP' ,selectedTHETA = 'ThetaOpt',
                             IP = False) 

OutDeflation = fourDvar(n,m_t, Nt, max_outer, max_inner, 
                            'Deflated_CG') 

# ==================== Displays ====================


plot(OutUnprec, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':d',label = 'No sLMP')

plot(OutLmpOne, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':o',label = r'$ \theta = 1$')

plot(OutLmpOneIP, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':o',label = r'$ \theta = 1$' + '+ IP')

plot(OutLmpLam, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':o',label = r'$ \theta = \lambda_k$')

plot(OutLmpMed, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':*',label = r'$ \theta = \frac{\lambda_k +1}{2}$')

plot(OutLmpOpt, max_outer, max_inner, OuterLoop = False, 
     LineStyle = ':*',label = r'$ \theta = \theta_{mass}$')

plot(OutDeflation, max_outer, max_inner, OuterLoop = False,
      LineStyle = ':*',label = 'Deflated CG')


plt.legend()
plt.show()
       