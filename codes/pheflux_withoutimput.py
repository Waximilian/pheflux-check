#!/usr/bin/env python
# coding: utf-8

# In[1]:


from casadi import *
import cobra
from cobra.flux_analysis import flux_variability_analysis
#from scipy.stats import entropy
import numpy as np
import pandas as pd
import csv
import json
import time
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


# In[2]:


##############################################################################
### Obtains expression value 'g' for one reaction, based on the GPR
def getG(rule,fpkmDic):
    orList=[]
    # Devide the rule by "or" and iterate over each resulting subrule
    for subrule in rule.split(" or "):
        vector = subrule.split(' and ')
        g_vector = []
        for gene in vector:
            gene = str(gene).replace("(","") # Removes "("
            gene = str(gene).replace(")","") # Removes "("
            gene = str(gene).replace(" ","") # Removes "("
            g_vector.append( 'G_'+gene ) 
        value = str(g_vector).replace("'","") # Removes "'"
        value = eval("min("+value+")", fpkmDic)
        orList.append(value) # Add the minimum to a list
    return( np.sum(orList) ) # Print the sum of the list
##############################################################################
### This function gives a useful vector to discriminate if the FPKM of the
### genes associated with a reaction is known.
def booleanVectorRule (rule, fpkmDic):
    boolean_list = []
    vector_rule = rule.replace("or","")
    vector_rule = vector_rule.replace("and","")
    vector_rule = vector_rule.replace("'","") # Removes "'"
    vector_rule = vector_rule.replace("(","") # Remove "("
    vector_rule = vector_rule.replace(")","")
    vector = vector_rule.split()
    g_vector = []
    for gene in vector:
        g_vector.append( ('G_'+gene) )
    for gene in g_vector:
        if gene in fpkmDic:
            boolean_list.append('True')
        else:
            boolean_list.append('False')
    return (boolean_list)
# ##############################################################################
# ### function to correct reverse reactions in constraints
# def correctedRevExn(reaction,model):
#     reaction_id = '_'.join(reaction.split('_')[0:-2])
#     reaction = model.reactions.get_by_id('R_'+reaction_id).reverse_id
#     return (reaction)


# In[3]:


##############################################################################
### Loading FPKM data
def loadFPKM(fpkm,condition,shuffle=False,shuffledFPKM=pd.DataFrame()):
    ##########################################################
    ## Gets gene IDs and their expression values
    genes=fpkm["Gene_ID"]
    if shuffle:
        fpkms=fpkm["Expression"].sample(frac=1).reset_index(drop=True) 
    else:
        fpkms=fpkm["Expression"]
    shuffledFPKM["Expression"] = fpkms
    ##########################################################
    ## Creates a dictionary gene_ID -> Expression value
    fpkmDic = {}
    for i in range(len(fpkms)): # Run over each line in fpkm file
        # 1. Get gene id and fpkm values for each line
        name = 'G_'+str(genes[i])
        fpkm = fpkms[i]
        if type(fpkm) == np.float64:
            fpkmDic[name] = fpkm
    ##########################################################
    ## Capping at 95%
    cap = np.percentile( list(fpkmDic.values()), 95)
    for i in fpkmDic:
        if fpkmDic[i]>cap:
            fpkmDic[i] = cap
    return(fpkmDic,shuffledFPKM)
##############################################################################
### Reloading FPKM data. Only for Homo sapiens.
def reloadFPKMHsapiens(fpkmDic, model):
    newfpkmDic = {}
    for gene in model.genes:
        if not 'G_'+gene.name in fpkmDic: continue
        fpkm = fpkmDic['G_'+gene.name]
        gene = 'G_'+gene.id
        newfpkmDic[gene] = fpkm
    return(newfpkmDic)
##############################################################################
### Loading C13 data
def loadC13(fluxDataFile):
    data_c13 = pd.read_csv(fluxDataFile,sep="\t", lineterminator='\n') 
    return(data_c13)


# In[4]:


##############################################################################
# UPDATE MODEL
def updateModel(model_default,uptakeRxns,process,defaultUptake,medium):
    model=model_default.copy()
    ##########################################################
    ## Add 'R_' to reactions names
    for reaction in model.reactions:
        reaction.id = 'R_'+reaction.id
    ##########################################################        
    ## Opening the model: exchange reactions
    for rxn in model.reactions: 
        if (rxn.lower_bound<0 and rxn.upper_bound>0):
            rxn.bounds = (-1000,1000)
        if (rxn.lower_bound>=0 and rxn.upper_bound>0):
            rxn.bounds = (0,1000)
        if (rxn.lower_bound<0 and rxn.upper_bound<=0):
            rxn.bounds = (-1000,0)
    ##########################################################
    ## Set carbon source or culture medium
    #############################################
    # Validation
    if process == 'validation':
        # Only one carbon source open: DC
        defaultUptake = 'R_'+defaultUptake
        model.reactions.get_by_id(defaultUptake).lower_bound=0
        for uptakeRxn in uptakeRxns.split(","):
            model.reactions.get_by_id("R_"+uptakeRxn).lower_bound = -1000
        # 8 carbon sources open: AC
        #for uptakeRxn in uptakeRxns:
        #    model.reactions.get_by_id("R_"+uptakeRxn).lower_bound = -1000
        # All carbon sources open: Full AC
        #for reaction in model.exchanges:
        #    reaction.bounds = (-1000,1000)
    #############################################
    # TCGA and GTEx
    if process == 'tcga' or process == 'gtex':
        #########################################
        # set bounds of exchange reactions to (0, 1000)
        for reaction in model.exchanges:
            reaction.bounds = (0, 1000)
        #########################################
        # add culture medium
        for reaction in medium['Reaction_ID']:
            if 'R_'+reaction in model.reactions:
                model.reactions.get_by_id('R_'+reaction).lower_bound = -1000
            
    return(model)


# In[5]:


##############################################################################
## Obtains a median value of the expression of genes associated with metabolism
def getEg(model,fpkmDic):
    g_metab = [] # gene expression of reactions partakin in the metabolism
    for i, reaction in enumerate(model.reactions):
        ##########################################################
        ## Gets the GPR and a boolean list with known or unknown genes
        rule = reaction.gene_reaction_rule
        boolean_list = booleanVectorRule(rule,fpkmDic)
        ##########################################################
        ## Gets the expression value 'g'
        if not ('False' in boolean_list or rule == ''): # get 'g' for reaction with GPR.
            g = getG(rule, fpkmDic)
            g_metab.append(g+1e-8)
    ##############################################################
    ## Obtains a median value
    E_g = np.median(g_metab)
    return(E_g,g_metab)


# In[6]:


##############################################################################
### SAVE PRIMAL VALUES
def getPrimalValues(model):
    ##########################################################
    ### Model optimize: save fluxes and primal values of variables
    sol = model.optimize()
    fba_primal = {}
    for reaction in model.reactions:
        f_name = reaction.id
        r_name = reaction.reverse_id
        fba_primal[f_name] = eval ('model.variables.'+f_name+'.primal')
        fba_primal[r_name] = eval ('model.variables.'+r_name+'.primal')
    return(fba_primal)
##############################################################################
### Save forward and reverse variables
def getFowardReverse(model):
    v_vars, rev_vars = [], []
    for reaction in model.reactions:
        v_vars.append(reaction.id)
        rev_vars.append(reaction.reverse_id)  
    return(v_vars,rev_vars)


# In[7]:


#################################################################################
### Variables and objective function: CasADI object
def setVariables(model,fpkmDic):
    v = vertcat() # saves the total of variables of the model. Used to """nlp['x']"""
    v_dic = {}
    v_fpkm = {} # 
    ubx, lbx = [],[]
    ##############################################################
    ## Gets the median value of 'g'
    E_g,g_metab = getEg(model,fpkmDic)     
    
    for i, reaction in enumerate(model.reactions):
        ##########################################################
        ## Gets the GPR and a boolean list with known or unknown genes
        rule = reaction.gene_reaction_rule # gene reaction rule
        boolean_list = booleanVectorRule(rule,fpkmDic) # useful to discriminate between genes with known fkpm.
        ##########################################################
        ## Gets the expression value 'g'
        # get 'g' for reaction with GPR.
        if not ('False' in boolean_list or rule == ''): 
            g = getG(rule, fpkmDic)+1e-8#*1e-8
            if getG(rule, fpkmDic)==0:
                print("No expression: ",reaction.id)
        # set 'g' (median value) for reaction without GPR. 
        else:
            g = E_g
        ##########################################################
        ## Set forward and reverse variables as a CasADI object
        # forward
        var_name = reaction.id
#         vf = SX.sym(var_name) # v forward
        expression = var_name+' = SX.sym("'+var_name+'")'
        exec(expression, globals())
        vf = eval(var_name)
        v = vertcat(v, vf)
        v_dic[reaction.id]=vf
        ubx.append(reaction.upper_bound)    
        lbx.append(0.0)

        # reverse
        var_name_reverse = reaction.reverse_id
#         vr = SX.sym(var_name_reverse) # v reverse
        expression = var_name_reverse+' = SX.sym("'+var_name_reverse+'")'
        exec(expression, globals())
        vr = eval(var_name_reverse)
        v = vertcat(v,vr)
        v_dic[reaction.reverse_id]=vr
        ubx.append(-reaction.lower_bound)
        lbx.append(0.0)

        v_fpkm[var_name] = g
        v_fpkm[var_name_reverse] = g
        ##########################################################
        ## Define a objective function
        for name in [vf,vr]:
            if i == 0:
                v_ViLogVi = ( (name)+1e-8 )*log( (name)+1e-8 ) # 1.1
                v_VilogQi = ( (name)+1e-8 )*log( g ) # 2.1
            else:
                v_ViLogVi += ( (name)+1e-8 )*log( (name)+1e-8 ) # 1.1
                v_VilogQi += ( (name)+1e-8 )*log( g ) # 2.1            
    ##############################################################
    ## Set objetive function
    f = (v_ViLogVi) - (v_VilogQi)
    return(v,v_dic,lbx,ubx,f)


# In[8]:


#################################################################################
### Define a sumV
def getSumV(v):
    for i in range(v.shape[0]): # VARIABLES
        name = v[i]
        if i == 0:
            sumVi = name
        else:
            sumVi += name     
    return(sumVi)


# In[9]:


#################################################################################
### Creating constraints
def createConstraints(model,k,v_dic,sumVi):
    g = vertcat()
    lbg,ubg=[],[]
    ##############################################################
    ## Gets the name of the forward/reverse variables
    v_vars, rev_vars = getFowardReverse(model)
    ##############################################################
    ## Defines constraints
    for met in model.metabolites:
        ##########################################################
        ## Gets constraint for a one metabolite
#         tmp_constraint = []
#         constraint = str(met.constraint.expression)
        constraint = str(met.constraint.expression).replace('+ ','+').replace('- ','-')
#         print('constraint:', constraint)
        ##########################################################
        ## Reconstruct the constraint as a CasADI object
        for i, field in enumerate(constraint.split()):
#             print('field:', field)
            if i == 0:
                tmp_constraint = eval(field)
            else:
                tmp_constraint += eval(field)
                
#             reaction = field.strip().split("*")[-1]
#             print('field, reaction:', field, '->', reaction)
#             if field == '-' or field == '+':
#                 connector = field        
#             else: 
#                 if reaction in v_vars:
#                     field = re.sub(r'\b'+reaction+r'\b', str(reaction), field)

#                 elif reaction in rev_vars:
#                     field = re.sub(r'\b'+reaction+r'\b', str(reaction), field)

#             if len(tmp_constraint) == 0:
#                 tmp_constraint.append(field)
#             else:
#                 if not (field=='+' or field=='-'):
#                     tmp_constraint.append(connector)
#                     tmp_constraint.append(field)
        ##########################################################
        ## Adds constraint to CasADI
#         tmp_constraint = ' '.join(tmp_constraint)
#         g = vertcat(g,eval(tmp_constraint,v_dic))
#         g = vertcat(g,eval(tmp_constraint))
        g = vertcat(g,tmp_constraint)
        lbg.append(0)
        ubg.append(0)
    ##############################################################
    ## SumV constraint
    g = vertcat(g,sumVi)
    lbg.append( k )
    ubg.append( k )
    
    return(ubg,lbg,g)


# In[10]:


#################################################################################
### OPTIMIZATION
def PheFlux(model,fpkmDic,k,init_time):
    ##############################################################
    ## Sets variables, sumV and constraints
    v,v_dic,lbx,ubx,f = setVariables(model,fpkmDic)
    sumVi = getSumV(v)
    ubg,lbg,g = createConstraints(model,k,v_dic,sumVi)    
    print('')
    ##############################################################
    # Non-linear programming
    nlp = {}     # NLP declaration
    nlp['x']=  v # variables
    nlp['f'] = f # objective function
    nlp['g'] = g # constraints
    ##############################################################
    # Create solver instance
    F = nlpsol('F','ipopt',nlp)
    ##############################################################
    # Solve the problem using a guess
    fba_primal = getPrimalValues(model)
    x0=[]
    for i in range(v.shape[0]): # VARIABLES
        x0.append(fba_primal[str(v[i])])       
    ##############################################################
    ## Solver
    start = time.time()
    sol=F(x0=x0,ubg=ubg,lbg=lbg,lbx=lbx,ubx=ubx)
    final = time.time()
    total_time = final - init_time
    optimization_time = final - start
    status = F.stats()['return_status']
    success = F.stats()['success']
    ##############################################################
    ## Save data as Pandas Series
    PheFlux = sol['x']
    PheFlux_fluxes = {}
    for num, i in enumerate (range(0, v.shape[0] , 2)):
        name = str(v[i])
        reaction_flux = ( PheFlux[i] - PheFlux[i+1] ) # (forward - reverse)
        PheFlux_fluxes[name] =  float(reaction_flux)
    PheFlux_fluxes = pd.Series(PheFlux_fluxes)

    return(PheFlux_fluxes,optimization_time,total_time,status,success, lbx, ubx)


# In[11]:


#################################################################################
######################### NON SPECIFIC TISSUE ###################################
#################################################################################
### Variables and objective function: CasADI object
def setVariables_NotSpecific(model,fpkmDic):
    v = vertcat() # saves the total of variables of the model. Used to """nlp['x']"""
    v_dic = {}
    v_fpkm = {} # 
    ubx, lbx = [],[]
    for i, reaction in enumerate(model.reactions):
        ##########################################################
        ## Gets the GPR and a boolean list with known or unknown genes
        rule = reaction.gene_reaction_rule # gene reaction rule
        boolean_list = booleanVectorRule(rule,fpkmDic) # useful to discriminate between genes with known fkpm.
        ##########################################################
        ## g = 1
        g = 1
        ##########################################################
        ## Set forward and reverse variables as a CasADI object
        # forward
        var_name = reaction.id
#         vf = SX.sym(var_name) # v forward
        expression = var_name+' = SX.sym("'+var_name+'")'
        exec(expression, globals())
        vf = eval(var_name)
        v = vertcat(v, vf)
        v_dic[reaction.id]=vf
        ubx.append(reaction.upper_bound)    
        lbx.append(0.0)

        # reverse
        var_name_reverse = reaction.reverse_id
#         vr = SX.sym(var_name_reverse) # v reverse
        expression = var_name_reverse+' = SX.sym("'+var_name_reverse+'")'
        exec(expression, globals())
        vr = eval(var_name_reverse)
        v = vertcat(v,vr)
        v_dic[reaction.reverse_id]=vr
        ubx.append(-reaction.lower_bound)
        lbx.append(0.0)

        v_fpkm[var_name] = g
        v_fpkm[var_name_reverse] = g
        ##########################################################
        ## Define a objective function
        for name in [vf,vr]:
            if i == 0:
                v_ViLogVi = ( (name)+1e-8 )*log( (name)+1e-8 ) # 1.1
                v_VilogQi = ( (name)+1e-8 )*log( g ) # 2.1
            else:
                v_ViLogVi += ( (name)+1e-8 )*log( (name)+1e-8 ) # 1.1
                v_VilogQi += ( (name)+1e-8 )*log( g ) # 2.1            
    ##############################################################
    ## Set objetive function
    f = (v_ViLogVi) - (v_VilogQi)
    return(v,v_dic,lbx,ubx,f)

#################################################################################
### OPTIMIZATION
def PheFlux_NotSpecific(model,fpkmDic,k,init_time):
    ##############################################################
    ## Sets variables, sumV and constraints
    v,v_dic,lbx,ubx,f = setVariables(model,fpkmDic)
    sumVi = getSumV(v)
    ubg,lbg,g = createConstraints(model,k,v_dic,sumVi)    
    print('')
    ##############################################################
    # Non-linear programming
    nlp = {}     # NLP declaration
    nlp['x']=  v # variables
    nlp['f'] = f # objective function
    nlp['g'] = g # constraints
    ##############################################################
    # Create solver instance
    F = nlpsol('F','ipopt',nlp)
    ##############################################################
    # Solve the problem using a guess
    fba_primal = getPrimalValues(model)
    x0=[]
    for i in range(v.shape[0]): # VARIABLES
        x0.append(fba_primal[str(v[i])])       
    ##############################################################
    ## Solver
    start = time.time()
    sol=F(x0=x0,ubg=ubg,lbg=lbg,lbx=lbx,ubx=ubx)
    final = time.time()
    total_time = final - init_time
    optimization_time = final - start
    status = F.stats()['return_status']
    success = F.stats()['success']
    ##############################################################
    ## Save data as Pandas Series
    PheFlux = sol['x']
    PheFlux_fluxes = {}
    for num, i in enumerate (range(0, v.shape[0] , 2)):
        name = str(v[i])
        reaction_flux = ( PheFlux[i] - PheFlux[i+1] ) # (forward - reverse)
        PheFlux_fluxes[name] =  float(reaction_flux)
    PheFlux_fluxes = pd.Series(PheFlux_fluxes)

    return(PheFlux_fluxes,optimization_time,total_time,status,success, lbx, ubx)


# In[12]:


#################################################################################
### Extract for experimental data
def computeMetrics(PheFlux_fluxes,c13File,uptakeRxn):
    ##############################################################
    ## Load C13 data  
    data_c13 = loadC13(c13File)  
    reactionID = []
    total_exp_fluxes = []
    for i in range(len(data_c13)):
        if "nan" in str(data_c13["Reaction_ID"][i]): continue
        if not ('Growth_rate' in data_c13["Reaction_ID"][i]):
            reactionID.append(data_c13["Reaction_ID"][i])
            total_exp_fluxes.append(data_c13["Flux"][i])
    ##############################################################
    ## Scale PheFlux fluxes
    predCSource = PheFlux_fluxes["R_"+uptakeRxn] # predicted flux of carbon source
    for i in range(len(data_c13)):
        if 'Growth_rate' in str(data_c13["Reaction_ID"][i]):
            growthRate = float(data_c13["Flux"][i])
        if uptakeRxn in str(data_c13["Reaction_ID"][i]):
            expCSource = data_c13["Flux"][i] # experimental flux of  carbon source
    beta =abs(expCSource / predCSource) # scale factor        
    pheflux_fluxes = []
    exp_fluxes = []
    name_fluxes = []
    for i, rxn in enumerate(reactionID):
        if 'R_'+rxn in PheFlux_fluxes:
            pheflux_fluxes.append(PheFlux_fluxes['R_'+rxn]*beta) #MODIFICATION
            exp_fluxes.append(total_exp_fluxes[i])
            name_fluxes.append(rxn)
    ##############################################################
    ## Pearson correlation 
    corr, pvalue = pearsonr(pheflux_fluxes, exp_fluxes)
    ##############################################################
    ## MSE  
    n = len(pheflux_fluxes) # or exp_fluxes
    sumdiffSq = 0
    for i in range(n):
        sumdiffSq += np.square( ( pheflux_fluxes[i]/(predCSource*beta) ) - ( exp_fluxes[i]/expCSource ) )
    MSE = ( 1/n ) * sumdiffSq
      
    return(corr,pvalue,MSE,pheflux_fluxes,exp_fluxes, name_fluxes)


# In[13]:


#################################################################################
### Figures
def doPlots(fpkmFile,MaxEnt_fluxes,maxent_fluxes, exp_fluxes,model,fpkmDic):
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig = plt.gcf()
    fig.set_size_inches(14, 4)

    fig.suptitle("Results for "+fpkmFile)
    ##############################################################
    ## Pearson correlation 
    ax1.scatter (maxent_fluxes, exp_fluxes)
    ax1.set_ylabel("Experimental fluxes, $w$ [mmol/g/h[]")
    ax1.set_xlabel("Predicted fluxes, $v \\cdot (w_{CS}/v_{CS})$ [mmol/g/h]")
    ax1.set_title("Predictions")
    ##############################################################
    ## Histogram of predicted flows 
    xx=[]
    for i in MaxEnt_fluxes:
        xx.append(i)
    ax2.hist(abs(np.array(xx)),bins=100,density=False)
    ax2.set_yscale("log")
    ax2.set_title("Fluxes")
    ax2.set_xlabel("Predicted fluxes, $v$ [mmol/g/h]")
    ax2.set_ylabel("Frequency")
    ##############################################################
    ## Histogram of gene expression in metabolism
    E_g,g_metab = getEg(model,fpkmDic)
    ax3.hist(g_metab,bins=150,density=False)
    ax3.set_yscale("log")
    ax3.set_title("Gene expression")
    ax3.set_xlabel("FPKM")
    ax3.set_ylabel("Frequency")

    fig.show()


# In[14]:


#################################################################################
### Table of validation results 
def createTable(results, dataSource,fpkmFile,corr,pvalue,MSE,MaxEnt_fluxes,optimization_time,total_time,status,success):

    countGRR,countIrr = 0,0
    for reaction in model.reactions:
        if reaction.gene_reaction_rule != "":
            countGRR +=1
        if reaction.lower_bound==0 or reaction.upper_bound==0:
            countIrr += 1
    rxnWGRR = countGRR/len(model.reactions)*100
    irrevRxns = countIrr/len(model.reactions)*100

    table_labels=["Data_source","Condition","Corr",
                  "Corr_value", "MSE", "Reactions_with_fpkm",
                  "Irreversible_reactions", "Max_flux",
                  "Optimization_time","Total_time","Status","Success"]
    table_values=[dataSource,fpkmFile,corr, 
                  pvalue,MSE,rxnWGRR,
                  irrevRxns,max(abs(MaxEnt_fluxes)),optimization_time,total_time, status, success ]

    if results.shape == (0,0):
        results = pd.DataFrame(columns=table_labels)  
    print(table_values)
    results.loc[results.shape[0]] = table_values

    # Table
    return( results )
    # export table
    #pd.DataFrame(table_values, columns=table_labels).to_excel('Ecoli_results.xlsx')


# In[15]:


#################################################################################
### Table of fluxes 
def fluxTable(scaled_fluxes, exp_fluxes, name_fluxes):
    ## Create a dataframe
    columns = ['Reaction','PHEFLUX', 'Experimental']
    df = pd.DataFrame(columns=columns)
    ## add values
    df['Reaction'] = name_fluxes
    df['PHEFLUX'] = scaled_fluxes
    df['Experimental'] = exp_fluxes
    
    return (df)


# In[16]:


#################################################################################
### Table of times and variable numbers for human network
def summaryTable(Summary ,condition, lbx, ubx, time, status):   
    variables = 0
    for i in range(len(lbx)):
        if lbx[i] != ubx[i]:
            variables += 1
    
    if Summary.shape == (0,0):
        Summary = pd.DataFrame(columns=['Condition', 'N° variables', 'Time', 'Status'])
        
    Summary.loc[Summary.shape[0]] = [condition, variables, time, status]
    return (Summary)


# In[23]:


#################################################################################
########################             PHEFLUX             ########################
#################################################################################
print('Hello world, Welcome to PheFlux ! \n')
processStart = time.time()

# Table of results
results = pd.DataFrame()
humanSummary = pd.DataFrame()
validationSummary = pd.DataFrame()
shuffle=False
shuffledFPKM = pd.DataFrame()
scaledSlns = pd.DataFrame()
#################################################################################
### Set a process: validation, tcga or gtex
# VALIDATION uses microorganism: S.cerevisiae, S.stipitis, Y.lipolytica, E.coli and B.subtilis
# TCGA or GTEx uses human (Homo sapiens) information: cancer and tissue-specific data
process = 'validation'

### select cancer type
if process == 'tcga':
    cancerType = 'Kidney' # Bronchus-Lung, Breast, Kidney or Brain
#################################################################################
### Load "InputData" file
if process == 'validation' or process == 'gtex':
    inputFileName = '../data/InputData.csv'
else: # tcga
    inputFileName = '../data/'+process+'/'+cancerType+'/InputData.csv'
inputData=pd.read_csv(inputFileName,sep="\t", lineterminator='\n', na_filter=False)
nRows,nCols=inputData.shape

shuffle=False
opt_time, t_time = [], []
for i in range(nRows):
    ##############################################################
    ## Load information from InputData
    condition    = inputData.loc[i]["Condition"]
    if process == 'validation' or process == 'gtex':
        geneExpFile  = '../data/'+process+'/'+inputData.loc[i]["GeneExpFile"]
        fluxDataFile = '../data/'+process+'/'+inputData.loc[i]["FluxDataFile"]
    else: # tcga
        geneExpFile  = '../data/'+process+'/'+cancerType+'/'+inputData.loc[i]["GeneExpFile"]
        fluxDataFile = '../data/'+process+'/'+cancerType+'/'+inputData.loc[i]["FluxDataFile"]
    uptakeRxns   = inputData.loc[i]["UptakeReactions"]
    network      = inputData.loc[i]["Network"]
    organism     = inputData.loc[i]["Organism"]
    
    
#     ##############################################################
#     ## non-specific tissue: gtex
#     if condition != 'NonSpecificTissue_54':continue # gtex non-specific tissue (g=1)


    ##############################################################
    ## TESTING RECON3D
    
#     network = 'Recon3D.xml'
    
    
    ##############################################################
    ## Messages in terminal
    print ('Condition ejecuted:', condition,'\n')    
    ##############################################################
    ## Set glucose uptake reaction      
    if network == 'RECON1.xml':
        defaultUptake = 'EX_glc__D_e'
    
    elif network == 'Recon3D.xml':
        defaultUptake = 'EX_glc__D_e'
    
    elif network == 'iMM904.xml':
        defaultUptake = 'EX_glc__D_e'
        
    elif network == 'iTL885.xml':
        defaultUptake = 'SS1232'
        
    elif network == 'iYali.xml':
        defaultUptake = 'y001808'
        
    elif network == 'iJO1366.xml':
        defaultUptake = 'EX_glc__D_e'
        
    elif network == 'iYO844.xml':
        defaultUptake = 'EX_glc__D_e'
    ##############################################################
    # Metabolic network
    model_default = cobra.io.read_sbml_model('../data/gems/'+network)
    fpkm = pd.read_csv(geneExpFile,sep="\t", lineterminator='\n') 
    ##############################################################
    init_time = time.time()
    # Load FPKM data
    fpkmDic,shuffledFPKM = loadFPKM(fpkm,condition,shuffle,shuffledFPKM)
    # Reload FPKM data for Hsapiens and load culture medium
    if process == 'tcga' or process == 'gtex':
        fpkmDic = reloadFPKMHsapiens(fpkmDic, model_default)
        #########################################################
        # load culture medium for tcga or gtex
        if process == 'gtex':
            mediumFile = '../data/'+process+'/'+uptakeRxns+'.csv'
            medium =  pd.read_csv(mediumFile,sep="\t", lineterminator='\n')
        elif process == 'tcga': # tcga
            mediumFile = '../data/'+process+'/'+cancerType+'/'+uptakeRxns+'.csv'
            medium =  pd.read_csv(mediumFile,sep="\t", lineterminator='\n')
            
            
    else: # validation
        medium = '?' ###############################################################################
        
        
    ##############################################################
    # Update model: Add R_, open bounds, and set carbon source 
    model = updateModel(model_default,uptakeRxns,process,defaultUptake,medium)
    ##############################################################
    # Compute flux predictions    
    k = 1000
    if condition != 'NonSpecificTissue_54':
        fluxes,optimization_time,total_time,status,success,lbx,ubx = PheFlux(model,fpkmDic,k,init_time)
    else:
        fluxes,optimization_time,total_time,status,success,lbx,ubx = PheFlux_NotSpecific(model,fpkmDic,k,init_time)
    ##############################################################
    # Validation process: correlation, MSE, plots and results
    if process == 'validation':
        # Compute metrics (pearson, MSE)
        uptakeRxn = uptakeRxns.split(',')[0]
        corr,pvalue,MSE,scaled_fluxes,exp_fluxes,name_fluxes = computeMetrics(fluxes,fluxDataFile,uptakeRxn)

        # Show plots and save them
        doPlots(condition,fluxes,scaled_fluxes, exp_fluxes,model,fpkmDic)
        scaledSlns[condition]=fluxes
        
        # Save all fluxes (not scaled)
        allfluxesFile = '../results/validation/pheflux_fluxes/allFluxes'+organism+'_'+condition+'.csv'
#         fluxes.to_csv(allfluxesFile, sep='\t')
        
        # Save scaled fluxes
        phefluxes = fluxTable(scaled_fluxes, exp_fluxes, name_fluxes)
        fluxFile = '../results/validation/pheflux_fluxes/'+organism+'_'+condition+'.csv'
#         phefluxes.to_csv(fluxFile, sep='\t', index=None)

        # Save results to table
        results = createTable(results, organism,condition,corr,pvalue,MSE,fluxes,optimization_time,total_time,status,success)
#         results.to_csv('../results/validation/PHEFLUX.csv', sep='\t', index=None)

        # summary table: times and variables
        summaryFile = '../results/'+process+'/PHEFLUX_summaryTable.csv'
        validationSummary = summaryTable(validationSummary ,condition, lbx, ubx, total_time, status)
#         validationSummary.to_csv(summaryFile, sep='\t')

    ##############################################################
    # Save results for Homo sapiens
    elif process == 'gtex':
        # fluxes
        network = network.split('.')[0].upper()
        resultsFile = '../results/'+process+'/'+network+'/'+condition+'_'+status+'.csv'
#         fluxes.to_csv(resultsFile, sep='\t')
        # summary table
        summaryFile = '../results/'+process+'/'+network+'/summaryTable.csv'
        humanSummary = summaryTable(humanSummary ,condition, lbx, ubx, total_time, status)
#         humanSummary.to_csv(summaryFile, sep='\t')
        
    else: # tcga
        network = network.split('.')[0].upper()
        resultsFile = '../results/'+process+'/'+network+'/'+process+'_'+cancerType+'/'+condition+'_'+status+'.csv'
#         fluxes.to_csv(resultsFile, sep='\t')
        # summary table: times and variables
        summaryFile = '../results/'+process+'/'+network+'/'+process+'_'+cancerType+'/summaryTable.csv'
        humanSummary = summaryTable(humanSummary ,condition, lbx, ubx, total_time, status)
#         humanSummary.to_csv(summaryFile, sep='\t')
    ##############################################################
    ## Messages in terminal
    opt_time.append(optimization_time)
    t_time.append(total_time)
    print('\n\n*', condition, "... is processed.")  
    
    print ('\n',' o '.center(108, '='),'\n')
#     break
    
#     ##################################################################
#     ## Saves results for Homo sapiens
    
#     # Total time per condition
#     if not len(t_time) == 1 and process != 'validation':
#         if process == 'gtex':
#             network = network.split('.')[0].upper()
#             timesFile = '../results/'+process+'/'+network+'/ejecutionTimes.csv'
#         elif process == 'tcga':
#             network = network.split('.')[0].upper()
#             timesFile = '../results/'+process+'/'+network+'/'+process+'_'+cancerType+'/ejecutionTimes.csv'

#         with open(timesFile, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerows([t_time])
    
#########################################################################################
processFinish = time.time()
processTime = processFinish - processStart
print ('Average time per optimization:', np.mean(opt_time), 's')
print ('Average time per condition:', np.mean(t_time), 's')
print ('Total process time:', processTime/60, 'min', '--> ~', (processTime/3600), 'h')


# In[ ]:


# resultsFile


# In[ ]:


# '../results/'+process+'/'+network+'/'+process+'_'+cancerType+'/'+condition+'_'+status+'.csv'


# In[ ]:


# inputData[:5]


# In[ ]:


# condition, geneExpFile, fluxDataFile, uptakeRxns, network, organism


# In[ ]:


# model_default


# In[ ]:


# model


# ___
# ### ¿¿ Cambios por la versión de COBRApy ??
# 
# Ahora no es necesaria la función ```correctedRevExn()``` y tampoco los ```'R_'``` en ```createConstraints()```.
# 
# Son cambios para mejor, terminan simplificando más el código.

# In[ ]:


# model_default.reactions[:3]


# In[ ]:


# print(model_default.metabolites[1].constraint.expression)


# In[ ]:


# model_default.reactions.get_by_id('IGPS').reverse_id


# In[ ]:


# ---


# In[ ]:


# model.reactions[:3]


# In[ ]:


# print(model.metabolites[1].constraint.expression)


# In[ ]:


# model.reactions.get_by_id('R_IGPS').reverse_id


# ___

# In[ ]:


# fpkm[:5]


# In[ ]:


# fpkmDic


# In[ ]:


# medium


# In[ ]:


# model.medium


# In[ ]:


# optimization_time, total_time, status,success


# In[ ]:


# fluxes


# In[ ]:


# fluxes['R_EX_lac__L_e'], fluxes['R_EX_lac__D_e']


# In[24]:


# results


# In[ ]:


# humanSummary


# In[25]:


validationSummary


# In[ ]:


# results['Corr'].min(), results['Corr'].max()


# In[ ]:


# results['MSE'].min(), results['MSE'].max()


# ___
# ___
# **CONSTRAINTS**

# In[ ]:


# from casadi import *
# import cobra
# from cobra.flux_analysis import flux_variability_analysis
# from scipy.stats import entropy
# import numpy as np
# import pandas as pd
# import csv
# import json
# import time
# from matplotlib import pyplot as plt
# from scipy.stats import pearsonr


# In[ ]:


# model = cobra.io.read_sbml_model('../data/gems/Recon3D.xml')


# In[ ]:


# v_vars, rev_vars = [], []
# for reaction in model.reactions:
#     v_vars.append(reaction.id)
#     rev_vars.append(reaction.reverse_id)  
# # return(v_vars,rev_vars)


# In[ ]:


# #################################################################################
# ### Creating constraints
# # def createConstraints(model,k,v_dic,sumVi):
# g = vertcat()
# lbg,ubg=[],[]
# ##############################################################
# ## Gets the name of the forward/reverse variables
# # v_vars, rev_vars = getFowardReverse(model)
# ##############################################################
# ## Defines constraints
# for i, met in enumerate(model.metabolites):
#     ##########################################################
#     ## Gets constraint for a one metabolite
# #     tmp_constraint = 0
#     constraint = str(met.constraint.expression).replace('+ ','+').replace('- ','-')
#     print('constraint:', constraint,'\n')
#     ##########################################################
#     ## Reconstruct the constraint as a CasADI object
#     for j, field in enumerate(constraint.split()):
#         print('field:', field)
# #         if field == '-' or field == '+':
# #             connector = field        
# #         else: 
# #             if reaction in v_vars:
# #                 field = re.sub(r'\b'+reaction+r'\b', str(reaction), field)

# #             elif reaction in rev_vars:
# #                 field = re.sub(r'\b'+reaction+r'\b', str(reaction), field)

#         if j == 0:
#             tmp_constraint = eval(field)
#         else:
# #                 tmp_constraint.append(connector)
#             tmp_constraint += eval(field)
    
#     ##########################################################
#     ## Adds constraint to CasADI
# #     tmp_constraint = ' '.join(tmp_constraint)
# #         g = vertcat(g,eval(tmp_constraint,v_dic))
#         g = vertcat(g,tmp_constraint)
#     lbg.append(0)
#     ubg.append(0)
#     ##############################################################
#     print('\n', tmp_constraint)
#     break


# In[ ]:


# constraint.split()


# In[ ]:




